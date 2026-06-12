# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AtroposGRPOTrainer
==================
A subclass of TRL's GRPOTrainer that replaces the rollout-generation layer with
Atropos as the trajectory source.  Any Atropos environment that implements the
ManagedServer API (i.e. returns `tokens`, `masks`, and `inference_logprobs` per
trajectory node) can be used as a drop-in rollout provider for this trainer.

Architecture
------------
This trainer uses the TRL vLLM server (started by `trl vllm-serve`) as the
sole vLLM backend.

The Atropos environment microservices are configured to call the same TRL
vLLM server (`--openai.base_url http://<host>:<port>/v1`) for generation.
The trainer owns the vLLM server's weights via TRL's native
`VLLMGeneration.sync_weights()` mechanism.

Data-flow overview
------------------
                   ┌──────────────────────────────────────┐
                   │  TRL vLLM server (trl vllm-serve)    │
                   │  :8001  (started automatically or    │
                   │  pointed to via vllm_server_*)        │
                   └──────────┬───────────────────────────┘
                              │  OpenAI-compatible /v1 API
                   ┌──────────▼───────────────────────────┐
                   │  Atropos env microservice(s)          │
                   │  • generates completions via TRL vLLM │
                   │  • scores trajectories                │
                   │  • POST /scored_data to Atropos API   │
                   └──────────┬───────────────────────────┘
                              │  POST /scored_data
                   ┌──────────▼───────────────────────────┐
                   │  Atropos API server  (:8000)          │
                   └──────────┬───────────────────────────┘
                              │  GET /batch
                   ┌──────────▼───────────────────────────┐
                   │  AtroposGRPOTrainer                   │
                   │  1. _convert_atropos_batch() → tensors│
                   │  2. compute_loss()  (from GRPOTrainer)│
                   │  3. optimizer step (from GRPOTrainer) │
                   │  4. vllm_generation.sync_weights()    │
                   │     (pushes weights → TRL vLLM server)│
                   └──────────────────────────────────────┘

Training loop
-------------
1. `train()` is called as normal.
2. Each `_prepare_inputs()` call *ignores* the HuggingFace dataset batch
   and instead polls the Atropos API until a ready batch is available.
3. The fetched batch is converted to the same tensor dict that
   `_generate_and_score_completions` normally returns, so `compute_loss`
   and all downstream logging code are completely unmodified.
4. Weight sync back to the TRL vLLM server is triggered inside our
   ``_prepare_inputs`` override at exactly the same cadence that the
   base class would sync (once per ``generate_every`` window).  The
   parent's ``use_vllm=True`` + ``vllm_mode="server"`` wiring is reused
   in full — we just call ``vllm_generation.sync_weights()`` ourselves
   because we bypass ``_generate_and_score_completions``.

Atropos batch contract (from /batch endpoint)
----------------------------------------------
The Atropos API returns a list of *group* dicts.  Each group dict contains
multiple parallel sequences that form one prompt group.  The trainer flattens
these groups into individual trajectories internally.

Each group dict from the API contains:
  tokens              : list[list[int]]   – full sequences (prompt + completion)
  masks               : list[list[int]]   – -100 for prompt, token_id for completion
  inference_logprobs  : list[list[float]] – log-probs for each completion token
  scores              : list[float]       – reward per trajectory in the group
  env_id              : int               – source environment id

Optional fields forwarded to logging:
  messages            : list              – full message history from the env

Advantages are computed by this trainer using group-relative normalisation
(identical to the standard GRPO formula) so that reward normalisation is
always consistent regardless of the environment.

Usage
-----
.. code-block:: python

    from atropos_grpo_trainer import AtroposGRPOTrainer, AtroposGRPOConfig

    config = AtroposGRPOConfig(
        output_dir="./my_run",
        per_device_train_batch_size=4,
        num_generations=8,
        # TRL vLLM server settings (same server the Atropos env will call)
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=8001,
        # Atropos API settings
        atropos_api_url="http://localhost:8000",
        atropos_group_size=8,
        max_steps=1000,
    )

    trainer = AtroposGRPOTrainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        args=config,
    )
    trainer.train()

Environment setup
-----------------
1. Start the TRL vLLM server (it serves *both* the trainer weight-sync
   and the Atropos environments):

    trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8001

2. Start the Atropos API server:

    run-api   # default :8000

3. Start one or more Atropos environments pointed at the TRL vLLM server:

    python environments/gsm8k_server.py serve \\
        --openai.model_name Qwen/Qwen2.5-1.5B-Instruct \\
        --openai.base_url http://localhost:8001/v1 \\
        --env.group_size 8 \\
        --slurm false

4. Start the trainer (use_vllm=True, vllm_mode="server"):

    python train_atropos.py
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, cast

import torch
from torch.utils.data import DataLoader

from trl.trainer import GRPOTrainer
from .atropos_grpo_config import AtroposGRPOConfig
from .atropos_api_client import AtroposAPIClient
from .placeholders import _AtroposPlaceholderDataset, _atropos_passthrough_reward
from trl.trainer.utils import pad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class AtroposGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that sources rollouts from an Atropos RL environment
    rather than generating them in-process.

    Key differences from the base GRPOTrainer
    ------------------------------------------
    * ``_prepare_inputs`` is overridden to fetch pre-scored trajectories
      from the Atropos API server instead of calling
      ``_generate_and_score_completions``.
    * Advantage computation uses the same group-relative GRPO normalisation
      so that all downstream loss code (``compute_loss``) is unchanged.
    * Weight sync back to the TRL vLLM server is triggered inside our
      ``_prepare_inputs`` override at exactly the same cadence that the
      base class would sync (once per ``generate_every`` window).  The
      parent's ``use_vllm=True`` + ``vllm_mode="server"`` wiring is reused
      in full — we just call ``vllm_generation.sync_weights()`` ourselves
      because we bypass ``_generate_and_score_completions``.
    * ``training_step``, ``compute_loss``, logging, and all other parent
      functionality are completely unmodified.
    * ``train_dataset`` is optional: if ``None`` is provided a lightweight
      placeholder dataset is created automatically.

    Design notes / known limitations
    --------------------------------
    * Advantage computation: ``_convert_atropos_batch`` applies a simplified
      group-relative GRPO normalisation (``(score - mean) / (std + 1e-4)``).
      This matches the behaviour of the parent class with default settings
      (``multi_objective_aggregation="sum_then_normalize"`` and
      ``scale_rewards="group"``).  If you change ``multi_objective_aggregation``
      or ``scale_rewards`` in the config, the advantage computation will NOT
      use those settings — the parent ignores the ``advantages`` tensor and
      just applies the loss using the pre-computed advantages.  For custom
      advantage aggregation, override ``_convert_atropos_batch``.

    * Evaluation: When ``eval_dataset`` is provided, evaluation uses the
      standard ``GRPOTrainer._generate_and_score_completions`` path (in-process
      generation without Atropos).  This is intentional — evaluation should
      reflect the model's independent ability.  If you need evaluation to also
      use Atropos, override ``_prepare_inputs`` and remove the ``mode == "eval"``
      early-return.

    * ``num_items_in_batch`` (used by DAPO/VESPO loss normalisation) uses the
      local per-process count rather than a cross-process gather, because each
      process independently fetches batches from the Atropos API at different
      times and an ``accelerator.gather`` collective would deadlock.  For the
      DAPO loss, this means the loss is normalised by the per-process token
      count rather than the global token count; the error is bounded because
      the per-process losses are averaged across GPUs by the optimizer
      all-reduce.
    """

    def __init__(
        self,
        model,
        args: Optional[AtroposGRPOConfig] = None,
        reward_funcs=None,
        train_dataset=None,
        eval_dataset=None,
        **kwargs,
    ):
        # ------------------------------------------------------------------ #
        # Validate / create config                                             #
        # ------------------------------------------------------------------ #
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = AtroposGRPOConfig(
                output_dir=f"{model_name.split('/')[-1]}-atropos-grpo"
            )

        if not isinstance(args, AtroposGRPOConfig):
            raise TypeError(
                "AtroposGRPOTrainer requires an AtroposGRPOConfig instance, "
                f"got {type(args).__name__}. Either use AtroposGRPOConfig directly "
                "or promote your GRPOConfig: "
                "AtroposGRPOConfig(**dataclasses.asdict(your_config))"
            )

        # Atropos provides rewards; a no-op pass-through reward_func satisfies
        # the TRL base-class requirement without interfering with Atropos scores.
        # The advantages tensor we inject in _convert_atropos_batch already
        # contains the normalised rewards, so the base-class reward pipeline
        # output is harmlessly overridden.
        if reward_funcs is None:
            reward_funcs = _atropos_passthrough_reward
        else:
            logger.warning(
                "AtroposGRPOTrainer: custom reward_funcs were provided, but when using "
                "Atropos trajectory fetching, rewards are produced by the environment "
                "and the advantages tensor injected by _convert_atropos_batch overrides "
                "the base-class reward pipeline output.  Custom reward_funcs will be "
                "applied during evaluation (which uses the in-process generation path) "
                "but are effectively ignored during training."
            )

        # Create a lightweight placeholder dataset when none is given.
        # _prepare_inputs ignores it completely.
        if train_dataset is None:
            total_steps = getattr(args, "max_steps", None) or 10_000
            train_dataset = _AtroposPlaceholderDataset(total_steps)

        # Warn if the user has set multi_objective_aggregation or scale_rewards
        # to non-default values, since AtroposGRPOTrainer hardcodes the GRPO
        # advantage computation and those settings will have no effect.
        if getattr(args, "multi_objective_aggregation", None) not in (None, "sum_then_normalize"):
            logger.warning(
                "AtroposGRPOTrainer: multi_objective_aggregation is set to '%s', but "
                "the advantage computation in _convert_atropos_batch hardcodes "
                "group-relative normalisation (equivalent to 'sum_then_normalize' "
                "with scale_rewards='group').  The setting will be ignored. "
                "Override _convert_atropos_batch for custom advantage aggregation.",
                args.multi_objective_aggregation,
            )
        if getattr(args, "scale_rewards", None) not in (None, "group"):
            logger.warning(
                "AtroposGRPOTrainer: scale_rewards is set to '%s', but the advantage "
                "computation in _convert_atropos_batch hardcodes group-relative "
                "normalisation.  The setting will be ignored. "
                "Override _convert_atropos_batch for custom scaling.",
                args.scale_rewards,
            )

        # ------------------------------------------------------------------ #
        # Delegate to GRPOTrainer.__init__ with use_vllm=True enforced.      #
        # The parent will create self.vllm_generation which we reuse for      #
        # sync_weights() after each training window.                          #
        # ------------------------------------------------------------------ #
        if not args.use_vllm:
            logger.warning(
                "AtroposGRPOTrainer: use_vllm is False but Atropos environments "
                "need a running TRL vLLM server.  Forcing use_vllm=True.  "
                "Set vllm_mode='server' and point vllm_server_host/port at the "
                "server started by `trl vllm-serve`."
            )
            args.use_vllm = True

        super().__init__(
            model=model,
            args=args,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # Narrow the type of self.args for PyLance: the parent class types it as
        # TrainingArguments, but we know it's AtroposGRPOConfig (which inherits
        # steps_per_generation from GRPOConfig).
        self.args = cast(AtroposGRPOConfig, self.args)

        # ------------------------------------------------------------------ #
        # Atropos API client                                                   #
        # ------------------------------------------------------------------ #
        self._atropos_args: AtroposGRPOConfig = args
        self._atropos_client = AtroposAPIClient(
            base_url=args.atropos_api_url,
            trainer_id=args.atropos_trainer_id,
            timeout=args.atropos_batch_timeout,
            poll_interval=args.atropos_poll_interval,
            max_retries=args.atropos_max_retries,
        )

        # ------------------------------------------------------------------ #
        # State                                                                #
        # ------------------------------------------------------------------ #
        self._atropos_registered: bool = False
        self._atropos_group_size: int = args.atropos_group_size
        # Cache the last fetched raw batch for potential debugging / logging
        self._last_raw_batch: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------------------- #
    # Registration                                                             #
    # ---------------------------------------------------------------------- #

    def _ensure_registered(self) -> None:
        """
        Register with the Atropos API exactly once, on the first call.

        Sends the full Registration schema expected by the Atropos API
        server, mapping trainer config fields to server-side keys.

        IMPORTANT: batch_size is the number of groups (not trajectories)
        per process per fetch.  The Atropos API server treats batch_size as
        the number of queue items (each queue item is one group containing
        ``group_size`` trajectories) to extract per batch call.  We send
        ``per_device_train_batch_size`` as the group count; the server will
        deliver that many groups, i.e.
        ``per_device_train_batch_size × atropos_group_size`` trajectories
        per fetch.
        """
        if self._atropos_registered:
            return

        if not self._atropos_client.health():
            raise ConnectionError(
                f"Cannot reach Atropos API at {self._atropos_args.atropos_api_url}. "
                "Ensure `run-api` is running before starting the trainer."
            )

        # Build the full Registration schema expected by the Atropos API server.
        # This includes all required fields from server.py::Registration.
        # NOTE: batch_size here means "number of groups per fetch" because the
        # API server's queue stores one dict per group (see server.py line 234).
        # The server's grab_exact_from_heterogeneous_queue uses batch_size to
        # determine how many queue items to pop per /batch call.
        # NOTE: max_token_len is used by the Atropos API server for environment
        # weighting calculations (see server.py status-env endpoint).  It should
        # represent the maximum total sequence length (prompt + completion), not
        # just the completion length.  We use max_completion_length * 2 as a
        # generous upper bound on prompt + completion, since the prompt can be
        # as long as the completion in practice.
        max_completion_length = getattr(self.args, "max_completion_length", 2048)
        max_total_token_len = max_completion_length * 2
        reg_payload: Dict[str, Any] = {
            "wandb_group": os.path.basename(self.args.output_dir),
            "wandb_project": "trl-atropos",
            # batch_size = number of groups per process per fetch.
            # Each group contains atropos_group_size trajectories.
            # Total trajectories per fetch = per_device_train_batch_size * atropos_group_size.
            "batch_size": self.args.per_device_train_batch_size,
            "max_token_len": max_total_token_len,
            "checkpoint_dir": self.args.output_dir,
            "save_checkpoint_interval": getattr(self.args, "save_steps", 500),
            "starting_step": self.state.global_step,
            "num_steps": getattr(self.args, "max_steps", 1000),
        }

        info = self._atropos_client.register(**reg_payload)
        logger.info("Registered with Atropos API. Server info: %s", info)
        self._atropos_registered = True

    # ---------------------------------------------------------------------- #
    # Core override: replace in-process rollout with Atropos batch fetch      #
    # ---------------------------------------------------------------------- #

    def _prepare_inputs(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Override the base class _prepare_inputs to fetch trajectories from
        Atropos instead of generating them in-process.

        During training we:
          1. Ensure we're registered with the Atropos API server.
          2. When it is time to fetch a new window
             (every ``steps_per_generation * num_iterations`` micro-steps):
             a. Sync current model weights to the TRL vLLM server so the
                Atropos environments pick up the latest policy.
             b. Poll /batch until a scored batch is available.
             c. Convert the Atropos payload to the tensor dict expected by
                ``compute_loss``.
             d. Split and buffer the result identically to the base class so
                that ``steps_per_generation`` and ``num_iterations`` mechanics
                work unchanged.
          3. Return the correct slice for the current micro-step.

        During evaluation we fall back to the standard in-process path
        (calling ``super()._prepare_inputs``) so that eval still uses the
        model directly.
        """
        mode = "train" if self.model.training else "eval"

        if mode == "eval":
            # For evaluation, use the standard TRL path.
            return super()._prepare_inputs(generation_batch)

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # ----------------------------------------------------------------
            # Sync weights to the TRL vLLM server BEFORE fetching the next
            # batch so that Atropos environments use the up-to-date policy.
            # This mirrors exactly what _generate_and_score_completions does
            # when use_vllm=True.
            # ----------------------------------------------------------------
            self._sync_weights_to_vllm()

            # Fetch a fresh scored batch from the Atropos API
            raw_batch = self._atropos_client.wait_for_batch()
            self._last_raw_batch = raw_batch

            prepared = self._convert_atropos_batch(raw_batch)

            # Mirror the base class split-and-buffer logic so that
            # steps_per_generation sub-batches are all used before the
            # next fetch (same schedule as the parent's buffering).
            from trl.trainer.utils import split_tensor_dict
            batches = split_tensor_dict(prepared, self.args.steps_per_generation)
            self._buffered_inputs = batches

        inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        return inputs

    # ---------------------------------------------------------------------- #
    # Weight sync to TRL vLLM server                                          #
    # ---------------------------------------------------------------------- #

    def _sync_weights_to_vllm(self) -> None:
        """
        Push the current model weights to the TRL vLLM server.

        This reuses the parent class's ``VLLMGeneration.sync_weights()``
        exactly.  We call it ourselves because we bypass
        ``_generate_and_score_completions``, which is normally where the
        parent triggers the sync.

        Uses ``self.state.global_step`` (which tracks optimizer steps) rather
        than ``self._step`` (the per-micro-step counter) for the sync guard,
        matching the parent class's behaviour.

        IMPORTANT: In distributed mode, ``sync_weights()`` is a NCCL
        collective that all processes must call simultaneously.  We use
        ``accelerator.wait_for_everyone()`` before syncing to ensure all
        processes are at the same generation window boundary.  This is safe
        because all processes reach this point when their local ``_step``
        hits the ``generate_every`` boundary, which happens at the same
        logical optimizer step (though not necessarily the same wall-clock
        time, since each process independently fetches batches from the
        API server).  The barrier ensures collective alignment.
        """
        if self.state.global_step != self._last_loaded_step:
            # Barrier to ensure all processes are at the same logical sync
            # boundary before calling the NCCL collective sync_weights().
            self.accelerator.wait_for_everyone()
            try:
                self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step
                logger.debug(
                    "Synced weights to TRL vLLM server at step %d",
                    self.state.global_step,
                )
            except Exception as exc:
                logger.warning(
                    "Weight sync to TRL vLLM server failed at step %d: %s",
                    self.state.global_step,
                    exc,
                )

    # ---------------------------------------------------------------------- #
    # Atropos payload → TRL tensor dict conversion                            #
    # ---------------------------------------------------------------------- #

    def _convert_atropos_batch(
        self, raw_batch: List[Dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Convert an Atropos batch (list of group dicts from the API server)
        into the tensor dict expected by ``compute_loss``.

        The Atropos API returns group-oriented data where each dict contains
        parallel lists of sequences.  This method flattens the groups into
        individual trajectories.

        Atropos API server fields (from _scored_data_to_dict in server.py)
        -------------------------------------------------------------------
        tokens              : list[list[int]]   – parallel sequences (prompt+completion)
        masks               : list[list[int]]   – -100 for prompt, token_id for completion
        inference_logprobs  : list[list[float]] – log-probs for completion tokens
        scores              : list[float]       – reward per sequence in the group
        env_id              : int               – source environment identifier

        This produces the same output shape as ``_generate_and_score_completions``
        so all downstream loss computation, logging, and metric code is completely
        unmodified.
        """
        device = self.accelerator.device
        group_size = self._atropos_group_size

        # ------------------------------------------------------------------ #
        # Step 0 – flatten group-oriented data into per-trajectory items      #
        # ------------------------------------------------------------------ #
        # Each item in raw_batch is a group dict from the API server.
        # We flatten: for each group, zip the parallel lists to produce one
        # record per trajectory.

        trajectories: List[Dict[str, Any]] = []
        env_id_counter: Counter = Counter()

        for group_idx, group_item in enumerate(raw_batch):
            # Extract fields with meaningful error messages.
            try:
                tokens_list: List[List[int]] = group_item["tokens"]
                masks_list: List[List[int]] = group_item["masks"]
            except KeyError as e:
                raise KeyError(
                    f"_convert_atropos_batch: missing field {e} in group item {group_idx}. "
                    "The Atropos API server /batch endpoint must return dicts with "
                    "'tokens' and 'masks' keys. See the Atropos batch contract in "
                    "ATROPOS_GRPO_TRAINER.md."
                ) from e

            # inference_logprobs is optional in ScoredData; default to 0.0 sentinel.
            # We use 0.0 as default because real log-probabilities are always ≤ 0
            # (log of probability ≤ 1).  A value of 0.0 means the token had
            # probability 1.0 in the sampling distribution, which won't happen in
            # practice with temperature > 0, making 0.0 a safe sentinel that won't
            # bias importance-sampling ratios.
            logprobs_list: List[List[float]] = group_item.get(
                "inference_logprobs",
                [[0.0] * len(seq) for seq in tokens_list],
            )

            # scores is a list of floats, one per trajectory in the group.
            scores_list: List[float] = group_item.get(
                "scores",
                [0.0] * len(tokens_list),
            )

            env_id = group_item.get("env_id")
            env_id_str = str(env_id) if env_id is not None else "unknown"
            env_id_counter[env_id_str] += len(tokens_list)

            # Validate shapes within the group.
            seq_count = len(tokens_list)
            if not (len(masks_list) == seq_count and len(logprobs_list) == seq_count and len(scores_list) == seq_count):
                raise ValueError(
                    f"_convert_atropos_batch: group item {group_idx} has mismatched sequence counts: "
                    f"tokens={seq_count}, masks={len(masks_list)}, "
                    f"inference_logprobs={len(logprobs_list)}, scores={len(scores_list)}. "
                    "All fields must contain the same number of trajectories."
                )

            for seq_idx in range(seq_count):
                trajectories.append({
                    "tokens": tokens_list[seq_idx],
                    "masks": masks_list[seq_idx],
                    "logprobs": logprobs_list[seq_idx],
                    "score": scores_list[seq_idx],
                })

        # Log env_id distribution for multi-environment awareness.
        if len(env_id_counter) > 1:
            logger.info(
                "Atropos batch contains trajectories from %d environments: %s",
                len(env_id_counter),
                dict(env_id_counter),
            )

        num_trajectories = len(trajectories)

        if num_trajectories == 0:
            raise ValueError(
                "Atropos batch is empty after flattening. "
                "The /batch endpoint returned group items, but none contained any sequences. "
                "Please verify the Atropos environment is producing valid trajectory data."
            )

        if num_trajectories % group_size != 0:
            raise ValueError(
                f"Atropos batch size {num_trajectories} is not divisible by "
                f"group_size {group_size}.  Ensure the Atropos environment "
                "group_size matches atropos_group_size in AtroposGRPOConfig. "
                f"env_id distribution: {dict(env_id_counter)}"
            )

        # ------------------------------------------------------------------ #
        # Step 1 – split tokens into prompt / completion                       #
        # ------------------------------------------------------------------ #
        # The Atropos API server uses -100 in masks to mark prompt positions.

        prompt_ids_list: List[List[int]] = []
        completion_ids_list: List[List[int]] = []
        completion_logps_list: List[List[float]] = []
        scores_list: List[float] = []

        for traj in trajectories:
            tokens: List[int] = traj["tokens"]
            masks: List[int] = traj["masks"]
            logprobs: List[float] = traj["logprobs"]

            if not (len(tokens) == len(masks) == len(logprobs)):
                raise ValueError(
                    f"_convert_atropos_batch: trajectory has mismatched lengths: "
                    f"tokens={len(tokens)}, masks={len(masks)}, "
                    f"logprobs={len(logprobs)}. "
                    "All three fields must have the same length."
                )

            # Find where the completion begins: first position where masks != -100.
            completion_start = next(
                (i for i, m in enumerate(masks) if m != -100),
                len(tokens),
            )

            prompt_ids_list.append(tokens[:completion_start])
            completion_ids_list.append(tokens[completion_start:])

            # Atropos uses 0.0 as a sentinel for "masked / invalid" logprob
            # (see the default value in the extraction loop above).
            # Slice only the completion portion and filter out the sentinel
            # values so they don't contaminate the importance-sampling ratio.
            # Real log-probabilities are always negative (log of probability ≤ 1),
            # so a value == 0.0 is unambiguously a sentinel.
            raw_lps = logprobs[completion_start:]
            # Replace sentinel 0.0 logprobs (meaning "not provided") with a very
            # negative value so they do not bias the importance-sampling ratio.
            # Real log-probabilities are always ≤ 0 (log of probability ≤ 1),
            # so 0.0 is unambiguously a sentinel indicating the field was absent
            # in the source data.  A value of -100.0 gives exp(-100) ≈ 3.7e-44,
            # effectively zero weight in the importance-sampling ratio.
            completion_logps_list.append(
                [lp if lp != 0.0 else -100.0 for lp in raw_lps]
            )

            scores_list.append(float(traj["score"]))

        # ------------------------------------------------------------------ #
        # Step 2 – GRPO advantage computation                                  #
        # ------------------------------------------------------------------ #
        # Scores arrive already assigned by the Atropos environment.
        # We apply group-relative normalisation to compute advantages,
        # matching the standard GRPO formula used by the parent class.

        scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
        num_groups = num_trajectories // group_size
        grouped = scores_tensor.view(num_groups, group_size)
        group_means = grouped.mean(dim=1, keepdim=True)
        group_stds = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = ((grouped - group_means) / (group_stds + 1e-4)).view(-1)

        # Log raw reward statistics (same keys as GRPOTrainer)
        mode = "train"
        self._metrics[mode]["reward"].append(scores_tensor.mean().item())
        self._metrics[mode]["reward_std"].append(scores_tensor.std().item())

        # ------------------------------------------------------------------ #
        # Step 3 – pad and tensorise                                           #
        # ------------------------------------------------------------------ #

        pad_id = self._tokenizer.pad_token_id

        # Prompt tensors (left-padded, matching GRPOTrainer convention)
        prompt_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids_list]
        prompt_ids = pad(
            prompt_tensors,
            padding_value=pad_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        prompt_mask = pad(
            [torch.ones_like(t) for t in prompt_tensors],
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        # Completion tensors (right-padded)
        completion_tensors = [
            torch.tensor(ids, dtype=torch.long) for ids in completion_ids_list
        ]
        completion_ids_t = pad(
            completion_tensors,
            padding_value=pad_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_mask = pad(
            [torch.ones_like(t) for t in completion_tensors],
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        # Per-token sampling log-probs from the vLLM policy that generated
        # the completions.  Used for the importance-sampling correction
        # between the inference policy (vLLM) and the current training policy.
        logp_tensors = [
            torch.tensor(lps, dtype=torch.float32) for lps in completion_logps_list
        ]
        sampling_per_token_logps = pad(
            logp_tensors,
            padding_value=0.0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)

        # ------------------------------------------------------------------ #
        # Step 4 – mask truncated completions (if configured)                  #
        # ------------------------------------------------------------------ #
        if self.mask_truncated_completions:
            eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # ------------------------------------------------------------------ #
        # Step 5 – old_per_token_logps, ref_per_token_logps, &                #
        #          importance_sampling_ratio in a single no_grad block         #
        # ------------------------------------------------------------------ #
        # When use_vllm=True, the parent class computes old_per_token_logps
        # because the vLLM inference log-probs (sampling_per_token_logps)
        # come from the vLLM backend while per_token_logps comes from the
        # PyTorch training model.  These can differ significantly even at
        # the same weight value due to different implementation details
        # (e.g. temperature scaling, float precision, kernel differences).
        #
        # We replicate that logic here so that the importance-sampling
        # correction in compute_loss works correctly.

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids_t], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids_t.size(1)

        old_per_token_logps: Optional[torch.Tensor] = None
        ref_per_token_logps: Optional[torch.Tensor] = None
        importance_sampling_ratio: Optional[torch.Tensor] = None

        generate_every = self.args.steps_per_generation * self.num_iterations

        with torch.no_grad():
            # ---- ref_per_token_logps (KL penalty, beta > 0) ---- #
            if self.beta != 0.0:
                ref_per_token_logps = self._compute_ref_logps(
                    prompt_ids=prompt_ids,
                    prompt_mask=prompt_mask,
                    completion_ids=completion_ids_t,
                    completion_mask=completion_mask,
                )
            # ---- old_per_token_logps (importance sampling correction) ---- #
            # Compute the PyTorch model's log-probs on the current weights
            # right after sync.  This gives us the "old" policy log-probs
            # for the importance-sampling ratio in compute_loss.
            if (
                self.args.gradient_accumulation_steps % generate_every != 0
                or (self.use_vllm and self.vllm_importance_sampling_correction)
            ):
                from trl.models.utils import disable_gradient_checkpointing
                with disable_gradient_checkpointing(
                    self.model, self.args.gradient_checkpointing_kwargs
                ):
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        self.args.per_device_train_batch_size,
                    )

            # ---- vLLM importance sampling ratio ---- #
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask

                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                is_ratio = torch.exp(logps_diff)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    is_ratio = torch.clamp(
                        is_ratio,
                        min=self.vllm_importance_sampling_clip_min,
                        max=self.vllm_importance_sampling_clip_max,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    min_val = (
                        self.vllm_importance_sampling_clip_min
                        if self.vllm_importance_sampling_clip_min is not None
                        else -float("inf")
                    )
                    max_val = (
                        self.vllm_importance_sampling_clip_max
                        if self.vllm_importance_sampling_clip_max is not None
                        else float("inf")
                    )
                    invalid_mis_mask = (is_ratio < min_val) | (is_ratio > max_val)
                    is_ratio = is_ratio.masked_fill(invalid_mis_mask, value=0.0)

                importance_sampling_ratio = is_ratio

        # ------------------------------------------------------------------ #
        # Step 6 – logging                                                     #
        # ------------------------------------------------------------------ #
        # Completion length metrics (same keys as GRPOTrainer)
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids_list], dtype=torch.float32
        )
        self._metrics[mode]["completion_length"].append(completion_lengths.mean().item())

        # Log rewards (use "atropos_env" as the reward source name)
        self._logs["rewards"]["atropos_env"].extend(scores_list)
        self._logs["advantages"].extend(advantages.tolist())

        # ------------------------------------------------------------------ #
        # Step 7 – assemble output dict (same shape as _generate_and_score)   #
        # ------------------------------------------------------------------ #
        # num_items_in_batch is used by DAPO / VESPO normalisation.
        # For other loss types it is unused but must be present.
        #
        # IMPORTANT: We use the local count (per-process), NOT a cross-process
        # gather.  The parent's _generate_and_score_completions uses a global
        # gather via accelerator.gather(), but that only works because all
        # processes call it simultaneously within a synchronized training step.
        # In Atropos mode each process independently fetches a batch from the
        # API server at different times, so a collective gather would deadlock.
        # Using the local count works correctly because:
        #   - For DAPO loss: loss normalises by global num_items_in_batch, but
        #     the loss is already scaled per-process; the per-process loss from
        #     each GPU is averaged by the optimizer all-reduce.
        #   - For other loss types: num_items_in_batch is unused.
        local_num_items = int(completion_mask.sum().item())
        num_items_in_batch = local_num_items

        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids_t,
            "completion_mask": completion_mask,
            "advantages": advantages.to(device),
            "num_items_in_batch": num_items_in_batch,
            # Sampling log-probs for the importance-sampling correction
            # between the vLLM inference policy and the current train policy.
            "sampling_per_token_logps": sampling_per_token_logps,
        }

        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps

        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio

        return output

    # ---------------------------------------------------------------------- #
    # Reference model logprob computation                                     #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _compute_ref_logps(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities under the reference model for
        the KL penalty term when beta > 0.

        Text-only variant: no multimodal arguments are passed through since
        the Atropos trainer sources data purely from the API (token IDs only).

        Re-uses the parent class ``_get_per_token_logps_and_entropies``
        which correctly handles PEFT (adapter swap), DeepSpeed, and FSDP.

        Args:
            prompt_ids: Left-padded prompt token IDs.
            prompt_mask: Left-padded prompt attention mask.
            completion_ids: Right-padded completion token IDs.
            completion_mask: Right-padded completion attention mask.
        """
        from trl.models.utils import disable_gradient_checkpointing

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size

        with disable_gradient_checkpointing(
            self.model, self.args.gradient_checkpointing_kwargs
        ):
            if self.ref_model is not None:
                ref_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                # PEFT path: temporarily disable the active adapter to get
                # base-model (reference) probabilities.
                from accelerate.utils import is_peft_model
                from trl.trainer.utils import use_adapter

                model = self.accelerator.unwrap_model(self.model)
                adapter_name = (
                    "ref"
                    if "ref" in getattr(model, "peft_config", {})
                    else None
                )
                with use_adapter(model, adapter_name=adapter_name):
                    ref_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        return ref_logps

    # ---------------------------------------------------------------------- #
    # Dataloader override                                                      #
    # ---------------------------------------------------------------------- #

    def get_train_dataloader(self) -> DataLoader:
        """
        Return a DataLoader that drives the TRL training loop with placeholder
        batches.

        The actual training data comes from the Atropos API (fetched inside
        ``_prepare_inputs``).  We still need a DataLoader to drive the TRL
        training loop, so we create one that emits trivially small placeholder
        dicts.  The ``steps_per_generation`` multiplication from the base class
        is preserved so that the buffering logic works correctly.

        If the user supplied a real dataset (e.g. for hybrid training), the
        parent's implementation is used instead.
        """
        if not isinstance(self.train_dataset, _AtroposPlaceholderDataset):
            return super().get_train_dataloader()

        # Build a DataLoader whose length matches the expected training
        # schedule so tqdm and early-stopping work correctly.
        total_steps = getattr(self.args, "max_steps", None) or 10_000
        # Match the batch size the base class uses (per-device × steps_per_gen)
        batch_size = self._train_batch_size * self.args.steps_per_generation
        dataset = _AtroposPlaceholderDataset(total_steps * batch_size)

        def _collate(batch):
            # Return a minimal dict; _prepare_inputs ignores it entirely.
            return {"prompt": [""] * len(batch)}

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_collate,
            num_workers=0,
        )

    # ---------------------------------------------------------------------- #
    # Startup helpers                                                          #
    # ---------------------------------------------------------------------- #

    def train(self, *args, **kwargs):
        """
        Verify connectivity to the Atropos API server before training starts,
        then delegate to the standard TRL training loop.
        """
        logger.info(
            "AtroposGRPOTrainer: connecting to Atropos API at %s",
            self._atropos_args.atropos_api_url,
        )
        self._ensure_registered()
        try:
            return super().train(*args, **kwargs)
        finally:
            # Notify the Atropos API server that training has finished so it
            # can clean up any per-trainer state (e.g., queue, buffer, envs).
            self._atropos_client.disconnect_trainer()
            logger.info("Disconnected from Atropos API server.")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_atropos_trainer(
    model: str,
    atropos_api_url: str = "http://localhost:8000",
    group_size: int = 8,
    per_device_train_batch_size: int = 4,
    max_steps: int = 1000,
    output_dir: str = "./atropos_grpo_output",
    # TRL vLLM server settings
    vllm_server_host: str = "0.0.0.0",
    vllm_server_port: int = 8001,
    vllm_server_base_url: Optional[str] = None,
    extra_config_kwargs: Optional[Dict[str, Any]] = None,
    **trainer_kwargs,
) -> AtroposGRPOTrainer:
    """
    Convenience factory for the most common configuration.

    The trainer is wired to the TRL vLLM server (started separately via
    ``trl vllm-serve``) for weight synchronisation.  The Atropos environment
    microservices should be configured to call that same server for generation.

    Args:
        model: HuggingFace model ID or local path.
        atropos_api_url: URL of the Atropos run-api server.
        group_size: Number of completions per prompt (must match env config).
        per_device_train_batch_size: Training micro-batch size per GPU.
        max_steps: Total training steps.
        output_dir: Directory for checkpoints and logs.
        vllm_server_host: Host where `trl vllm-serve` is listening.
        vllm_server_port: Port where `trl vllm-serve` is listening.
        vllm_server_base_url: Optional explicit base URL (overrides host+port).
        extra_config_kwargs: Additional kwargs passed to AtroposGRPOConfig.
        **trainer_kwargs: Additional kwargs passed to AtroposGRPOTrainer.

    Returns:
        A configured AtroposGRPOTrainer ready to call .train() on.

    Example::

        trainer = make_atropos_trainer(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            atropos_api_url="http://localhost:8000",
            vllm_server_host="0.0.0.0",
            vllm_server_port=8001,
            group_size=8,
            max_steps=500,
        )
        trainer.train()
    """
    config_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_generations": group_size,
        "atropos_api_url": atropos_api_url,
        "atropos_group_size": group_size,
        "max_steps": max_steps,
        # TRL vLLM server — always use server mode
        "use_vllm": True,
        "vllm_mode": "server",
        "vllm_server_host": vllm_server_host,
        "vllm_server_port": vllm_server_port,
    }

    if vllm_server_base_url:
        config_kwargs["vllm_server_base_url"] = vllm_server_base_url

    if extra_config_kwargs:
        config_kwargs.update(extra_config_kwargs)

    config = AtroposGRPOConfig(**config_kwargs)
    return AtroposGRPOTrainer(model=model, args=config, **trainer_kwargs)