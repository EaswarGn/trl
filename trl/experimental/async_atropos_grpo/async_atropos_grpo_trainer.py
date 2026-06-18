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
AsyncAtroposGRPOTrainer: async GRPO trainer sourcing rollouts from Atropos.

A subclass of ``AsyncGRPOTrainer`` that replaces the vLLM-backed rollout
worker with an ``AtroposRolloutWorker`` that polls the Atropos API for
pre-scored trajectory batches.

Architecture
------------
The trainer uses the fully asynchronous training loop from ``AsyncGRPOTrainer``:
rollout generation (via Atropos) runs in a CUDA-free child process, while
the GPU-bound training loop consumes scored ``RolloutSample`` objects from a
shared ``mp.Queue``.  This decouples generation from gradient updates,
allowing both to proceed concurrently.

Data flow::

    Atropos API server (run-api, port 8000)
         | GET /batch
         v
    AtroposRolloutWorker (child process)
        - Polls /batch
        - Computes GRPO advantages
        - Pushes RolloutSample -> mp.Queue
         | mp.Queue[RolloutSample]
         v
    AsyncAtroposGRPOTrainer
        - AsyncGRPOTrainer training loop (inherited)
        - compute_loss() (inherited)
        - _sync_weight() (inherited, NCCL -> vLLM dev-mode server)

The vLLM server must be started with VLLM_SERVER_DEV_MODE=1 and NCCL weight
transfer enabled so that the trainer can push model updates to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback

from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.async_grpo.async_grpo_trainer import EnvironmentFactory, RewardFunc
from trl.experimental.async_grpo.weight_transfer import WeightTransferClient

from .async_atropos_grpo_config import AsyncAtroposGRPOConfig
from .atropos_rollout_worker import AtroposRolloutWorker

logger = logging.getLogger(__name__)


class AsyncAtroposGRPOTrainer(AsyncGRPOTrainer):
    """
    Asynchronous GRPO trainer that sources rollouts from the Atropos API.

    This trainer extends ``AsyncGRPOTrainer`` by plugging in an
    ``AtroposRolloutWorker`` that polls the Atropos ``run-api`` server for
    pre-scored trajectory batches.  The async training loop (weight sync,
    loss computation, gradient accumulation, logging) is fully inherited
    from ``AsyncGRPOTrainer``.

    The vLLM server must be a vanilla vLLM instance started with dev mode
    and NCCL weight transfer enabled::

        VLLM_SERVER_DEV_MODE=1 vllm serve <MODEL> \\
            --max-model-len <LEN> \\
            --logprobs-mode processed_logprobs \\
            --weight-transfer-config '{"backend":"nccl"}'

    Example usage::

        from trl.experimental.async_atropos_grpo import (
            AsyncAtroposGRPOTrainer,
            AsyncAtroposGRPOConfig,
        )

        config = AsyncAtroposGRPOConfig(
            output_dir="./my_run",
            atropos_api_url="http://localhost:8000",
            atropos_group_size=8,
            max_steps=1000,
            vllm_server_base_url="http://localhost:8001",
            weight_sync_steps=1,
        )

        trainer = AsyncAtroposGRPOTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            args=config,
        )
        trainer.train()
    """

    _name = "AsyncAtroposGRPO"

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: AsyncAtroposGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Any, Any] | None = None,
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | None = None,
        # Allow injecting a custom rollout worker (e.g. for testing)
        rollout_worker=None,
        **kwargs,
    ):
        # Default config
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = AsyncAtroposGRPOConfig(
                output_dir=f"{model_name.split('/')[-1]}-async-atropos-grpo",
            )

        if not isinstance(args, AsyncAtroposGRPOConfig):
            raise TypeError(
                f"AsyncAtroposGRPOTrainer requires an AsyncAtroposGRPOConfig instance, "
                f"got {type(args).__name__}."
            )

        # Create the AtroposRolloutWorker if no custom worker was injected.
        # We pass it as rollout_worker to the parent so it skips creating
        # a default AsyncRolloutWorker.
        if rollout_worker is None:
            rollout_worker = AtroposRolloutWorker(
                atropos_api_url=args.atropos_api_url,
                atropos_trainer_id=args.atropos_trainer_id,
                group_size=args.atropos_group_size,
                batch_timeout=args.atropos_batch_timeout,
                poll_interval=args.atropos_poll_interval,
                max_retries=args.atropos_max_retries,
                max_inflight_batches=args.atropos_max_inflight_batches,
                queue_maxsize=args.queue_maxsize,
                log_completions=args.log_completions,
                num_completions_to_print=args.num_completions_to_print,
            )

        # Use pass-through reward funcs when none are given — Atropos provides
        # the actual scores.
        if reward_funcs is None:
            def _passthrough_reward(**kw) -> list[float]:
                prompts = kw.get("prompts", kw.get("prompt", None))
                if prompts is not None:
                    return [0.0] * len(prompts)
                return []
            reward_funcs = _passthrough_reward

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # Store config for later use
        self._atropos_args = args

        # Delegates to AsyncGRPOTrainer.__init__.
        # The parent loads the model and tokenizer.  When a custom
        # rollout_worker is passed, the parent skips creating both
        # WeightTransferClient and AsyncRolloutWorker — it just stores
        # rollout_worker and sets weight_transfer = None.
        # We fix weight_transfer after the call.
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            tools=tools,
            environment_factory=environment_factory,
            rollout_worker=rollout_worker,
            **kwargs,
        )

        # ------------------------------------------------------------------ #
        # FIX: Create WeightTransferClient if the parent skipped it
        # ------------------------------------------------------------------ #
        # When a custom rollout_worker is passed, AsyncGRPOTrainer.__init__
        # sets self.weight_transfer = None.  We need weight transfer to push
        # model updates to the vLLM server so the Atropos environments
        # generate from the latest policy.  Create it now.
        if self.weight_transfer is None and self.accelerator.is_main_process:
            self._init_weight_transfer_for_atropos()

        # Registration state
        self._atropos_registered = False

    def _init_weight_transfer_for_atropos(self) -> None:
        """Create WeightTransferClient for pushing weights to the vLLM server.

        This replicates the weight metadata collection that the parent does
        in its default path (when no custom rollout_worker is given).
        """
        weight_names, weight_dtype_names, weight_shapes = [], [], []
        for name, param in self.model.named_parameters():
            # DDP/FSDP1 wrapping — strip "module." prefix if present
            name = name.removeprefix("module.")
            weight_names.append(name)
            weight_dtype_names.append(str(param.dtype).split(".")[-1])
            weight_shapes.append(list(param.shape))

        self.weight_transfer = WeightTransferClient(
            vllm_server_url=self.args.vllm_server_base_url,
            server_timeout=self.args.vllm_server_timeout,
            weight_update_info={
                "names": weight_names,
                "dtype_names": weight_dtype_names,
                "shapes": weight_shapes,
                "packed": True,
                "is_checkpoint_format": True,
            },
        )
        logger.info(
            "Created WeightTransferClient for vLLM server at %s",
            self.args.vllm_server_base_url,
        )

    # ------------------------------------------------------------------ #
    # Atropos registration                                                 #
    # ------------------------------------------------------------------ #

    def _ensure_registered(self) -> None:
        """Register with the Atropos API server exactly once."""
        if self._atropos_registered:
            return
        # The child process (AtroposRolloutWorker) handles actual registration.
        # We do a lightweight health check here to fail fast if the API server
        # is unreachable — before spawning the child.
        if not self.accelerator.is_main_process:
            self._atropos_registered = True
            return

        import requests
        try:
            resp = requests.get(f"{self._atropos_args.atropos_api_url}/", timeout=5.0)
            if resp.status_code >= 500:
                raise ConnectionError(
                    f"Atropos API at {self._atropos_args.atropos_api_url} returned "
                    f"status {resp.status_code}."
                )
            logger.info(
                "Atropos API at %s is reachable",
                self._atropos_args.atropos_api_url,
            )
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach Atropos API at {self._atropos_args.atropos_api_url}. "
                "Ensure `run-api` is running before starting the trainer."
            ) from e

        self._atropos_registered = True

    # ------------------------------------------------------------------ #
    # train() override                                                      #
    # ------------------------------------------------------------------ #

    def train(self, *args, **kwargs):
        """Register with Atropos, then start the async training loop."""
        # Only register on the main process
        if self.accelerator.is_main_process:
            self._ensure_registered()

        # Barrier so all processes wait for registration before proceeding
        self.accelerator.wait_for_everyone()

        try:
            return super().train(*args, **kwargs)
        finally:
            # Clean shutdown — the parent's _inner_training_loop handles
            # rollout_worker.stop() and weight_transfer.destroy().
            pass