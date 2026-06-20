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
AtroposRolloutWorker
====================

A rollout worker that polls the Atropos API for pre-scored trajectory batches
instead of generating completions via vLLM.  This worker implements the
``RolloutWorkerProtocol`` from ``trl.experimental.async_grpo`` and can be
plugged directly into ``AsyncGRPOTrainer`` (or its subclass
``AsyncAtroposGRPOTrainer``).

Architecture
------------
The worker lives in a CUDA-free child process (spawned via ``multiprocessing``)
and communicates with the parent trainer through a shared ``mp.Queue``
(``rollout_buffer``).  The child process:

1. Polls the Atropos API server (GET /batch) for scored trajectory groups.
2. Flattens each group into individual ``RolloutSample`` instances.
3. Decodes prompt and completion text via the tokenizer.
4. Computes group-relative GRPO advantages and detailed rollout metrics.
5. Pushes the samples onto the shared queue.

Data-flow
---------
Atropos API /batch → child process → mp.Queue[RolloutSample] → AsyncGRPOTrainer

The parent trainer never blocks on API calls — all HTTP I/O happens in the
child process, keeping the GPU training loop uninterrupted.
"""

import logging
import multiprocessing as mp
import os
import pickle
import queue
import time
import traceback
from collections import Counter
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.sharedctypes import Synchronized as MPValue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Optional

import numpy as np
import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trl.experimental.async_grpo.async_rollout_worker import RolloutSample
from trl.trainer.utils import print_prompt_completions_sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Env vars the child must drop so accelerate's PartialState() initialises in
# single-process mode instead of trying to join the parent's process group.
_CHILD_ENV_TO_STRIP = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCH_FR_DUMP_TEMP_FILE",
    "NCCL_DEBUG_FILE",
)


def _scrub_child_env() -> None:
    """Strip distributed-training env vars and CUDA from the child process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for k in _CHILD_ENV_TO_STRIP:
        os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Child process entry point
# ---------------------------------------------------------------------------


def _child_main(
    worker_kwargs: dict[str, Any],
    samples_queue: MPQueue,
    model_version_value: MPValue,
    stop_event: MPEvent,
    child_ready_event: MPEvent,
    heartbeat_value: MPValue,
    failed_event: MPEvent,
    exception_info_queue: MPQueue,
) -> None:
    """Entry point for the spawned child process.

    Scrub the environment, initialise the polling loop, and
    signal readiness to the parent.  Any unhandled exception is forwarded
    to the parent via ``failed_event`` + ``exception_info_queue``.
    """
    _scrub_child_env()
    # accelerate.logging.get_logger requires PartialState() to have been called.
    from accelerate.state import PartialState

    PartialState()

    loop = _AtroposPollingLoop(
        **worker_kwargs,
        rollout_buffer=samples_queue,
        model_version_value=model_version_value,
        stop_event=stop_event,
        heartbeat_value=heartbeat_value,
        failed_event=failed_event,
        exception_info_queue=exception_info_queue,
    )
    child_ready_event.set()
    try:
        loop.run()
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Polling loop (lives in child process)
# ---------------------------------------------------------------------------


class _AtroposPollingLoop:
    """Synchronous polling loop that runs inside the child process.

    Owns the HTTP session, the tokenizer, and the loop control state.
    Talks to the Atropos API via ``requests``.  Pushes scored
    ``RolloutSample`` instances into the shared ``mp.Queue``.
    """

    def __init__(
        self,
        *,
        atropos_api_url: str,
        group_size: int,
        batch_timeout: float,
        poll_interval: float,
        max_retries: int,
        log_completions: bool,
        num_completions_to_print: int,
        rollout_buffer: MPQueue,
        model_version_value: MPValue,
        stop_event: MPEvent,
        heartbeat_value: MPValue,
        failed_event: MPEvent,
        exception_info_queue: MPQueue,
        processing_class_name: str | None = None,
    ):
        self.atropos_api_url = atropos_api_url.rstrip("/")
        self.group_size = group_size
        self.batch_timeout = batch_timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print

        self.rollout_buffer = rollout_buffer
        self._model_version_value = model_version_value
        self._stop_event = stop_event
        self._heartbeat_value = heartbeat_value
        self._failed_event = failed_event
        self._exception_info_queue = exception_info_queue

        # ------------------------------------------------------------------ #
        # Tokenizer: loaded in child so prompt/completion text can be decoded.
        # ``processing_class_name`` is a plain str (picklable), never a
        # tokenizer object.  We load the tokenizer here in the child, avoiding
        # any torch/tokenizers CUDA initialisation in the parent.
        # ------------------------------------------------------------------ #
        self.tokenizer: Optional[PreTrainedTokenizerBase]  = None
        if processing_class_name is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(processing_class_name)
                logger.info(
                    "Loaded tokenizer '%s' for prompt/completion text decoding.",
                    processing_class_name,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load tokenizer '%s': %s. "
                    "RolloutSample.prompt/completion will be empty lists.",
                    processing_class_name,
                    exc,
                )

        # State
        self._session: requests.Session | None = None
        self._total_groups_processed = 0
        # Timing accumulators for detailed metrics
        self._generation_start_time: float | None = None
        self._total_completion_tokens = 0

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    def run(self) -> None:
        self._session = requests.Session()
        try:
            self._poll_loop()
        except BaseException as e:
            info = (type(e).__name__, str(e), traceback.format_exc())
            try:
                self._exception_info_queue.put_nowait(info)
            except Exception:
                pass
            self._failed_event.set()
            logger.exception(f"Atropos polling loop failed: {e}")
            raise
        finally:
            if self._session is not None:
                self._session.close()

    # ------------------------------------------------------------------
    # Batch fetching
    # ------------------------------------------------------------------

    def _fetch_batch(self, session: requests.Session | None = None) -> list[dict[str, Any]] | None:
        """Poll /batch once.  Returns the list of group dicts or None.

        ``session`` is optional and, when provided, is used instead of
        ``self._session``.  This keeps the shared session free of concurrent
        access from the thread-pool executor workers.
        """
        sess = session if session is not None else self._session
        url = f"{self.atropos_api_url}/batch"
        for attempt in range(self.max_retries):
            try:
                resp = sess.get(url, timeout=30.0)
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    return None
                logger.warning(
                    "GET /batch attempt %d/%d failed: %s – retrying in 1s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                time.sleep(1.0)
                continue

            if resp.status_code == 204:
                return None
            try:
                resp.raise_for_status()
            except requests.RequestException:
                return None

            data = resp.json()
            if isinstance(data, dict):
                batch = data.get("batch")
                if batch is None:
                    return None
                return batch
            return data

        return None

    def _wait_for_batch(self) -> list[dict[str, Any]]:
        """Block until a batch is available, polling every ``poll_interval``.

        Each thread gets its own ``requests.Session`` so that concurrent
        HTTP calls from the thread-pool executor are safe.
        """
        with requests.Session() as session:
            deadline = time.monotonic() + self.batch_timeout
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    raise RuntimeError("Stop event set while waiting for batch.")
                batch = self._fetch_batch(session)
                if batch is not None and len(batch) > 0:
                    return batch
                time.sleep(self.poll_interval)
        raise TimeoutError(
            f"No batch available from Atropos API at {self.atropos_api_url} "
            f"within {self.batch_timeout}s."
        )

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_rollout_metrics(
        self,
        samples: list[RolloutSample],
        conversion_time_s: float,
        buffer_wait_time_s: float,
    ) -> None:
        """Attach extra per-sample metrics to every ``RolloutSample`` in-place.

        Additional metrics beyond ``async_grpo``:
            - ``conversion_time_ms``: wall-clock time to convert the raw batch
              into ``RolloutSample`` objects (split prompt/completion, compute
              advantages, decode text).
            - ``buffer_wait_time_s``: time spent retrying ``put_nowait`` on the
              mp.Queue when it was full.
            - ``generation_tok_per_s``: rolling throughput since this child
              process started.
            - ``buffer_qsize``: number of items in rollout buffer.
        """
        assert self._generation_start_time is not None
        elapsed = time.monotonic() - self._generation_start_time
        generation_tok_per_sec = self._total_completion_tokens / elapsed if elapsed > 0 else 0.0
        buffer_qsize = self.rollout_buffer.qsize()

        for sample in samples:
            sample.metrics["conversion_time_ms"] = conversion_time_s * 1000
            sample.metrics["buffer_wait_time_s"] = buffer_wait_time_s
            sample.metrics["generation_tok_per_s"] = generation_tok_per_sec
            sample.metrics["buffer_qsize"] = buffer_qsize

    # ------------------------------------------------------------------
    # Batch conversion: Atropos group dicts → RolloutSample list
    # ------------------------------------------------------------------

    def _convert_batch(self, raw_batch: list[dict[str, Any]]) -> list[RolloutSample]:
        """Convert a list of Atropos group dicts to individual ``RolloutSample``s.

        Each group dict contains parallel lists of sequences (tokens, masks,
        inference_logprobs, scores).  This method flattens the groups,
        splits tokens into prompt and completion parts, decodes them to text
        using the tokenizer, and applies group-relative GRPO advantage
        normalisation.

        Atropos batch contract (from /batch endpoint)
        ----------------------------------------------
        Each group dict:
            tokens              : list[list[int]]   – parallel sequences (prompt+completion)
            masks               : list[list[int]]   – -100 for prompt, token_id for completion
            inference_logprobs  : list[list[float]] – logprobs for completion tokens
            scores              : list[float]       – reward per sequence in the group
            env_id              : int               – source environment identifier
        """
        group_size = self.group_size
        trajectories: list[dict[str, Any]] = []
        env_id_counter: Counter = Counter()

        for group_idx, group_item in enumerate(raw_batch):
            try:
                tokens_list: list[list[int]] = group_item["tokens"]
                masks_list: list[list[int]] = group_item["masks"]
            except KeyError as e:
                raise KeyError(
                    f"Atropos batch group {group_idx} missing required field {e}. "
                    "Each group dict must contain 'tokens' and 'masks'."
                ) from e

            logprobs_list: list[list[float]] = group_item.get(
                "inference_logprobs",
                [[0.0] * len(seq) for seq in tokens_list],
            )
            scores_list: list[float] = group_item.get(
                "scores",
                [0.0] * len(tokens_list),
            )

            env_id = group_item.get("env_id")
            env_id_str = str(env_id) if env_id is not None else "unknown"
            env_id_counter[env_id_str] += len(tokens_list)

            seq_count = len(tokens_list)
            if not (
                len(masks_list) == seq_count
                and len(logprobs_list) == seq_count
                and len(scores_list) == seq_count
            ):
                raise ValueError(
                    f"Atropos batch group {group_idx} has mismatched sequence counts: "
                    f"tokens={seq_count}, masks={len(masks_list)}, "
                    f"logprobs={len(logprobs_list)}, scores={len(scores_list)}."
                )

            for seq_idx in range(seq_count):
                trajectories.append(
                    {
                        "tokens": tokens_list[seq_idx],
                        "masks": masks_list[seq_idx],
                        "logprobs": logprobs_list[seq_idx],
                        "score": scores_list[seq_idx],
                    }
                )

        if len(env_id_counter) > 1:
            logger.info(
                "Atropos batch contains trajectories from %d environments: %s",
                len(env_id_counter),
                dict(env_id_counter),
            )

        num_trajectories = len(trajectories)
        if num_trajectories == 0:
            raise ValueError("Atropos batch is empty after flattening groups.")

        if num_trajectories % group_size != 0:
            raise ValueError(
                f"Atropos batch size {num_trajectories} is not divisible by "
                f"group_size {group_size}.  Ensure the Atropos environment "
                "group_size matches the config."
            )

        # ------------------------------------------------------------------
        # Split tokens into prompt / completion parts and decode to text
        # ------------------------------------------------------------------
        completion_logps_list: list[list[float]] = []
        scores_list: list[float] = []
        full_ids_list: list[list[int]] = []
        completion_mask_list: list[list[int]] = []
        prompt_texts: list[str] = []
        completion_texts: list[str] = []
        prompt_ids_list: list[list[int]] = []
        completion_ids_list_decoded: list[list[int]] = []

        for traj in trajectories:
            tokens: list[int] = traj["tokens"]
            masks: list[int] = traj["masks"]
            logprobs: list[float] = traj["logprobs"]

            if not (len(tokens) == len(masks) == len(logprobs)):
                raise ValueError(
                    f"Trajectory has mismatched lengths: tokens={len(tokens)}, "
                    f"masks={len(masks)}, logprobs={len(logprobs)}."
                )

            # Find completion start: first position where masks != -100
            completion_start = next(
                (i for i, m in enumerate(masks) if m != -100),
                len(tokens),
            )

            prompt_ids = tokens[:completion_start]
            comp_ids = tokens[completion_start:]

            full_ids_list.append(tokens)
            completion_mask_list.append(
                [0] * completion_start + [1] * (len(tokens) - completion_start)
            )
            prompt_ids_list.append(prompt_ids)
            completion_ids_list_decoded.append(comp_ids)

            # Replace sentinel 0.0 logprobs (meaning "not provided") with -100.0
            # so they don't bias the importance-sampling ratio.
            raw_lps = logprobs[completion_start:]
            completion_logps_list.append(
                [lp if lp != 0.0 else -100.0 for lp in raw_lps]
            )

            scores_list.append(float(traj["score"]))

            # Decode prompt and completion text using the tokenizer.
            # If no tokenizer is available, fall back to empty strings.
            if self.tokenizer is not None:
                try:
                    prompt_texts.append(self.tokenizer.decode(prompt_ids, skip_special_tokens=False))
                except Exception:
                    prompt_texts.append("")
                try:
                    completion_texts.append(self.tokenizer.decode(comp_ids, skip_special_tokens=False))
                except Exception:
                    completion_texts.append("")
            else:
                prompt_texts.append("")
                completion_texts.append("")

        # ------------------------------------------------------------------
        # GRPO advantage computation (group-relative normalization)
        # ------------------------------------------------------------------
        scores_tensor = np.array(scores_list, dtype=np.float64)
        num_groups = num_trajectories // group_size
        grouped = scores_tensor.reshape(num_groups, group_size)
        group_means = grouped.mean(axis=1, keepdims=True)
        group_stds = grouped.std(axis=1, ddof=0, keepdims=True)
        advantages = ((grouped - group_means) / (group_stds + 1e-4)).ravel()

        reward_mean = float(scores_tensor.mean())
        reward_std = float(scores_tensor.std())

        # ------------------------------------------------------------------
        # Build RolloutSample list
        # ------------------------------------------------------------------
        samples: list[RolloutSample] = []

        # Log batch-level metadata
        logger.info(
            "Scored batch: %d trajectories in %d groups, "
            "reward_mean=%.4f, reward_std=%.4f",
            num_trajectories,
            num_groups,
            reward_mean,
            reward_std,
        )

        for i in range(num_trajectories):
            # Build old_log_probs: pad with zeros for prompt positions, then
            # append completion logprobs.
            prompt_len = (
                completion_mask_list[i].index(1)
                if 1 in completion_mask_list[i]
                else len(completion_mask_list[i])
            )
            old_log_probs_list = [0.0] * prompt_len + completion_logps_list[i]

            samples.append(
                RolloutSample(
                    prompt=prompt_texts[i],          # decoded prompt text
                    completion=completion_texts[i],  # decoded completion text
                    input_ids=full_ids_list[i],
                    completion_mask=completion_mask_list[i],
                    old_log_probs=old_log_probs_list,
                    advantage=float(advantages[i]),
                    model_version=self.model_version,
                    metrics={
                        "reward": float(scores_list[i]),
                        "reward_mean": reward_mean,
                        "reward_std": reward_std,
                    },
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Main polling loop: fetch & push batches."""

        # Prime the generation-start clock so that throughput can be measured
        # from the first successful conversion, not from arbitrary process start.
        self._generation_start_time = None

        buffer_total_wait_s = 0.0  # rolling total for logging

        try:
            while not self._stop_event.is_set():
                self._heartbeat_value.value = time.time()

                try:
                    raw_batch = self._wait_for_batch()
                except RuntimeError as exc:
                    if "Stop event set" in str(exc):
                        break
                    logger.warning("Batch fetch failed: %s", exc)
                    raw_batch = None
                except Exception as exc:
                    logger.warning("Batch fetch failed: %s", exc)
                    raw_batch = None

                if not raw_batch:
                    continue

                # Initialise the generation clock on the first real batch
                if self._generation_start_time is None:
                    self._generation_start_time = time.monotonic()

                # Time the batch conversion phase
                t_convert_start = time.monotonic()
                samples = self._convert_batch(raw_batch)
                conversion_time_s = time.monotonic() - t_convert_start

                # Count completion tokens for throughput measurement
                for s in samples:
                    self._total_completion_tokens += sum(s.completion_mask)

                # Push each sample onto the shared queue, tracking queue-wait time
                t_buffer_start = time.monotonic()
                buffer_wait_time_s = 0.0
                for sample in samples:
                    while True:
                        try:
                            self.rollout_buffer.put_nowait(sample)
                            break
                        except queue.Full:
                            block_start = time.monotonic()
                            logger.info(
                                "Rollout buffer full (qsize=%d), waiting for trainer to consume...",
                                self.rollout_buffer.qsize(),
                            )
                            time.sleep(0.5)
                            buffer_total_wait_s += (time.monotonic() - block_start)
                buffer_wait_time_s = time.monotonic() - t_buffer_start

                # Attach rollout metrics to each sample before the trainer consumes them
                self._compute_rollout_metrics(samples, conversion_time_s, buffer_wait_time_s)

                # Optionally print a human-readable prompt/completion sample
                if self.log_completions and samples:
                    print_prompt_completions_sample(
                        prompts=[s.prompt for s in samples],
                        completions=[s.completion for s in samples],
                        rewards={"reward": [s.metrics["reward"] for s in samples]},
                        advantages=[s.advantage for s in samples],
                        step=self._total_groups_processed,
                        num_samples=self.num_completions_to_print,
                    )

                # Heartbeat – signal liveness so the parent knows we're still alive
                self._heartbeat_value.value = time.time()
                self._total_groups_processed += 1

                # Detailed per-batch log line
                elapsed_s = time.monotonic() - (
                    self._generation_start_time or time.monotonic()
                )
                tok_per_sec = (
                    self._total_completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
                )
                group_size_denom = (
                    self.group_size if getattr(self, "group_size", 0) > 0 else 1
                )
                num_groups = len(samples) // group_size_denom
                logger.info(
                    "Processed batch %d: %d samples in %d groups, "
                    "conversion=%.1fms, buffer_wait=%.1fs, "
                    "completion_tok_per_s=%.1f, total_completion_tokens=%d, "
                    "buffer_qsize=%d, reward_mean=%.4f, reward_std=%.4f",
                    self._total_groups_processed,
                    len(samples),
                    num_groups,
                    conversion_time_s * 1000,
                    buffer_wait_time_s,
                    tok_per_sec,
                    self._total_completion_tokens,
                    self.rollout_buffer.qsize(),
                    samples[0].metrics["reward_mean"] if samples else 0.0,
                    samples[0].metrics["reward_std"] if samples else 0.0,
                )

                # Periodic summary every 10 batches
                if self._total_groups_processed % 10 == 0:
                    logger.info(
                        "[summary] batches_processed=%d, "
                        "total_completion_tokens=%d, avg_tok_per_s=%.1f, "
                        "total_buffer_wait_s=%.1fs",
                        self._total_groups_processed,
                        self._total_completion_tokens,
                        tok_per_sec,
                        buffer_total_wait_s,
                    )
        finally:
            pass


# ---------------------------------------------------------------------------
# Parent-side controller
# ---------------------------------------------------------------------------


class AtroposRolloutWorker:
    """Parent-side controller for the Atropos polling child process.

    Implements ``RolloutWorkerProtocol`` so it can be passed as
    ``rollout_worker`` to ``AsyncGRPOTrainer``.

    The child process runs ``_AtroposPollingLoop`` which polls the Atropos API
    for scored trajectory batches and pushes them onto the shared
    ``mp.Queue`` (``rollout_buffer``).

    Pickling note
    -------------
    All constructor arguments (except the shared synchronisation primitives
    created here) are forwarded to the child process via ``pickle``.  They
    must be picklable — use plain types (str, int, float, etc.).
    """

    def __init__(
        self,
        *,
        atropos_api_url: str = "http://localhost:8000",
        group_size: int = 8,
        batch_timeout: float = 300.0,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        queue_maxsize: int = 0,
        child_ready_timeout: int = 300,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
        processing_class_name: str | None = None,
    ):
        ctx = mp.get_context("spawn")
        self._mp_ctx = ctx
        self.rollout_buffer = ctx.Queue(maxsize=queue_maxsize)
        self._model_version_value = ctx.Value("i", 0)
        self._stop_event_mp = ctx.Event()
        self._child_ready_event = ctx.Event()
        self._heartbeat_value = ctx.Value("d", 0.0)
        self._failed_event = ctx.Event()
        self._exception_info_queue = ctx.Queue(maxsize=1)

        self._worker_kwargs = {
            "atropos_api_url": atropos_api_url,
            "group_size": group_size,
            "batch_timeout": batch_timeout,
            "poll_interval": poll_interval,
            "max_retries": max_retries,
            "log_completions": log_completions,
            "num_completions_to_print": num_completions_to_print,
            "processing_class_name": processing_class_name,
        }
        self._child_ready_timeout = child_ready_timeout
        self._process: mp.Process | None = None

    @property
    def model_version(self) -> int:
        return int(self._model_version_value.value)

    @model_version.setter
    def model_version(self, value: int) -> None:
        with self._model_version_value.get_lock():
            self._model_version_value.value = int(value)

    # ------------------------------------------------------------------
    # RolloutWorkerProtocol implementation
    # ------------------------------------------------------------------

    def update_model_version(self, model_version: int) -> None:
        """Tell the child process the current policy version for staleness tracking."""
        self.model_version = model_version

    def start(self) -> None:
        """Spawn the child process and wait for it to signal readiness."""
        if self._process is not None:
            logger.warning(
                "AtroposRolloutWorker.start() called but child is already running; ignoring."
            )
            return

        self._heartbeat_value.value = time.time()

        # Validate picklability before spawning
        try:
            pickle.dumps(self._worker_kwargs)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise TypeError(
                "AtroposRolloutWorker constructor arguments must be picklable. "
                "Use plain types (str, int, float)."
            ) from e

        self._process = self._mp_ctx.Process(
            target=_child_main,
            args=(
                self._worker_kwargs,
                self.rollout_buffer,
                self._model_version_value,
                self._stop_event_mp,
                self._child_ready_event,
                self._heartbeat_value,
                self._failed_event,
                self._exception_info_queue,
            ),
            name="atropos-rollout-worker-child",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "AtroposRolloutWorker spawned child pid=%d; "
            "waiting up to %ds for the ready signal",
            self._process.pid,
            self._child_ready_timeout,
        )

        deadline = time.monotonic() + self._child_ready_timeout
        while not self._child_ready_event.wait(timeout=1.0):
            if not self._process.is_alive():
                exit_code = self._process.exitcode
                self._process = None
                raise RuntimeError(
                    f"AtroposRolloutWorker child exited during init (exitcode={exit_code}). "
                    "Check the child's stderr for the traceback."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"AtroposRolloutWorker child did not signal ready within "
                    f"{self._child_ready_timeout}s."
                )
        logger.info("AtroposRolloutWorker child is ready")

    def check_health(self, stale_after_s: float) -> None:
        """Raise if the child crashed or hasn't ticked the heartbeat within ``stale_after_s``."""
        if self._failed_event.is_set():
            try:
                type_name, msg, tb = self._exception_info_queue.get_nowait()
                cause = RuntimeError(f"{type_name}: {msg}\n{tb}")
            except queue.Empty:
                cause = None
            raise RuntimeError("Atropos rollout worker child has failed; see chained exception.") from cause
        age = time.time() - self._heartbeat_value.value
        if age > stale_after_s:
            raise RuntimeError(
                f"Atropos rollout worker heartbeat stale: {age:.0f}s > {stale_after_s:.0f}s; "
                "child is hung."
            )

    def stop(self) -> None:
        """Stop the child process and release its resources."""
        if self._process is None:
            return
        logger.info("Stopping AtroposRolloutWorker child process...")
        self._stop_event_mp.set()
        if self._process._popen is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                logger.warning("Child did not exit within 15s; terminating.")
                self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
        self._process = None