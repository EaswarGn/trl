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
from typing import Any, Dict, Optional, Union

from datasets import Dataset
import torch
from transformers import PreTrainedTokenizerBase, TrainerCallback
from .placeholders import _DummyIterableDataset, _passthrough_reward

from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.async_grpo.async_grpo_trainer import EnvironmentFactory, RewardFunc, RolloutWorkerProtocol
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
        model_name: str,
        args: AsyncAtroposGRPOConfig | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        **kwargs,
    ):
        # Default config
        if args is None:
            args = AsyncAtroposGRPOConfig(
                output_dir=f"{model_name.split('/')[-1]}-async-atropos-grpo",
            )

        if not isinstance(args, AsyncAtroposGRPOConfig):
            raise TypeError(
                f"AsyncAtroposGRPOTrainer requires an AsyncAtroposGRPOConfig instance, "
                f"got {type(args).__name__}."
            )

        rollout_worker: RolloutWorkerProtocol = AtroposRolloutWorker(
            atropos_api_url=args.atropos_api_url,
            group_size=args.atropos_group_size,
            batch_timeout=args.atropos_batch_timeout,
            poll_interval=args.atropos_poll_interval,
            max_retries=args.atropos_max_retries,
            queue_maxsize=args.queue_maxsize,
            log_completions=args.log_completions,
            num_completions_to_print=args.num_completions_to_print,
            processing_class_name=model_name,
        )

        # ``AsyncGRPOTrainer`` requires a non-None train_dataset, but the
        # dummy placeholder is never actually consumed — the inherited
        # ``get_train_dataloader`` returns a ``RolloutQueueDataset`` backed
        # by the rollout worker's queue instead.
        _passthrough_dataset = _DummyIterableDataset()

        super().__init__(
            model=model_name,
            reward_funcs=_passthrough_reward,
            args=args,
            train_dataset=_passthrough_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            rollout_worker=rollout_worker,
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.weight_transfer = self._init_weight_transfer_client()
        else:
            self.weight_transfer = None

        # Registration state
        self._atropos_registered = False
        self.atropos_configs = args

    def _init_weight_transfer_client(self) -> WeightTransferClient:
        """Collect weight metadata from the loaded model and create a WeightTransferClient."""
        weight_names, weight_dtype_names, weight_shapes = [], [], []
        for name, param in self.model.named_parameters():
            # DDP/FSDP1 wrapping, avoids vllm module not exist error
            name = name.removeprefix("module.")
            weight_names.append(name)
            weight_dtype_names.append(str(param.dtype).split(".")[-1])
            weight_shapes.append(list(param.shape))

        return WeightTransferClient(
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

    # ------------------------------------------------------------------ #
    # Atropos registration                                                 #
    # ------------------------------------------------------------------ #
    def register_with_atropos(self) -> None:
        """Register with the Atropos API server exactly once."""
        if self._atropos_registered:
            return

        url = f"{self.atropos_configs.atropos_api_url}/register"
        payload = {
            # wandb fields are required strings - use empty string if None
            "wandb_group": self.atropos_configs.atropos_env_wandb_group or "",
            "wandb_project": self.atropos_configs.atropos_env_wandb_project or "",
            "batch_size": self.atropos_configs.per_device_train_batch_size,
            "max_token_len": self.atropos_configs.atropos_max_tokens,
            "starting_step": self.state.global_step,
            "checkpoint_dir": self.atropos_configs.output_dir,
            "save_checkpoint_interval": self.atropos_configs.save_steps * self.atropos_configs.gradient_accumulation_steps,
            "num_steps": self.atropos_configs.max_steps * self.atropos_configs.gradient_accumulation_steps,
        }

        import requests
        from requests.models import Response
        import time
        for attempt in range(self.atropos_configs.atropos_max_retries):
            try:
                resp: Response = requests.post(url, json=payload, timeout=30.0)
                resp.raise_for_status()
                logger.info("Registered with Atropos API at %s", self.atropos_configs.atropos_api_url)
                self._atropos_registered = True
                return
            except requests.RequestException as exc:
                if attempt == self.atropos_configs.atropos_max_retries - 1:
                    raise ConnectionError(
                        f"Cannot register with Atropos API at {self.atropos_configs.atropos_api_url} "
                        f"after {self.atropos_configs.atropos_max_retries} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "Registration attempt %d/%d failed: %s – retrying in 2s",
                    attempt + 1,
                    self.atropos_configs.atropos_max_retries,
                    exc,
                )
                time.sleep(2.0)

    # ------------------------------------------------------------------ #
    # _inner_training_loop() override
    # ------------------------------------------------------------------ #

    def _inner_training_loop(self, *args, **kwargs):
        """Register with Atropos, then start the async training loop."""
        # Only register on the main process
        if self.accelerator.is_main_process:
            self.register_with_atropos()

        # Barrier so all processes wait for registration before proceeding
        self.accelerator.wait_for_everyone()

        return super()._inner_training_loop(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Evaluation override
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Override evaluate method to prevent actual evaluation from running.

        In AsyncAtroposGRPOTrainer, evaluation is handled by the Atropos environment,
        not by the trainer itself. This method logs that information and returns
        an empty metrics dictionary without running any evaluation.
        """
        import inspect
        logger.info(
            inspect.cleandoc(
                """
                AsyncAtroposGRPOTrainer does not run evaluation - the Atropos environment
                is responsible for evaluation. Returning empty metrics.

                You can also run evaluation separately on your env with the command below:
                python <your_environment>.py evaluate \\
                --openai.base_url <openai_url> \\
                --openai.api_key <api_key> \\
                --openai.model_name <model_id>
                """
            )
        )
        # Return empty metrics dictionary matching the expected return type
        return {}