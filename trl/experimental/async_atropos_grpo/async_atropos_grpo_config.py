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

import warnings
from dataclasses import dataclass, field

from trl.experimental.async_grpo import AsyncGRPOConfig


@dataclass
class AsyncAtroposGRPOConfig(AsyncGRPOConfig):
    """
    Configuration for [`AsyncAtroposGRPOTrainer`].

    Extends [`AsyncGRPOConfig`] with Atropos-specific fields. All standard
    `AsyncGRPOConfig` fields (including vLLM server settings, async rollout
    pipeline parameters, GRPO training hyperparameters, etc.) are inherited.

    Atropos-specific fields
    -----------------------
    atropos_api_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
        Base URL of the Atropos run-api server.
    atropos_group_size (`int`, *optional*, defaults to `8`):
        Number of completions per prompt group. Must match the `group_size`
        configured in the Atropos environment.
    atropos_batch_timeout (`float`, *optional*, defaults to `300.0`):
        Seconds to wait for a batch from the Atropos API before raising
        `TimeoutError`. Increase this if your environment is slow to score.
    atropos_poll_interval (`float`, *optional*, defaults to `1.0`):
        Seconds between `/batch` polls when no data is available yet.
    atropos_max_retries (`int`, *optional*, defaults to `3`):
        Number of HTTP retries on transient failures when polling the
        Atropos API.
    atropos_env_wandb_project (`str` or `None`, *optional*, defaults to `None`):
        Name of wandb project to use for atropos env metrics, if enabled in env.
    atropos_env_wandb_group (`str` or `None`, *optional*, defaults to `None`):
        Name of wandb group to use for atropos env metrics.
    atropos_max_tokens (`int`, *optional*, defaults to `2048`):
        Maximum number of tokens to use for prompt and completion.
        This value should be equal to the max prompt+completion tokens you expect.
        Good place to start is `max_completion_length` * 2.

    Ignored parameters (replaced by Atropos API)
    --------------------------------------------
    The following parameters are inherited from `AsyncGRPOConfig` but have **no effect**
    in `AsyncAtroposGRPOTrainer` because the Atropos API handles generation and scoring:

    - `num_generations`: Replaced by `atropos_group_size`
    - `max_completion_length`: Replaced by `atropos_max_tokens`
    - `temperature`: Controlled by the Atropos environment
    - `chat_template_kwargs`: Atropos handles chat templating
    - `max_tool_calling_iterations`: Atropos environment handles tool use
    - `max_inflight_tasks`: Atropos polls batches, doesn't manage inflight tasks
    - `request_timeout`: Only used for vLLM `/v1/completions` calls

    If you explicitly set any of these, a warning will be raised.
    """

    # Parameters that control the connection to Atropos
    atropos_api_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the Atropos run-api server."},
    )
    atropos_group_size: int = field(
        default=8,
        metadata={
            "help": (
                "Number of completions per prompt group. "
                "Must match the group_size in the Atropos environment config."
            )
        },
    )
    atropos_batch_timeout: float = field(
        default=300.0,
        metadata={"help": "Seconds to wait for a batch before raising TimeoutError."},
    )
    atropos_poll_interval: float = field(
        default=1.0,
        metadata={"help": "Seconds between /batch polls when no data is available."},
    )
    atropos_max_retries: int = field(
        default=3,
        metadata={"help": "Number of HTTP retries on transient failures."},
    )
    atropos_env_wandb_project: str | None = field(
        default=None,
        metadata={
            "help": "Name of wandb project to use for atropos env metrics, if enabled in env."
        },
    )
    atropos_env_wandb_group: str | None = field(
        default=None,
        metadata={
            "help": "Name of wandb group to use for atropos env metrics."
        },
    )
    atropos_max_tokens: int = field(
        default=2048,
        metadata={
            "help": "Maximum number of tokens to use for prompt and completion. "
            "This value should be equal to the max prompt+completion tokens you expect. "
            "Good place to start is `max_completion_length` * 2"
        },
    )

    # Override unused AsyncGRPOConfig parameters with None defaults and deprecation warnings
    # These are replaced by Atropos API / environment settings
    num_generations: int | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Replaced by `atropos_group_size`. "
            "The Atropos environment controls the number of generations per prompt."
        },
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Replaced by `atropos_max_tokens`. "
            "The Atropos environment controls the max completion length."
        },
    )
    temperature: float | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Controlled by the Atropos environment. "
            "The Atropos environment sets the sampling temperature."
        },
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Atropos handles chat templating. "
            "The Atropos environment applies the chat template."
        },
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Atropos environment handles tool use. "
            "The Atropos environment controls max tool-calling iterations."
        },
    )
    max_inflight_tasks: int | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Atropos polls batches, doesn't manage inflight tasks. "
            "This parameter is only used by the vLLM-based AsyncRolloutWorker."
        },
    )
    request_timeout: int | None = field(
        default=None,
        metadata={
            "help": "IGNORED in AsyncAtroposGRPOTrainer. Only used for vLLM /v1/completions calls. "
            "AtroposRolloutWorker uses its own `atropos_batch_timeout` and `atropos_poll_interval`."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Warn if user explicitly sets ignored parameters
        ignored_params = {
            "num_generations": "atropos_group_size",
            "max_completion_length": "atropos_max_tokens",
            "temperature": "Atropos environment temperature setting",
            "chat_template_kwargs": "Atropos environment chat templating",
            "max_tool_calling_iterations": "Atropos environment tool use handling",
            "max_inflight_tasks": "Atropos batch polling (not used)",
            "request_timeout": "atropos_batch_timeout / atropos_poll_interval",
        }

        for param, replacement in ignored_params.items():
            value = getattr(self, param)
            if value is not None:
                warnings.warn(
                    f"Parameter '{param}' is ignored in AsyncAtroposGRPOTrainer. "
                    f"It is replaced by {replacement}. "
                    f"Value provided ({value}) will have no effect.",
                    UserWarning,
                    stacklevel=2,
                )