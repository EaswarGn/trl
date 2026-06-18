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
    atropos_trainer_id (`str`, *optional*, defaults to `"trl_async_atropos"`):
        Identifier sent to the Atropos API during `/register`. Useful when
        running multiple trainers against the same API.
    atropos_batch_timeout (`float`, *optional*, defaults to `300.0`):
        Seconds to wait for a batch from the Atropos API before raising
        `TimeoutError`. Increase this if your environment is slow to score.
    atropos_poll_interval (`float`, *optional*, defaults to `1.0`):
        Seconds between `/batch` polls when no data is available yet.
    atropos_max_retries (`int`, *optional*, defaults to `3`):
        Number of HTTP retries on transient failures when polling the
        Atropos API.
    atropos_max_inflight_batches (`int`, *optional*, defaults to `2`):
        Maximum number of batches to fetch ahead and buffer locally in the
        rollout worker. Larger values smooth over API latency but increase
        the risk of stale policy data.
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
    atropos_trainer_id: str = field(
        default="trl_async_atropos",
        metadata={"help": "Identifier sent to the Atropos API on /register."},
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
    atropos_max_inflight_batches: int = field(
        default=2,
        metadata={
            "help": "Maximum number of batches to fetch ahead and buffer locally. "
            "Larger values smooth over API latency but increase the risk of stale policy data."
        },
    )