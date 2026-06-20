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
from trl.trainer import GRPOConfig
from typing import Optional


# ---------------------------------------------------------------------------
# AtroposGRPOConfig
# ---------------------------------------------------------------------------

@dataclass
class AtroposGRPOConfig(GRPOConfig):
    """
    Configuration for AtroposGRPOTrainer.

    Extends GRPOConfig with Atropos-specific fields.  All standard GRPOConfig
    fields (including use_vllm, vllm_mode, vllm_server_host, vllm_server_port,
    vllm_server_base_url, beta, etc.) are inherited and should be used as
    normal to control the TRL vLLM server connection and training behaviour.

    Atropos-specific fields
    -----------------------
    atropos_api_url : str
        Base URL of the Atropos run-api server.
    atropos_group_size : int
        Number of completions per prompt group.  Must match the group_size
        configured in the Atropos environment.
    atropos_trainer_id : str
        Identifier sent to the Atropos API during /register.
    atropos_batch_timeout : float
        Seconds to wait for a batch before raising TimeoutError.
    atropos_poll_interval : float
        Seconds between /batch polls when no data is available yet.
    atropos_max_retries : int
        Number of HTTP retries on transient failures.
    """

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
        default="trl_grpo",
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