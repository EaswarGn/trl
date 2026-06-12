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

from torch.utils.data import IterableDataset
from typing import List

# ---------------------------------------------------------------------------
# Dummy dataset so the TRL training loop has something to iterate over
# ---------------------------------------------------------------------------

class _AtroposPlaceholderDataset(IterableDataset):
    """
    Yields trivial placeholder dicts indefinitely.

    AtroposGRPOTrainer completely bypasses the HuggingFace dataset in
    `_prepare_inputs`, fetching real batches from the Atropos API instead.
    This dataset exists solely to satisfy the Trainer base-class contract
    which requires a non-empty `train_dataset`.
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps

    def __iter__(self):
        for _ in range(self.total_steps):
            yield {"prompt": ""}
            

# ---------------------------------------------------------------------------
# Helper: pass-through reward function
# ---------------------------------------------------------------------------

def _atropos_passthrough_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    A no-op reward function used when no explicit reward_funcs are passed.

    When running in Atropos mode, all reward computation happens inside the
    environment microservice.  This function returns 0.0 for every sample so
    that the base-class reward machinery does not conflict with the advantages
    computed from Atropos scores.

    Note: the advantages tensor injected by ``_convert_atropos_batch`` is
    already normalised; the base-class reward pipeline runs but its output is
    completely overridden by the pre-computed advantages.
    """
    return [0.0] * len(prompts)