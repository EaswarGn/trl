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

import requests
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal Atropos API client
# ---------------------------------------------------------------------------

class AtroposAPIClient:
    """
    Thin HTTP client for the Atropos trajectory API.

    Only the endpoints needed by the trainer are implemented:
      - GET  /batch     – pull a ready batch of scored trajectories
      - POST /register  – register this trainer with the API server
    """

    def __init__(
        self,
        base_url: str,
        trainer_id: str = "trl_grpo",
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.trainer_id = trainer_id
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    "GET %s failed (attempt %d/%d): %s – retrying in 2s",
                    url, attempt + 1, self.max_retries, exc,
                )
                time.sleep(2.0)

    def _post(self, endpoint: str, json: Any = None, **kwargs) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=json, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    "POST %s failed (attempt %d/%d): %s – retrying in 2s",
                    url, attempt + 1, self.max_retries, exc,
                )
                time.sleep(2.0)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def health(self) -> bool:
        """Return True if the API server is reachable."""
        try:
            self._get("/", timeout=5.0)
            return True
        except Exception:
            return False

    def register(self, **registration_kwargs) -> Dict[str, Any]:
        """
        Register this trainer with the Atropos API.

        Sends the full Registration schema expected by the Atropos API
        server.  All keyword arguments are forwarded as JSON fields.

        Typical fields expected by the server:
            wandb_group           : str   – W&B group name
            wandb_project         : str   – W&B project name
            batch_size            : int   – total sequences per fetch
            max_token_len         : int   – maximum token length
            checkpoint_dir        : str   – path for checkpoints
            save_checkpoint_interval : int – steps between checkpoints
            starting_step         : int   – step to resume from
            num_steps             : int   – total training steps

        Returns the server's registration acknowledgment dict.
        """
        resp = self._post("/register", json=registration_kwargs, timeout=30.0)
        return resp.json()

    def fetch_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Poll /batch once and return the payload list if a batch is ready.
        Returns None if the server responded with 204 or the batch value is
        falsy (None / empty list).

        Uses the internal ``_get`` helper for retry-on-transient-failure logic
        (up to ``max_retries`` attempts with 2s backoff).
        """
        try:
            resp = self._get("/batch", timeout=30.0)
        except requests.RequestException:
            # Server may be temporarily unavailable; return None so the caller
            # can retry on the next poll cycle.
            return None

        if resp.status_code == 204:
            return None
        # _get already calls raise_for_status(), so if we get here the status
        # is OK.  Still handle the 204 case explicitly for the edge where
        # the server returns 204 after a redirect (not fully canonical).
        data = resp.json()
        # The API returns {"batch": [...]} or {"batch": None}
        if isinstance(data, dict):
            batch = data.get("batch")
            if batch is None:
                return None
            return batch
        return data

    def wait_for_batch(self) -> List[Dict[str, Any]]:
        """
        Block until a batch is ready, polling every `poll_interval` seconds.
        Raises TimeoutError if `timeout` seconds elapse without a batch.
        """
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            batch = self.fetch_batch()
            if batch is not None:
                return batch
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"No batch available from Atropos API at {self.base_url} "
            f"within {self.timeout}s."
        )

    def disconnect_trainer(self) -> None:
        """
        Notify the Atropos API server that this trainer is shutting down.

        This is a best-effort call — exceptions are swallowed so that a
        failure to disconnect does not interrupt the training shutdown flow.
        """
        try:
            self._post("/disconnect-trainer", timeout=5.0)
        except Exception:
            pass
