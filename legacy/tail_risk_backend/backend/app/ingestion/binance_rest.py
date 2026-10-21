"""Public Binance REST client for initial Spot depth snapshots."""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests


class BinanceDepthClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 20,
        retries: int = 5,
        session=None,
        fallback_base_urls: tuple[str, ...] = (),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.base_urls = tuple(dict.fromkeys((self.base_url, *(url.rstrip("/") for url in fallback_base_urls))))
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.session = session or requests.Session()
        self._host_index = 0
        self._host_index_lock = threading.Lock()

    @property
    def current_base_url(self) -> str:
        with self._host_index_lock:
            return self.base_urls[self._host_index]

    def rotate_endpoint(self) -> str:
        """Move to the next official endpoint after a snapshot is sequence-stale."""
        with self._host_index_lock:
            self._host_index = (self._host_index + 1) % len(self.base_urls)
            return self.base_urls[self._host_index]

    def fetch_snapshot(self, symbol: str, limit: int = 5000) -> dict[str, Any]:
        if not 1 <= limit <= 5000:
            raise ValueError("Binance depth snapshot limit must be between 1 and 5000")
        params = {"symbol": symbol.upper(), "limit": limit}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            with self._host_index_lock:
                host_index = self._host_index
                self._host_index = (self._host_index + 1) % len(self.base_urls)
            base_url = self.base_urls[host_index]
            timeout = self.timeout_seconds if host_index == 0 else min(self.timeout_seconds, 3.0)
            try:
                response = self.session.get(
                    f"{base_url}/api/v3/depth",
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                snapshot = response.json()
                if "lastUpdateId" not in snapshot or "bids" not in snapshot or "asks" not in snapshot:
                    raise ValueError("Binance depth response is missing snapshot fields")
                with self._host_index_lock:
                    self._host_index = host_index
                return snapshot
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = RuntimeError(f"{base_url}: {exc}")
                if attempt + 1 < self.retries:
                    time.sleep(min(0.25 * 2**attempt + random.random() * 0.15, 2))
        raise RuntimeError(f"Could not fetch {symbol.upper()} Binance depth snapshot from configured endpoints: {last_error}") from last_error
