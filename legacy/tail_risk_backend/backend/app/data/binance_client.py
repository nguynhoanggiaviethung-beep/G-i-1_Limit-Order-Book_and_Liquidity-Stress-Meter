"""Public Binance Spot klines client with pagination and retry handling."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_volume", "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]


def to_milliseconds(value: datetime | str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


class BinanceClient:
    """Fetch public OHLCV klines; no API key is needed for these endpoints."""

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: float = 20.0,
        retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "5m",
        start_time: datetime | str | int | None = None,
        end_time: datetime | str | int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch all pages in [start_time, end_time], preserving API fields."""
        if not 1 <= limit <= 1000:
            raise ValueError("Binance Spot kline limit must be between 1 and 1000")
        start_ms, end_ms = to_milliseconds(start_time), to_milliseconds(end_time)
        cursor = start_ms
        rows: list[list[Any]] = []
        while True:
            params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            response = self.session.get(
                f"{self.base_url}/api/v3/klines", params=params, timeout=self.timeout
            )
            if response.status_code == 418:
                raise RuntimeError("Binance IP auto-ban response (418); stop requests and retry later")
            response.raise_for_status()
            page = response.json()
            if not page:
                break
            rows.extend(page)
            last_open_ms = int(page[-1][0])
            if len(page) < limit or (end_ms is not None and last_open_ms >= end_ms):
                break
            next_cursor = last_open_ms + 1
            if cursor is not None and next_cursor <= cursor:
                raise RuntimeError("Binance pagination cursor did not advance")
            cursor = next_cursor
            LOGGER.info("Fetched %s %s klines through %s", symbol, interval, last_open_ms)
            time.sleep(0.08)
        frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
        if not frame.empty:
            frame["symbol"] = symbol.upper()
            frame["interval"] = interval
            if start_ms is not None:
                frame = frame[frame["open_time_ms"] >= start_ms]
            if end_ms is not None:
                frame = frame[frame["open_time_ms"] <= end_ms]
            frame = frame.reset_index(drop=True)
        return frame

    def server_time(self) -> datetime:
        response = self.session.get(f"{self.base_url}/api/v3/time", timeout=self.timeout)
        response.raise_for_status()
        return datetime.fromtimestamp(response.json()["serverTime"] / 1000, tz=timezone.utc)
