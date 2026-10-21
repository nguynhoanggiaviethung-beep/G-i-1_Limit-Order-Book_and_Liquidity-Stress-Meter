"""One safe path for initial backfills and incremental Binance synchronization."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from app.data.binance_client import BinanceClient, RAW_COLUMNS
from app.data.data_validator import normalize_klines, validate_klines


LOGGER = logging.getLogger(__name__)


class DataManager:
    def __init__(self, client: BinanceClient, data_dir: Path, interval: str = "5m", start_date: str = "2019-01-01") -> None:
        self.client = client
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.start_date = start_date

    def _path(self, area: str, symbol: str) -> Path:
        return self.data_dir / area / f"symbol={symbol}" / f"interval={self.interval}" / "klines.parquet"

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".parquet.{os.getpid()}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def sync(self, symbol: str) -> dict[str, int | str]:
        raw_path = self._path("raw", symbol)
        processed_path = self._path("processed", symbol)
        old_raw = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame(columns=RAW_COLUMNS + ["symbol", "interval"])
        old_processed = pd.read_parquet(processed_path) if processed_path.exists() else pd.DataFrame()
        if not old_processed.empty:
            latest = pd.to_datetime(old_processed["open_time"], utc=True).max()
            # Re-fetch the last candle to cover interrupted writes and boundary uncertainty.
            start = int(latest.timestamp() * 1000)
        else:
            start = self.start_date

        end = self.client.server_time()
        incoming = self.client.fetch_klines(symbol, self.interval, start_time=start, end_time=end)
        if incoming.empty and old_raw.empty:
            raise RuntimeError(f"Binance returned no candles for {symbol} from {start}")
        combined = pd.concat([old_raw, incoming], ignore_index=True)
        if not combined.empty:
            combined["open_time_ms"] = pd.to_numeric(combined["open_time_ms"], errors="raise").astype("int64")
            combined = combined.drop_duplicates(["symbol", "interval", "open_time_ms"], keep="last")
            combined = combined.sort_values("open_time_ms").reset_index(drop=True)
        processed = normalize_klines(combined)
        now = self.client.server_time()
        interval_delta = pd.Timedelta(self.interval)
        processed = processed[processed["open_time"] + interval_delta <= pd.Timestamp(now)].reset_index(drop=True)
        # Raw retains the original Binance rows; processed contains only closed, normalized candles.
        if not processed.empty:
            valid_keys = set(processed["open_time"].astype("int64") // 1_000_000)
            combined = combined[combined["open_time_ms"].isin(valid_keys)].reset_index(drop=True)
        report = validate_klines(processed, self.interval, now=now, raise_on_error=False)
        self._atomic_parquet(combined, raw_path)
        self._atomic_parquet(processed, processed_path)
        LOGGER.info("Synced %s: %s rows; %s missing intervals; %s duplicates", symbol, len(processed), report.get("missing_intervals", 0), report["duplicates"])
        return {"symbol": symbol, "rows": len(processed), "new_rows": len(incoming), "gaps": report["gaps"],
                "raw_file": str(raw_path), "processed_file": str(processed_path)}
