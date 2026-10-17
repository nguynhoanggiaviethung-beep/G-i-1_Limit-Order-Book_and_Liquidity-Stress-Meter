"""Background synchronization of closed Binance candles on a fixed interval."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.config import Settings
from app.data.binance_client import BinanceClient
from app.data.data_manager import DataManager
from app.features.realized_volatility import daily_realized_volatility


LOGGER = logging.getLogger(__name__)


class AutoSyncService:
    """Keep local candle and daily-feature Parquet files current."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = BinanceClient(
            settings.binance_base_url,
            settings.request_timeout_seconds,
            settings.request_retries,
        )
        self.manager = DataManager(
            self.client,
            settings.data_dir,
            settings.interval,
            settings.start_date,
        )
        self.status_path = settings.data_dir.parent / "results" / "metrics" / "auto_sync_status.json"
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._state: dict = {
            "enabled": settings.auto_sync_enabled,
            "status": "starting" if settings.auto_sync_enabled else "disabled",
            "interval_seconds": settings.auto_sync_interval_seconds,
            "grace_seconds": settings.auto_sync_grace_seconds,
            "symbols": list(settings.symbols),
            "last_started_at_utc": None,
            "last_success_at_utc": None,
            "last_error": None,
            "symbols_result": {},
        }
        self._load_saved_status()

    def _load_saved_status(self) -> None:
        if not self.status_path.exists():
            return
        try:
            saved = json.loads(self.status_path.read_text(encoding="utf-8"))
            for key in ("last_started_at_utc", "last_success_at_utc", "last_error", "symbols_result"):
                if key in saved:
                    self._state[key] = saved[key]
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read saved auto-sync status from %s", self.status_path)

    def get_status(self) -> dict:
        with self._state_lock:
            return dict(self._state)

    def _save_status(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(f".json.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(self.get_status(), indent=2), encoding="utf-8")
            temporary.replace(self.status_path)
        finally:
            temporary.unlink(missing_ok=True)

    def sync_once(self) -> dict:
        """Synchronize every configured symbol once; safe against overlapping calls."""
        if not self._run_lock.acquire(blocking=False):
            return {"status": "already_running"}
        started = datetime.now(timezone.utc).isoformat()
        results: dict[str, dict] = {}
        errors: dict[str, str] = {}
        with self._state_lock:
            self._state.update(status="syncing", last_started_at_utc=started, last_error=None)
        self._save_status()
        try:
            for symbol in self.settings.symbols:
                try:
                    result = self.manager.sync(symbol)
                    processed_path = Path(result["processed_file"])
                    candles = pd.read_parquet(processed_path)
                    features = daily_realized_volatility(candles)
                    feature_path = (
                        self.settings.data_dir
                        / "features"
                        / f"symbol={symbol}"
                        / "daily_realized_volatility.parquet"
                    )
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = feature_path.with_suffix(f".parquet.{os.getpid()}.tmp")
                    try:
                        features.to_parquet(temporary, index=False)
                        temporary.replace(feature_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    result["latest_closed_candle_utc"] = (
                        pd.to_datetime(candles["open_time"], utc=True).max().isoformat()
                        if not candles.empty else None
                    )
                    result["feature_rows"] = len(features)
                    results[symbol] = result
                except Exception as exc:  # Continue syncing other symbols if one fails.
                    LOGGER.exception("Automatic Binance sync failed for %s", symbol)
                    errors[symbol] = f"{type(exc).__name__}: {exc}"

            completed = datetime.now(timezone.utc).isoformat()
            if errors and not results:
                overall_status = "error"
            elif errors:
                overall_status = "degraded"
            else:
                overall_status = "ok"
            with self._state_lock:
                self._state.update(
                    status=overall_status,
                    last_success_at_utc=completed if results else self._state["last_success_at_utc"],
                    last_error=errors or None,
                    symbols_result=results,
                )
            self._save_status()
            return {"status": overall_status, "symbols": results, "errors": errors}
        finally:
            self._run_lock.release()

    @staticmethod
    def seconds_until_next_run(now: datetime, interval: int, grace: int) -> float:
        """Align polling to interval boundaries, then allow the candle to close."""
        epoch = now.timestamp()
        next_boundary = (int(epoch // interval) + 1) * interval
        return max(1.0, next_boundary + grace - epoch)

    async def run_forever(self) -> None:
        """Run immediately at startup, then after each configured interval boundary."""
        while True:
            try:
                await asyncio.to_thread(self.sync_once)
            except Exception:
                LOGGER.exception("Automatic Binance sync cycle failed unexpectedly")
                with self._state_lock:
                    self._state.update(
                        status="error",
                        last_error="Unexpected synchronization worker failure; retry scheduled",
                    )
                self._save_status()
            delay = self.seconds_until_next_run(
                datetime.now(timezone.utc),
                self.settings.auto_sync_interval_seconds,
                self.settings.auto_sync_grace_seconds,
            )
            LOGGER.info("Next automatic Binance sync in %.1f seconds", delay)
            await asyncio.sleep(delay)
