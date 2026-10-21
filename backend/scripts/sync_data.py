"""Initial backfill or incremental update for configured Binance symbols."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.data.binance_client import BinanceClient  # noqa: E402
from app.data.data_manager import DataManager  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="Symbol to sync; may be repeated")
    parser.add_argument("--start-date", help="Override configured historical start date (YYYY-MM-DD)")
    parser.add_argument("--interval", help="Override configured candle interval")
    args = parser.parse_args()
    configure_logging()
    settings = load_settings()
    symbols = [value.upper() for value in (args.symbol or settings.symbols)]
    interval = args.interval or settings.interval
    client = BinanceClient(settings.binance_base_url, settings.request_timeout_seconds, settings.request_retries)
    manager = DataManager(client, settings.data_dir, interval, args.start_date or settings.start_date)
    for symbol in symbols:
        try:
            result = manager.sync(symbol)
            logging.getLogger(__name__).info("Sync complete: %s", result)
        except Exception:
            logging.getLogger(__name__).exception("Sync failed for %s", symbol)
            raise


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("sync_data.py")
