"""Collect Binance Spot L2 depth and trade events into replayable JSONL.GZ files."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.websocket_collector import BinanceWebSocketCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", choices=("BTCUSDT", "ETHUSDT"), help="Repeat to collect multiple symbols; defaults to config")
    parser.add_argument("--duration-seconds", type=float, default=0, help="0 collects until Ctrl+C")
    parser.add_argument("--output-root", type=Path, help="Defaults to data_raw/binance_spot")
    parser.add_argument("--interim-root", type=Path, help="Defaults to data_interim/binance_spot")
    parser.add_argument("--reconnect-max-seconds", type=float, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    symbols = tuple(args.symbol or settings.symbols)
    if any(symbol not in settings.symbols for symbol in symbols):
        raise SystemExit(f"Requested symbols must be configured in {REPOSITORY_ROOT / 'config' / 'symbols.yaml'}")
    output_root = args.output_root or settings.raw_data_dir
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    collector = BinanceWebSocketCollector(
        settings=settings,
        output_root=output_root,
        interim_root=args.interim_root,
        duration_seconds=args.duration_seconds,
        reconnect_max_seconds=args.reconnect_max_seconds,
    )
    try:
        manifests = asyncio.run(collector.run(symbols))
    except KeyboardInterrupt:
        print("Capture stopped by user. Partial files and manifests were saved.")
        print(f"Raw source files: {collector.capture_root}")
        print(f"Normalized replay events: {collector.interim_capture_root / 'events'}")
        return 130
    for manifest in manifests:
        print(json.dumps({
            "symbol": manifest["symbol"],
            "started_at_utc": manifest["started_at_utc"],
            "ended_at_utc": manifest["ended_at_utc"],
            "counts": manifest["counts"],
            "capture_root": str(collector.capture_root),
            "interim_event_root": manifest["interim_event_root"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
