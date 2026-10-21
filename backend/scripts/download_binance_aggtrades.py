"""Download public Binance aggregate trades for a UTC time range."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.binance_client import BinancePublicClient
from app.data.storage import PartitionedGzipJsonlWriter


def _utc_milliseconds(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def _iso_utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--start", required=True, help="Inclusive ISO-8601 UTC timestamp")
    parser.add_argument("--end", required=True, help="Inclusive ISO-8601 UTC timestamp")
    parser.add_argument("--output-root", type=Path, help="Defaults to data_raw/binance_spot/historical_trades")
    parser.add_argument("--request-interval-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if args.request_interval_seconds < 0.1:
        raise SystemExit("--request-interval-seconds must be at least 0.1 to limit public API traffic")
    start_ms, end_ms = _utc_milliseconds(args.start), _utc_milliseconds(args.end)
    if end_ms < start_ms:
        raise SystemExit("--end must be greater than or equal to --start")
    settings = load_settings()
    if args.symbol not in settings.symbols:
        raise SystemExit(f"{args.symbol} is not in the configured symbol list {settings.symbols}")
    output_root = args.output_root or (REPOSITORY_ROOT / "data_raw" / "binance_spot" / "historical_trades")
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    download_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_root = output_root / f"download_id={download_id}"
    writer = PartitionedGzipJsonlWriter(
        output_root, args.symbol, settings.lob_raw_file_max_bytes,
        flush_rows=settings.lob_flush_rows, flush_seconds=settings.lob_flush_seconds,
    )
    client = BinancePublicClient(settings)
    cursor: int | None = None
    count = 0
    first_trade_ms = last_trade_ms = None
    page = 0
    try:
        while True:
            params = {"symbol": args.symbol, "limit": 1000}
            if cursor is None:
                params["startTime"] = start_ms
            else:
                params["fromId"] = cursor
            params["endTime"] = end_ms
            rows = client.get_json("/api/v3/aggTrades", params)
            if not isinstance(rows, list):
                raise RuntimeError("Binance aggTrades endpoint returned an unexpected response")
            if not rows:
                break
            page += 1
            last_id_in_page = cursor
            for trade in rows:
                trade_id = int(trade["a"])
                trade_ms = int(trade["T"])
                last_id_in_page = trade_id
                if not start_ms <= trade_ms <= end_ms:
                    continue
                buyer_maker = bool(trade["m"])
                writer.write({
                    "kind": "trade",
                    "symbol": args.symbol,
                    "event_time_ms": trade_ms,
                    "trade_timestamp_ms": trade_ms,
                    "timestamp_utc": _iso_utc(trade_ms),
                    "trade_id": trade_id,
                    "price": str(trade["p"]),
                    "quantity": str(trade["q"]),
                    "is_buyer_maker": buyer_maker,
                    "aggressor_side": "sell" if buyer_maker else "buy",
                    "source": "binance_spot_rest_aggTrades",
                    "raw_event": trade,
                })
                count += 1
                first_trade_ms = trade_ms if first_trade_ms is None else min(first_trade_ms, trade_ms)
                last_trade_ms = trade_ms if last_trade_ms is None else max(last_trade_ms, trade_ms)
            if len(rows) < 1000:
                break
            if last_id_in_page is None or (cursor is not None and last_id_in_page < cursor):
                raise RuntimeError("Binance aggTrades pagination did not advance")
            cursor = int(last_id_in_page) + 1
            time.sleep(args.request_interval_seconds)
    finally:
        files = writer.file_metadata()

    manifest = {
        "schema_version": "binance_aggtrades_download.v1",
        "download_id": download_id,
        "source": "Binance Spot public REST GET /api/v3/aggTrades",
        "symbol": args.symbol,
        "requested_start_utc": _iso_utc(start_ms),
        "requested_end_utc": _iso_utc(end_ms),
        "first_trade_utc": _iso_utc(first_trade_ms) if first_trade_ms is not None else None,
        "last_trade_utc": _iso_utc(last_trade_ms) if last_trade_ms is not None else None,
        "row_count": count,
        "pages": page,
        "files": files,
        "historical_lob": False,
        "limitation": "Aggregate trades do not contain historical depth updates and cannot reconstruct a historical limit order book.",
    }
    manifest_path = output_root / "manifests" / f"symbol={args.symbol}" / f"aggtrades_{start_ms}_{end_ms}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Downloaded {count:,} aggregate trades across {page} pages for {args.symbol}.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
