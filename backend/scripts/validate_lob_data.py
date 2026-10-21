"""Validate normalized one-second LOB Parquet partitions without modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.grid_store import GridStore


def validate_partition(path: Path, symbol: str) -> dict:
    frame = pd.read_parquet(path)
    required = ["timestamp_utc", "book_valid"]
    required.extend(f"{side}_{field}_{level}" for side in ("bid", "ask") for field in ("price", "qty") for level in range(1, 21))
    missing = [name for name in required if name not in frame]
    if missing:
        return {
            "path": str(path), "symbol": symbol, "rows": int(len(frame)),
            "status": "failed", "missing_columns": missing,
        }
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    invalid_timestamp = int(timestamps.isna().sum())
    duplicate_timestamp = int(timestamps.duplicated().sum())
    valid_timestamps = timestamps.dropna()
    out_of_order = int((valid_timestamps.diff().dropna() < pd.Timedelta(0)).sum())
    sorted_timestamp = timestamps.dropna().sort_values()
    missing_seconds = int(max(0, (sorted_timestamp.diff().dt.total_seconds().fillna(1) != 1).sum()))
    numeric = frame[[f"{side}_{field}_{level}" for side in ("bid", "ask") for field in ("price", "qty") for level in range(1, 21)]].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = int(numeric.isna().any(axis=1).sum())
    bid_prices = numeric[[f"bid_price_{level}" for level in range(1, 21)]].to_numpy(dtype=float)
    ask_prices = numeric[[f"ask_price_{level}" for level in range(1, 21)]].to_numpy(dtype=float)
    bid_qty = numeric[[f"bid_qty_{level}" for level in range(1, 21)]].to_numpy(dtype=float)
    ask_qty = numeric[[f"ask_qty_{level}" for level in range(1, 21)]].to_numpy(dtype=float)
    if frame["book_valid"].dtype == object:
        valid_flag = frame["book_valid"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy()
    else:
        valid_flag = frame["book_valid"].fillna(False).astype(bool).to_numpy()
    level_order_bad = int((np.any(np.diff(bid_prices, axis=1) >= 0, axis=1) | np.any(np.diff(ask_prices, axis=1) <= 0, axis=1)).sum())
    crossed = int((bid_prices[:, 0] >= ask_prices[:, 0]).sum())
    bad_quantity = int((
        np.any(~np.isfinite(bid_qty) | (bid_qty < 0), axis=1)
        | np.any(~np.isfinite(ask_qty) | (ask_qty < 0), axis=1)
        | (bid_qty[:, 0] <= 0)
        | (ask_qty[:, 0] <= 0)
    ).sum())
    bad_price = int((np.any(~np.isfinite(bid_prices) | (bid_prices <= 0), axis=1) | np.any(~np.isfinite(ask_prices) | (ask_prices <= 0), axis=1)).sum())
    failed_rows = int(
        (~valid_flag).sum() + invalid_timestamp + duplicate_timestamp + out_of_order
        + invalid_numeric + level_order_bad + crossed + bad_quantity + bad_price
    )
    return {
        "path": str(path), "symbol": symbol, "rows": int(len(frame)),
        "first_timestamp_utc": sorted_timestamp.min().isoformat() if len(sorted_timestamp) else None,
        "last_timestamp_utc": sorted_timestamp.max().isoformat() if len(sorted_timestamp) else None,
        "invalid_book_flags": int((~valid_flag).sum()),
        "invalid_timestamps": invalid_timestamp,
        "duplicate_timestamps": duplicate_timestamp,
        "out_of_order_timestamps": out_of_order,
        "missing_or_non_one_second_steps": missing_seconds,
        "rows_with_non_numeric_levels": invalid_numeric,
        "rows_with_level_order_errors": level_order_bad,
        "rows_with_crossed_or_locked_book": crossed,
        "rows_with_invalid_quantity": bad_quantity,
        "rows_with_invalid_price": bad_price,
        "status": "failed" if failed_rows else "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--grid-root", type=Path, help="Defaults to configured data directory/grid")
    parser.add_argument("--report", type=Path, help="Defaults to outputs/manifests/lob_validation_<UTC>.json")
    args = parser.parse_args()
    settings = load_settings()
    root = args.grid_root or (settings.data_dir / "grid")
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    store = GridStore(root)
    symbols = (args.symbol,) if args.symbol else settings.symbols
    partitions = []
    for symbol in symbols:
        partitions.extend(validate_partition(path, symbol) for path in store._files(symbol))
    if not partitions:
        print(f"No book_1s.parquet partitions found under {root}")
        return 2
    report = {
        "schema_version": "lob_validation.v1",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "grid_root": str(root),
        "status": "passed" if all(item["status"] == "passed" for item in partitions) else "failed",
        "partition_count": len(partitions),
        "row_count": sum(item.get("rows", 0) for item in partitions),
        "partitions": partitions,
    }
    destination = args.report or (REPOSITORY_ROOT / "outputs" / "manifests" / f"lob_validation_{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}.json")
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    print(f"LOB validation {report['status']}: {report['row_count']:,} rows across {report['partition_count']} partitions")
    print(f"Report: {destination}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
