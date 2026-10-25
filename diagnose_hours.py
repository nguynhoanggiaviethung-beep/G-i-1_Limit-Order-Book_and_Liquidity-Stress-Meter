"""Kiểm tra từng giờ của một ngày: file orderbook có snapshot dùng được hay không.

Chạy từ thư mục gốc repo:
    python diagnose_hours.py --day 2026-04-25
    python diagnose_hours.py --day 2026-04-21        # ngày replay tốt, để so sánh
    python diagnose_hours.py --day 2026-04-25 --hour 3 --verbose
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import zstandard as zstd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw" / "cryptohftdata" / "binance_spot"
ZSTD_MAGIC = bytes.fromhex("28b52ffd")


def read(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != ZSTD_MAGIC:
        return pq.read_table(path).to_pandas()
    with tempfile.TemporaryFile(dir=path.parent) as scratch:
        with path.open("rb") as source:
            zstd.ZstdDecompressor().copy_stream(source, scratch)
        scratch.seek(0)
        return pq.ParquetFile(scratch).read().to_pandas()


def summarize(frame: pd.DataFrame) -> dict:
    out = {"rows": len(frame)}
    event_type = frame["event_type"].astype("string").fillna("(null)")
    out["snapshot_rows"] = int((event_type == "snapshot").sum())
    out["update_rows"] = int((event_type == "update").sum())
    other = sorted(set(event_type.unique()) - {"snapshot", "update"})
    out["other_types"] = ",".join(other) or "-"
    snap = frame[event_type == "snapshot"].copy()
    usable = 0
    groups = 0
    if not snap.empty:
        snap["side"] = snap["side"].astype("string").str.lower()
        for _, block in snap.groupby("received_time", sort=False):
            groups += 1
            has_both = (block["side"] == "bid").any() and (block["side"] == "ask").any()
            ids = pd.to_numeric(block["last_update_id"], errors="coerce").dropna()
            ids_final = pd.to_numeric(block["final_update_id"], errors="coerce").dropna()
            if has_both and (len(ids) or len(ids_final)):
                usable += 1
    out["snapshot_groups"] = groups
    out["snapshot_usable"] = usable
    upd = frame[event_type == "update"]
    out["update_missing_ids"] = int(
        pd.to_numeric(upd["first_update_id"], errors="coerce").isna().sum()
        + pd.to_numeric(upd["final_update_id"], errors="coerce").isna().sum()
    ) if not upd.empty else 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hour", type=int)
    parser.add_argument("--verbose", action="store_true", help="in cột, dtype và vài dòng snapshot đầu tiên")
    args = parser.parse_args()
    hours = [args.hour] if args.hour is not None else range(24)
    rows = []
    for hour in hours:
        path = RAW / args.day / f"{hour:02d}" / f"{args.symbol}_orderbook.parquet.zst"
        if not path.is_file():
            rows.append({"hour": hour, "rows": "MISSING"})
            continue
        frame = read(path)
        if args.verbose:
            print(f"\n== {path}\n{frame.dtypes}\n")
            snap = frame[frame["event_type"].astype("string") == "snapshot"]
            print(snap.head(8).to_string())
        rows.append({"hour": hour, **summarize(frame)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n{args.symbol} {args.day}: snapshot_usable = số snapshot đủ cả bid/ask và có update-id\n")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())