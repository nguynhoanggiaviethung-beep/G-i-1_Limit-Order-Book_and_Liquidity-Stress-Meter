"""Quét data_raw: ngày nào có snapshot order book (điều kiện bắt buộc để replay).

Chạy từ thư mục gốc repo. Chỉ đọc cột event_type nên nhanh hơn diagnose_hours.py.

    python scan_snapshots.py                       # mặc định: giờ 0,6,12,18 của mọi ngày, BTCUSDT
    python scan_snapshots.py --hours 0-23          # kiểm tra đủ 24 giờ (chậm hơn nhiều)
    python scan_snapshots.py --start 2026-04-23 --end 2026-05-05

Kết quả cũng được ghi ra data_interim/cryptohftdata_6m/snapshot_scan.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard as zstd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw" / "cryptohftdata" / "binance_spot"
ZSTD_MAGIC = bytes.fromhex("28b52ffd")


def parse_hours(text: str) -> list[int]:
    hours: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            hours.extend(range(int(a), int(b) + 1))
        else:
            hours.append(int(part))
    return sorted(set(h for h in hours if 0 <= h <= 23))


def count_snapshot_rows(path: Path) -> int:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == ZSTD_MAGIC:
        with tempfile.TemporaryFile(dir=path.parent) as scratch:
            with path.open("rb") as source:
                zstd.ZstdDecompressor().copy_stream(source, scratch)
            scratch.seek(0)
            table = pq.ParquetFile(scratch).read(columns=["event_type"])
    else:
        table = pq.read_table(path, columns=["event_type"])
    return int(pc.sum(pc.equal(table["event_type"], "snapshot")).as_py() or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2026, 4, 19))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 10, 18))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hours", default="0,6,12,18", help="vd: 0,6,12,18 hoặc 0-23")
    args = parser.parse_args()
    hours = parse_hours(args.hours)

    out_path = ROOT / "data_interim" / "cryptohftdata_6m" / "snapshot_scan.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,hours_checked,hours_with_snapshot,missing_files,status"]
    print(f"{args.symbol}: quét giờ {hours} của từng ngày\n")
    print(f"{'date':<12}{'checked':>8}{'with_snap':>10}{'missing':>8}  status")
    previous = None
    for offset in range((args.end - args.start).days + 1):
        day = (args.start + dt.timedelta(days=offset)).isoformat()
        checked = with_snap = missing = 0
        for hour in hours:
            path = RAW / day / f"{hour:02d}" / f"{args.symbol}_orderbook.parquet.zst"
            if not path.is_file():
                missing += 1
                continue
            checked += 1
            try:
                if count_snapshot_rows(path) > 0:
                    with_snap += 1
            except Exception as exc:  # corrupt/partial file
                print(f"  ! {path}: {type(exc).__name__}: {exc}")
        if checked == 0:
            status = "KHÔNG CÓ FILE"
        elif with_snap == 0:
            status = "THIẾU SNAPSHOT"
        elif with_snap < checked:
            status = "snapshot một phần"
        else:
            status = "ok"
        lines.append(f"{day},{checked},{with_snap},{missing},{status}")
        if status != previous or offset % 10 == 0 or status != "ok":
            print(f"{day:<12}{checked:>8}{with_snap:>10}{missing:>8}  {status}", flush=True)
        previous = status
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())