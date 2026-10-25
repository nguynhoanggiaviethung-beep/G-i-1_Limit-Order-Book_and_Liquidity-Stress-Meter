"""Tải và replay dữ liệu CryptoHFTData 6 tháng trên máy local, theo từng đợt 2 ngày.

Chạy lại bao nhiêu lần cũng được: ngày nào đã có grid cho cả BTCUSDT và ETHUSDT thì bỏ qua,
file raw đã tải thì báo "cached". Log đầy đủ từng đợt nằm ở data_interim/cryptohftdata_6m/logs/
(thư mục này bị .gitignore loại).

Ví dụ:
    python run_6m_local.py
    python run_6m_local.py --start 2026-05-01 --end 2026-05-10 --workers 1
    python run_6m_local.py --skip-download      # chỉ replay, dùng file raw đã có
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent
GRID_ROOT = ROOT / "data_processed" / "cryptohftdata_6m" / "grid"
MANIFEST_ROOT = ROOT / "outputs" / "manifests" / "archive_6m"
LOG_ROOT = ROOT / "data_interim" / "cryptohftdata_6m" / "logs"
SYMBOLS = ("BTCUSDT", "ETHUSDT")

DEFAULT_START = "2026-04-19"
DEFAULT_END = "2026-10-18"
DEFAULT_WORKERS = 2  # mỗi luồng replay dùng vài GB RAM; 2 luồng là an toàn với máy 16 GB

# Ép UTF-8 cho tiến trình con để đường dẫn có dấu (vd. "Gói 1") và log không gây UnicodeError trên Windows.
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def day_done(day: dt.date) -> bool:
    return all(
        any((GRID_ROOT / f"symbol={symbol}" / f"date={day.isoformat()}").glob("*.parquet"))
        for symbol in SYMBOLS
    )


def make_chunks(start: dt.date, end: dt.date, days: int) -> list[tuple[str, str]]:
    chunks = []
    cursor = start
    while cursor <= end:
        last = min(cursor + dt.timedelta(days=days - 1), end)
        span = (last - cursor).days + 1
        if not all(day_done(cursor + dt.timedelta(days=offset)) for offset in range(span)):
            chunks.append((cursor.isoformat(), last.isoformat()))
        cursor = last + dt.timedelta(days=1)
    return chunks


def run_logged(command: list[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        return subprocess.run(
            command, cwd=ROOT, env=CHILD_ENV, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def log_tail(log_path: pathlib.Path, chars: int = 3000) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")[-chars:]
    except OSError:
        return "(không đọc được log)"


def download_chunk(index: int, total: int, start: str, end: str) -> bool:
    log_path = LOG_ROOT / f"download_{start}_{end}.log"
    print(f"[Tải {index}/{total}] {start} -> {end} ...", flush=True)
    code = run_logged(
        [sys.executable, "backend/scripts/download_cryptohftdata_archive.py",
         "--start-date", start, "--end-date", end],
        log_path,
    )
    if code == 0:
        print(f"[Tải {index}/{total}] xong: {start} -> {end}", flush=True)
        return True
    print(
        f"[Tải {index}/{total}] LỖI ({start} -> {end}), bỏ qua replay đợt này; chạy lại script để tải tiếp.\n"
        f"{log_tail(log_path)}\n  (log đầy đủ: {log_path})",
        flush=True,
    )
    return False


def replay_chunk(index: int, total: int, start: str, end: str) -> tuple[bool, str]:
    log_path = LOG_ROOT / f"replay_{start}_{end}.log"
    print(f"[Replay {index}/{total}] bắt đầu: {start} -> {end}", flush=True)
    code = run_logged(
        [sys.executable, "backend/scripts/replay_cryptohftdata_archive.py",
         "--start-date", start, "--end-date", end,
         "--output-root", str(GRID_ROOT),
         "--manifest-root", str(MANIFEST_ROOT / f"part_{start}_{end}")],
        log_path,
    )
    if code == 0:
        return True, f"[Replay {index}/{total}] HOÀN TẤT: {start} -> {end}"
    return False, (
        f"[Replay {index}/{total}] LỖI (exit {code}) {start} -> {end}\n"
        f"{log_tail(log_path)}\n  (log đầy đủ: {log_path})"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=DEFAULT_START, type=dt.date.fromisoformat)
    parser.add_argument("--end", default=DEFAULT_END, type=dt.date.fromisoformat)
    parser.add_argument("--days", type=int, default=2, help="số ngày mỗi đợt")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="số luồng replay song song")
    parser.add_argument("--skip-download", action="store_true", help="không tải, chỉ replay file raw đã có")
    args = parser.parse_args()
    if args.end < args.start or args.days < 1 or args.workers < 1:
        raise SystemExit("--end phải >= --start; --days và --workers phải >= 1")

    chunks = make_chunks(args.start, args.end, args.days)
    total = len(chunks)
    print(f"=== CÒN {total} ĐỢT CẦN XỬ LÝ (đợt đã xong được bỏ qua tự động) ===", flush=True)
    failures: list[str] = []

    # Tải đợt i trong luồng chính; replay các đợt đã tải chạy nền song song, nên không phải
    # chờ tải xong cả 6 tháng mới bắt đầu replay.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index, (start, end) in enumerate(chunks, 1):
            if not args.skip_download and not download_chunk(index, total, start, end):
                failures.append(f"tải {start} -> {end}")
                continue
            futures[pool.submit(replay_chunk, index, total, start, end)] = (start, end)
        for future in as_completed(futures):
            ok, message = future.result()
            print(message, flush=True)
            if not ok:
                failures.append("replay {} -> {}".format(*futures[future]))

    remaining = [
        day.isoformat()
        for day in (args.start + dt.timedelta(days=n) for n in range((args.end - args.start).days + 1))
        if not day_done(day)
    ]
    if failures or remaining:
        print(f"\nCHƯA XONG: {len(failures)} đợt lỗi, còn {len(remaining)} ngày chưa có grid đủ 2 symbol.")
        for item in failures:
            print(f"  - {item}")
        if remaining:
            print(f"  Ngày còn thiếu (tối đa 10): {', '.join(remaining[:10])}")
            print("  (ngày nhà cung cấp không có file hoặc 0 giây hợp lệ cũng nằm ở đây; xem QA trong outputs/manifests/archive_6m)")
        print("Chạy lại script để tiếp tục.")
        return 1
    print(f"\nHOÀN TẤT {args.start} -> {args.end}: grid ở {GRID_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())