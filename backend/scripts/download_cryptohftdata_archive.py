"""Download hourly Binance Spot L2 and trade files from CryptoHFTData."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT


API_ROOT = "https://api.cryptohftdata.com/v1/download"
ZSTD_MAGIC = bytes.fromhex("28b52ffd")
PARQUET_MAGIC = b"PAR1"


def _display_path(path: Path) -> str:
    """Repo-relative path when possible; absolute otherwise (e.g. output on another drive)."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


class RequestPacer:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_request_at)
            self.next_request_at = scheduled + self.interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _valid_archive_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in {PARQUET_MAGIC, ZSTD_MAGIC} and path.stat().st_size > 4
    except OSError:
        return False


def _download_one(
    file_key: str,
    destination: Path,
    pacer: RequestPacer,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    retries: int,
) -> tuple[str, int]:
    if _valid_archive_file(destination):
        size = destination.stat().st_size
        record = {
            "file": file_key,
            "local_path": _display_path(destination),
            "status": "cached",
            "bytes": size,
            "downloaded_at_utc": None,
        }
        with manifest_lock:
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "cached", size

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{API_ROOT}?{urlencode({'file': file_key})}"
    last_error = "unknown error"
    for attempt in range(retries + 1):
        pacer.wait()
        try:
            digest = hashlib.sha256()
            written = 0
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                if response.status_code == 404:
                    record = {"file": file_key, "status": "missing", "http_status": 404}
                    with manifest_lock:
                        with manifest_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    return "missing", 0
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After")
                    time.sleep(min(float(retry_after), 30.0) if retry_after else min(2**attempt, 20))
                    continue
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
            if not _valid_archive_file(temporary):
                temporary.unlink(missing_ok=True)
                raise ValueError("Downloaded file is not a Parquet or Zstandard archive")
            os.replace(temporary, destination)
            record: dict[str, Any] = {
                "file": file_key,
                "local_path": _display_path(destination),
                "status": "downloaded",
                "bytes": written,
                "sha256": digest.hexdigest(),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with manifest_lock:
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return "downloaded", written
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            last_error = f"{type(exc).__name__}: {exc}"
            break

    record = {"file": file_key, "status": "error", "error": last_error}
    with manifest_lock:
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return "error", 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat, help="First UTC date, inclusive")
    parser.add_argument("--end-date", required=True, type=date.fromisoformat, help="Last UTC date, inclusive")
    parser.add_argument("--symbol", action="append", choices=("BTCUSDT", "ETHUSDT"), help="Repeat for each symbol; defaults to both")
    parser.add_argument("--data-type", action="append", choices=("orderbook", "trades"), help="Repeat to choose types; defaults to both")
    parser.add_argument("--output-root", type=Path, help="Defaults to data_raw/cryptohftdata/binance_spot")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent downloads; request starts remain rate-limited")
    parser.add_argument("--requests-per-minute", type=int, default=48, help="Anonymous provider limit is 60/minute")
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    if args.workers < 1 or not 1 <= args.requests_per_minute <= 55 or args.retries < 0:
        raise SystemExit("workers/retries must be non-negative and request rate must be 1–55 per minute")

    symbols = tuple(dict.fromkeys(args.symbol or ("BTCUSDT", "ETHUSDT")))
    data_types = tuple(dict.fromkeys(args.data_type or ("orderbook", "trades")))
    output_root = args.output_root or (REPOSITORY_ROOT / "data_raw" / "cryptohftdata" / "binance_spot")
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = output_root.parent / f"download_manifest_{args.start_date}_{args.end_date}_{run_id}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    requests_to_make: list[tuple[str, Path]] = []
    for day in _date_range(args.start_date, args.end_date):
        day_text = day.isoformat()
        for hour in range(24):
            hour_text = f"{hour:02d}"
            for symbol in symbols:
                for data_type in data_types:
                    file_name = f"{symbol}_{data_type}.parquet.zst"
                    key = f"binance_spot/{day_text}/{hour_text}/{symbol}_{data_type}.parquet"
                    requests_to_make.append((key, output_root / day_text / hour_text / file_name))

    pacer = RequestPacer(args.requests_per_minute)
    manifest_lock = threading.Lock()
    counts = {"downloaded": 0, "cached": 0, "missing": 0, "error": 0}
    total_bytes = 0
    print(f"Requesting {len(requests_to_make):,} hourly files for {', '.join(symbols)} ({', '.join(data_types)}).")
    print(f"Source: CryptoHFTData Binance Spot; raw files are kept outside Git at {output_root}.")
    print(f"Anonymous rate: up to {args.requests_per_minute} requests/minute; manifest: {manifest_path}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_download_one, key, path, pacer, manifest_path, manifest_lock, args.retries)
            for key, path in requests_to_make
        ]
        for index, future in enumerate(as_completed(futures), 1):
            status, size = future.result()
            counts[status] += 1
            total_bytes += size
            if index % 25 == 0 or index == len(futures):
                print(
                    f"Progress {index:,}/{len(futures):,}: downloaded={counts['downloaded']:,}, "
                    f"cached={counts['cached']:,}, missing={counts['missing']:,}, "
                    f"errors={counts['error']:,}, bytes={total_bytes:,}",
                    flush=True,
                )
    print(json.dumps({**counts, "bytes": total_bytes, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())