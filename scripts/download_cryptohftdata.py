"""Resumable downloader for CryptoHFTData hourly Binance Spot files.

This fetches raw source files only. It does not manufacture or normalize data.
The free anonymous REST tier is rate-limited; use CRYPTOHFTDATA_API_KEY from
the local environment only if the provider has issued one.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.cryptohftdata.com/v1/download"


def hours(start: date, end: date):
    cursor = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    while cursor < stop:
        yield cursor
        cursor += timedelta(hours=1)


def download_one(relpath: str, destination: Path, api_key: str | None) -> str:
    query = {"file": relpath}
    if api_key:
        query["api_key"] = api_key
    request = urllib.request.Request(BASE_URL + "?" + urllib.parse.urlencode(query),
                                     headers={"User-Agent": "LOB-Liquidity-Stress-Meter/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "missing"
        if exc.code == 429:
            return "rate_limited"
        raise RuntimeError(f"HTTP {exc.code} for {relpath}") from exc
    if not payload:
        return "empty"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-06-28", help="UTC start date, inclusive")
    parser.add_argument("--end", default="2025-12-31", help="UTC end date, inclusive")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--data-types", nargs="+", choices=["orderbook", "trades"],
                        default=["orderbook", "trades"])
    parser.add_argument("--exchange", default="binance_spot")
    parser.add_argument("--output", default="data_raw/cryptohftdata")
    parser.add_argument("--requests-per-minute", type=int, default=55,
                        help="Keep below the anonymous free limit of 60/minute")
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if start < date(2025, 6, 28) or end < start:
        parser.error("CryptoHFTData Binance coverage starts 2025-06-28; end must be >= start")
    if not 1 <= args.requests_per_minute <= 55:
        parser.error("requests-per-minute must be between 1 and 55")
    output = (ROOT / args.output).resolve()
    # The manifest is opened before the first hourly download. Create the root
    # folder here so a fresh checkout can start without manual preparation.
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.jsonl"
    key = os.environ.get("CRYPTOHFTDATA_API_KEY")
    delay = 60.0 / args.requests_per_minute
    counts = {"downloaded": 0, "existing": 0, "missing": 0, "empty": 0}
    last_request = 0.0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for hour in hours(start, end):
            for symbol in args.symbols:
                for data_type in args.data_types:
                    filename = f"{symbol}_{data_type}.parquet"
                    relpath = f"{args.exchange}/{hour:%Y-%m-%d}/{hour:%H}/{filename}"
                    destination = output / args.exchange / f"{hour:%Y-%m-%d}" / f"{hour:%H}" / filename
                    if destination.exists() and destination.stat().st_size:
                        counts["existing"] += 1
                        continue
                    wait = delay - (time.monotonic() - last_request)
                    if wait > 0:
                        time.sleep(wait)
                    try:
                        status = download_one(relpath, destination, key)
                        last_request = time.monotonic()
                        if status == "rate_limited":
                            time.sleep(60)
                            status = download_one(relpath, destination, key)
                            last_request = time.monotonic()
                        if status == "downloaded":
                            counts[status] += 1
                        elif status in counts:
                            counts[status] += 1
                        record = {"utc_hour": hour.isoformat(), "symbol": symbol,
                                  "data_type": data_type, "path": str(destination),
                                  "status": status,
                                  "bytes": destination.stat().st_size if status == "downloaded" else 0}
                    except Exception as exc:
                        record = {"utc_hour": hour.isoformat(), "symbol": symbol,
                                  "data_type": data_type, "path": str(destination),
                                  "status": "error", "error": str(exc)}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
