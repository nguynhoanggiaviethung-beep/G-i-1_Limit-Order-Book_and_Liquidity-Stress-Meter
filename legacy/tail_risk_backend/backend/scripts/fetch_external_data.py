"""Fetch optional derivative and macro event data without altering the frozen spot snapshot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.data.external_client import EVENT_COLUMNS, ExternalDataClient  # noqa: E402


def _atomic_write(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="Binance USD-M pair; repeatable")
    parser.add_argument("--start", help="Start UTC date; defaults to settings.yaml")
    parser.add_argument("--end", help="End UTC date; defaults to the frozen research cutoff")
    parser.add_argument("--period", default="1d", choices=("1h", "4h", "12h", "1d"))
    parser.add_argument("--skip-binance", action="store_true")
    parser.add_argument("--skip-fred", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    research = load_research_config()
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    start = pd.Timestamp(args.start or settings.start_date)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(args.end or research["as_of_utc"])
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        parser.error("--end must be later than --start")

    client = ExternalDataClient(
        timeout=settings.request_timeout_seconds,
        retries=settings.request_retries,
    )
    frames: list[pd.DataFrame] = []
    source_errors: list[str] = []
    if not args.skip_binance:
        for symbol in symbols:
            for name, fetch in (
                ("funding", lambda: client.fetch_funding_rates(symbol, start, end)),
                ("open_interest", lambda: client.fetch_open_interest_history(symbol, start, end, args.period)),
                ("basis", lambda: client.fetch_basis_history(symbol, start, end, args.period)),
            ):
                try:
                    frame = fetch()
                    frames.append(frame)
                    if frame.empty:
                        print(f"{symbol} {name}: API returned no observations in the requested time range", flush=True)
                    else:
                        first = frame["observation_time_utc"].min()
                        last = frame["observation_time_utc"].max()
                        print(f"{symbol} {name}: {len(frame):,} rows from {first} through {last}", flush=True)
                except Exception as exc:
                    message = f"{symbol} {name}: {type(exc).__name__}: {exc}"
                    source_errors.append(message)
                    print(f"ERROR {message}", flush=True)

    if not args.skip_fred:
        api_key = os.getenv("FRED_API_KEY", "").strip()
        if not api_key:
            print("FRED skipped: set FRED_API_KEY to fetch VIX and broad-USD proxy data", flush=True)
        else:
            for series_id in ("VIXCLS", "DTWEXBGS"):
                try:
                    frame = client.fetch_fred_daily(series_id, api_key, start, end)
                    frames.append(frame)
                    print(f"FRED {series_id}: {len(frame):,} initial-release rows", flush=True)
                except Exception as exc:
                    message = f"FRED {series_id}: {type(exc).__name__}: {exc}"
                    source_errors.append(message)
                    print(f"ERROR {message}", flush=True)

    if not frames:
        print("No external sources were selected.", flush=True)
        return
    incoming = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
    if incoming.empty:
        print("No external observations were returned; existing data was left untouched.", flush=True)
        if source_errors:
            print("Some source requests failed; see the errors above.", flush=True)
        return

    destination = settings.data_dir / "external" / "events.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_parquet(destination) if destination.exists() else pd.DataFrame(columns=EVENT_COLUMNS)
    combined = pd.concat([previous, incoming], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        ["symbol", "feature_name", "observation_time_utc", "source"], keep="last"
    ).sort_values(["feature_name", "symbol", "observation_time_utc"], na_position="first")
    _atomic_write(combined.reset_index(drop=True), destination)
    print(f"Merged {len(incoming):,} fetched rows into {len(combined):,} records: {destination}", flush=True)
    if source_errors:
        print(f"{len(source_errors)} source request(s) failed; successful source data was still saved.", flush=True)
    if not args.skip_binance:
        print(
            "Binance open-interest and basis history endpoints expose only recent data; "
            "their returned date ranges are shown above and must not be assumed complete for the frozen OOS period.",
            flush=True,
        )


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("fetch_external_data.py")
