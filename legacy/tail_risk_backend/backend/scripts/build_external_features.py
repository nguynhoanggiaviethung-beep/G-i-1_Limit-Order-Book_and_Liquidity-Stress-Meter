"""Build separate point-in-time daily features from external event data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.core.versioning import sha256_file  # noqa: E402
from app.features.external import attach_point_in_time_external  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    label = pd.Timestamp(research["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
    event_path = settings.data_dir / "external" / "events.parquet"
    feature_root = settings.data_dir / "features" / "snapshots" / label
    output_root = settings.data_dir / "features" / "external" / label
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    if not event_path.exists():
        raise FileNotFoundError(f"Fetch external source events first: {event_path}")
    events = pd.read_parquet(event_path)
    coverage = {}
    for symbol in symbols:
        daily_path = feature_root / f"symbol={symbol}" / "daily_rv.parquet"
        if not daily_path.exists():
            raise FileNotFoundError(f"Build frozen daily features first: {daily_path}")
        daily = pd.read_parquet(daily_path)
        external_daily, report = attach_point_in_time_external(
            daily,
            events,
            symbol,
            min_coverage=args.min_coverage,
            coverage_start=research["windows"]["oos_start"],
        )
        destination = output_root / f"symbol={symbol}" / "external_daily.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        external_daily.to_parquet(temporary, index=False)
        temporary.replace(destination)
        coverage[symbol] = report
        included = [name for name, item in report.items() if item["included"]]
        print(f"{symbol}: included {included or 'no features'}; wrote {destination}", flush=True)

    report_path = output_root / "coverage.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps({
            "as_of_utc": research["as_of_utc"],
            "coverage_start_utc": research["windows"]["oos_start"],
            "min_coverage": args.min_coverage,
            "external_events_sha256": sha256_file(event_path),
            "symbols": coverage,
        }, indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    print(f"Wrote point-in-time coverage report: {report_path}")


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("build_external_features.py")
