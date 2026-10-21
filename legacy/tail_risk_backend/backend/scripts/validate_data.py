"""Validate the frozen research snapshot (or live candles with --live)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.data.data_validator import validate_klines  # noqa: E402
from app.data.snapshot import load_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="UTC timestamp; defaults to frozen research config")
    parser.add_argument("--live", action="store_true", help="Validate rolling data/processed instead")
    args = parser.parse_args()
    settings = load_settings()
    research = load_research_config()
    cutoff = pd.Timestamp(args.as_of or research["as_of_utc"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    if args.live:
        frames = {}
        for symbol in research["symbols"]:
            path = settings.data_dir / "processed" / f"symbol={symbol}" / f"interval={settings.interval}" / "klines.parquet"
            if path.exists():
                frames[symbol] = pd.read_parquet(path)
        manifest = {"as_of_utc": datetime.now(timezone.utc).isoformat(), "data_version": "live"}
        destination = ROOT / "results" / "metrics" / "data_quality.json"
    else:
        frames, manifest = load_snapshot(cutoff)
        destination = ROOT / "results" / "metrics" / "snapshot_quality.json"

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": manifest["as_of_utc"],
        "data_version": manifest["data_version"],
        "interval": settings.interval,
        "symbols": {},
    }
    for symbol in research["symbols"]:
        frame = frames.get(symbol)
        if frame is None:
            report["symbols"][symbol] = {"status": "missing"}
            continue
        summary = validate_klines(frame, settings.interval, now=cutoff.to_pydatetime(), raise_on_error=False)
        summary["status"] = "warning" if summary["errors"] else "ok"
        summary["first_open_time_utc"] = pd.to_datetime(frame["open_time"], utc=True).min().isoformat()
        summary["last_open_time_utc"] = pd.to_datetime(frame["open_time"], utc=True).max().isoformat()
        summary["days_with_any_gaps"] = int(
            frame.assign(utc_date=pd.to_datetime(frame["open_time"], utc=True).dt.floor("D"))
            .groupby("utc_date").size().lt(int(research["sampling"]["bars_per_day"])).sum()
        )
        report["symbols"][symbol] = summary
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote snapshot validation report to {destination}")


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("validate_data.py")
