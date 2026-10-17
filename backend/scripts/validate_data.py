"""Validate saved candles and write a small machine-readable quality report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402
from app.core.config import load_settings  # noqa: E402
from app.data.data_validator import validate_klines  # noqa: E402


def main() -> None:
    settings = load_settings()
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interval": settings.interval,
        "symbols": {},
    }
    for symbol in settings.symbols:
        path = settings.data_dir / "processed" / f"symbol={symbol}" / f"interval={settings.interval}" / "klines.parquet"
        if not path.exists():
            report["symbols"][symbol] = {"status": "missing", "file": str(path)}
            continue
        frame = pd.read_parquet(path)
        summary = validate_klines(frame, settings.interval, raise_on_error=False)
        summary["status"] = "warning" if summary["errors"] else "ok"
        summary["first_open_time"] = frame["open_time"].min().isoformat() if not frame.empty else None
        summary["last_open_time"] = frame["open_time"].max().isoformat() if not frame.empty else None
        summary["file"] = str(path)
        report["symbols"][symbol] = summary
    destination = ROOT / "results" / "metrics" / "data_quality.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote data quality report to {destination}")


if __name__ == "__main__":
    main()
