"""Check saved LOB feature partitions and write per-symbol quality reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.config import REPOSITORY_ROOT, load_settings
from app.quality.validation import validate_lob_samples


def main() -> None:
    settings = load_settings()
    reports = {}
    for symbol in settings.symbols:
        root = settings.data_dir / "features" / "lob" / f"symbol={symbol}"
        paths = sorted(root.glob("date=*/*.parquet"))
        if not paths:
            reports[symbol] = {"status": "no_data", "path": str(root)}
            continue
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        report = validate_lob_samples(
            frame,
            settings.lob_sampling_interval_seconds,
            settings.lob_stale_after_seconds,
        )
        report.update({"status": "ok" if not report["errors"] else "quality_issues", "source_files": len(paths)})
        reports[symbol] = report
    output = REPOSITORY_ROOT / "results" / "metrics" / "lob_data_quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
