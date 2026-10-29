"""Assemble the six-month study grid from the exact and warm-start archive replays.

For every symbol/day the partition with the most valid book seconds is chosen; ties go to
the exact (provider-snapshot) replay. Partitions are hard-linked into the unified grid, so
no data is copied when source and target share a drive. Every choice is written to
``outputs/manifests/unified_6m/grid_sources.csv`` with the study split it belongs to, so
later analysis can separate exact and approximate days.

The untouched test window must consist of exact-replay days only; the script refuses to
finish otherwise.

    python backend/scripts/build_unified_6m_grid.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings


SOURCES = (
    ("exact", "provider_snapshot_exact", REPOSITORY_ROOT / "data_processed" / "cryptohftdata_6m" / "grid"),
    ("warm", "warm_start_approximate", REPOSITORY_ROOT / "data_processed" / "cryptohftdata_6m_warm" / "grid"),
)
PARTITION_FILE = "book_1s.parquet"
SECONDS_PER_DAY = 86_400


def _partition(root: Path, symbol: str, day: str) -> Path:
    return root / f"symbol={symbol}" / f"date={day}" / PARTITION_FILE


def _valid_seconds(path: Path) -> tuple[int, int]:
    # ParquetFile reads the single file; read_table would infer hive partitions from the path.
    table = pq.ParquetFile(path).read(columns=["book_valid"])
    valid = table.column("book_valid").to_pandas().fillna(False).astype(bool)
    return int(valid.sum()), table.num_rows


def _link(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return "unchanged"
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _split(day: str, settings) -> str:
    stamp = pd.Timestamp(day, tz="UTC")
    if stamp <= pd.Timestamp(settings.lob_development_end_utc):
        return "development"
    if stamp <= pd.Timestamp(settings.lob_untouched_test_end_utc):
        return "untouched_test"
    return "pilot_seen"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 19))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 18))
    parser.add_argument("--manifest-root", type=Path, default=REPOSITORY_ROOT / "outputs" / "manifests" / "unified_6m")
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    output_root = settings.data_dir / "grid"
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else REPOSITORY_ROOT / args.manifest_root

    rows: list[dict] = []
    days = [(args.start + timedelta(days=offset)).isoformat() for offset in range((args.end - args.start).days + 1)]
    for symbol in settings.symbols:
        for day in days:
            candidates = []
            for name, reconstruction, root in SOURCES:
                path = _partition(root, symbol, day)
                if path.exists():
                    valid, total = _valid_seconds(path)
                    candidates.append((valid, name, reconstruction, path, total))
            split = _split(day, settings)
            if not candidates:
                rows.append({"symbol": symbol, "date": day, "split": split, "source": "missing"})
                continue
            # Highest valid seconds wins; the stable sort keeps exact ahead of warm on ties.
            chosen = sorted(candidates, key=lambda item: -item[0])[0]
            valid, name, reconstruction, path, total = chosen
            alternative = next((item[0] for item in candidates if item[1] != name), None)
            action = _link(path, _partition(output_root, symbol, day))
            rows.append({
                "symbol": symbol,
                "date": day,
                "split": split,
                "source": name,
                "book_reconstruction": reconstruction,
                "valid_seconds": valid,
                "rows": total,
                "coverage": round(valid / SECONDS_PER_DAY, 6),
                "alternative_valid_seconds": alternative,
                "source_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "link": action,
            })

    manifest_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "symbol", "date", "split", "source", "book_reconstruction", "valid_seconds", "rows",
        "coverage", "alternative_valid_seconds", "source_path", "link",
    ]
    with (manifest_root / "grid_sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    summary_rows = []
    for (symbol, split, source), group in frame.groupby(["symbol", "split", "source"], sort=True):
        summary_rows.append({
            "symbol": symbol,
            "split": split,
            "source": source,
            "days": int(len(group)),
            "valid_seconds": int(group.get("valid_seconds", pd.Series(dtype=float)).fillna(0).sum()),
        })
    summary = {
        "settings_file": settings_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        "grid_root": output_root.relative_to(REPOSITORY_ROOT).as_posix(),
        "date_range": [args.start.isoformat(), args.end.isoformat()],
        "selection_rule": "most valid book seconds per symbol/day; ties prefer the exact replay",
        "by_symbol_split_source": summary_rows,
    }
    (manifest_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'symbol':<9}{'split':<16}{'source':<9}{'days':>5}{'coverage':>10}")
    for row in summary_rows:
        expected = row["days"] * SECONDS_PER_DAY
        coverage = row["valid_seconds"] / expected if expected else 0.0
        print(f"{row['symbol']:<9}{row['split']:<16}{row['source']:<9}{row['days']:>5}{coverage:>10.2%}")
    print(f"\nUnified grid: {output_root}")
    print(f"Manifest:     {manifest_root / 'grid_sources.csv'}")

    bad_test = frame[(frame["split"] == "untouched_test") & (frame["source"] != "exact")]
    if not bad_test.empty:
        print("\nERROR: untouched test days without an exact replay:")
        print(bad_test[["symbol", "date", "source"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
