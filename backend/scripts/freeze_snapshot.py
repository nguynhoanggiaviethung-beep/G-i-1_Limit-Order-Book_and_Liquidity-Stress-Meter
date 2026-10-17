"""Freeze a reproducible research dataset at the configured as-of timestamp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.data.snapshot import freeze_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="UTC timestamp; defaults to config/research.yaml")
    parser.add_argument("--force", action="store_true", help="Replace this snapshot directory")
    args = parser.parse_args()
    path = freeze_snapshot(args.as_of, force=args.force)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"snapshot": str(path), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
