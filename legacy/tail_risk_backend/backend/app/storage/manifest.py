"""Partition path and append-only provenance manifest helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_manifest(data_dir: Path, symbol: str, entry: dict) -> Path:
    path = Path(data_dir) / "state" / "lob" / symbol.lower() / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest_written_at_utc": datetime.now(timezone.utc).isoformat(), **entry}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path
