"""Verify the local CryptoHFTData raw files against the committed owner snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "manifests" / "cryptohftdata_qa.json"
DATA_ROOT = ROOT / "data_raw" / "cryptohftdata" / "binance_spot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="committed QA manifest containing expected source hashes")
    parser.add_argument("--allow-extra", action="store_true",
                        help="allow other Binance Spot files beyond the pinned baseline")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"ERROR: expected manifest not found: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = manifest.get("files", [])
    if not expected_files:
        print("ERROR: manifest has no expected source files")
        return 2

    failures: list[str] = []
    expected_paths: set[str] = set()
    verified = 0
    for entry in expected_files:
        rel = Path(entry["source_file"])
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT):
            failures.append(f"unsafe manifest path: {entry['source_file']}")
            continue
        expected_paths.add(rel.as_posix())
        if not path.is_file():
            failures.append(f"MISSING {rel.as_posix()}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["source_bytes"]:
            failures.append(f"SIZE {rel.as_posix()}: expected {entry['source_bytes']}, got {actual_bytes}")
            continue
        actual_hash = sha256(path)
        if actual_hash.lower() != entry["sha256"].lower():
            failures.append(f"SHA256 {rel.as_posix()}: expected {entry['sha256']}, got {actual_hash}")
            continue
        verified += 1

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in DATA_ROOT.rglob("*.parquet")
    } if DATA_ROOT.exists() else set()
    extras = sorted(actual_paths - expected_paths)
    if extras and not args.allow_extra:
        failures.append(f"EXTRA: {len(extras)} file(s) outside the pinned baseline; pass --allow-extra to verify the baseline subset")
        failures.extend(f"  {path}" for path in extras[:10])

    if failures:
        print(f"FAILED: {verified}/{len(expected_files)} expected files matched")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"OK: {verified}/{len(expected_files)} files match the committed SHA-256 and byte-size manifest")
    print(f"Scope: {manifest.get('observed_dates')} | {manifest.get('observed_symbols')}")
    if extras:
        print(f"NOTE: {len(extras)} additional file(s) exist outside this pinned snapshot; they were allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
