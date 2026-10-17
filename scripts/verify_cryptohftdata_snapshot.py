"""Đối chiếu các file CryptoHFTData thô với snapshot checksum đã commit."""
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
                        help="manifest QA đã commit, chứa hash nguồn kỳ vọng")
    parser.add_argument("--allow-extra", action="store_true",
                        help="cho phép có file Binance Spot ngoài baseline đã ghim")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"LỖI: không tìm thấy manifest tham chiếu: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = manifest.get("files", [])
    if not expected_files:
        print("LỖI: manifest không có danh sách file nguồn cần kiểm tra")
        return 2

    failures: list[str] = []
    expected_paths: set[str] = set()
    verified = 0
    for entry in expected_files:
        rel = Path(entry["source_file"])
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT):
            failures.append(f"ĐƯỜNG DẪN KHÔNG AN TOÀN: {entry['source_file']}")
            continue
        expected_paths.add(rel.as_posix())
        if not path.is_file():
            failures.append(f"THIẾU {rel.as_posix()}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["source_bytes"]:
            failures.append(
                f"SAI KÍCH THƯỚC {rel.as_posix()}: cần {entry['source_bytes']}, thực tế {actual_bytes}"
            )
            continue
        actual_hash = sha256(path)
        if actual_hash.lower() != entry["sha256"].lower():
            failures.append(
                f"SAI SHA-256 {rel.as_posix()}: cần {entry['sha256']}, thực tế {actual_hash}"
            )
            continue
        verified += 1

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in DATA_ROOT.rglob("*.parquet")
    } if DATA_ROOT.exists() else set()
    extras = sorted(actual_paths - expected_paths)
    if extras and not args.allow_extra:
        failures.append(
            f"THỪA: có {len(extras)} file ngoài baseline đã ghim; "
            "dùng --allow-extra để chỉ xác minh phần baseline"
        )
        failures.extend(f"  {path}" for path in extras[:10])

    if failures:
        print(f"KHÔNG ĐẠT: khớp {verified}/{len(expected_files)} file kỳ vọng")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"ĐẠT: {verified}/{len(expected_files)} file khớp kích thước và SHA-256 trong manifest đã commit")
    print(f"Phạm vi: {manifest.get('observed_dates')} | {manifest.get('observed_symbols')}")
    if extras:
        print(f"LƯU Ý: có thêm {len(extras)} file ngoài snapshot đã ghim; đã cho phép theo tùy chọn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
