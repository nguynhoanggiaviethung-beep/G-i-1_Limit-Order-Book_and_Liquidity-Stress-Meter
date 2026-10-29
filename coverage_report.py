"""Tổng hợp độ phủ (giây hợp lệ / giây kỳ vọng) của các ngày đã replay, từ các file qa.json.

Chạy từ thư mục gốc repo:
    python coverage_report.py
    python coverage_report.py --root outputs/manifests/archive_6m
    python coverage_report.py --min-coverage 0.95     # chỉ in ngày dưới ngưỡng

Ghi thêm data_interim/cryptohftdata_6m/coverage_report.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("outputs/manifests/archive_6m"))
    parser.add_argument("--min-coverage", type=float, default=1.01, help="chỉ in ngày có độ phủ nhỏ hơn mức này")
    parser.add_argument("--start", help="chỉ lấy ngày >= YYYY-MM-DD")
    parser.add_argument("--end", help="chỉ lấy ngày <= YYYY-MM-DD")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root

    rows = []
    for qa_path in sorted(root.rglob("qa.json")):
        try:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"! không đọc được {qa_path}: {exc}")
            continue
        day = str(qa.get("date_utc", "?"))
        if (args.start and day < args.start) or (args.end and day > args.end):
            continue
        expected = int(qa.get("expected_seconds", 0))
        valid = int(qa.get("valid_seconds", 0))
        rows.append({
            "date": day,
            "symbol": qa.get("symbol", "?"),
            "valid": valid,
            "expected": expected,
            "coverage": (valid / expected) if expected else 0.0,
            "gaps": int(qa.get("sequence_gap_count", 0)),
            "bootstraps": int(qa.get("snapshot_bootstraps", 0)),
            "recoveries": int(qa.get("snapshot_recoveries", 0)),
        })
    if not rows:
        print(f"Không tìm thấy qa.json dưới {root}")
        return 1
    # ngày xuất hiện ở nhiều đợt (đợt cũ lỗi, đợt mới tốt): lấy bản có độ phủ cao nhất
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["date"], row["symbol"])
        if key not in best or row["coverage"] > best[key]["coverage"]:
            best[key] = row
    ordered = sorted(best.values(), key=lambda r: (r["date"], r["symbol"]))

    out_path = ROOT / "data_interim" / "cryptohftdata_6m" / "coverage_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("date,symbol,valid_seconds,expected_seconds,coverage,sequence_gaps,snapshot_bootstraps,snapshot_recoveries\n")
        for r in ordered:
            handle.write(f"{r['date']},{r['symbol']},{r['valid']},{r['expected']},{r['coverage']:.4f},{r['gaps']},{r['bootstraps']},{r['recoveries']}\n")

    print(f"{'date':<12}{'symbol':<9}{'coverage':>9}{'valid':>9}{'expected':>10}{'gaps':>6}")
    shown = 0
    for r in ordered:
        if r["coverage"] < args.min_coverage:
            print(f"{r['date']:<12}{r['symbol']:<9}{r['coverage']:>9.1%}{r['valid']:>9,}{r['expected']:>10,}{r['gaps']:>6}")
            shown += 1
    total_valid = sum(r["valid"] for r in ordered)
    total_expected = sum(r["expected"] for r in ordered)
    print(f"\n{len(ordered)} ngày-symbol, {shown} dòng in ra. Tổng độ phủ: {total_valid / total_expected:.1%}" if total_expected else "")
    print(f"Đã ghi {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())