"""Đo sai số của book dựng KHÔNG cần snapshot (warm-start) so với book chính xác từ snapshot.

Cách làm: trên các ngày đã có grid chính xác (giai đoạn có snapshot), replay lại bằng warm-start với
--ignore-snapshots và cứ mỗi R giây "quên" book để khởi động lại từ đầu, rồi so từng giây với grid chính xác
theo tuổi của book (số giây kể từ lúc bắt đầu dựng). Nhờ đó biết cần khởi động bao lâu thì sai số đủ nhỏ.

Chạy từ thư mục gốc repo (sau khi đã replay xong grid chính xác):
    python validate_warm_start.py --dates 2026-08-26,2026-09-10,2026-09-30,2026-10-14
    python validate_warm_start.py --dates 2026-09-10 --symbol ETHUSDT --skip-replay

Kết quả: bảng in ra + data_interim/validation_warm/validation_<SYMBOL>.csv
Nhãn "stress tức thời" ở đây = spread_ticks > P97.5 | price_impact_bps > P97.5 | depth_notional_10 < P2.5
(ngưỡng lấy từ chính các ngày chính xác đã chọn); KHÔNG gồm rv_60s vì rv phụ thuộc chuỗi liên tục.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.features.classification_features import build_classification_features  # noqa: E402

BIN_EDGES = [0, 60, 300, 900, 1800, 3600, np.inf]
BIN_LABELS = ["0-1m", "1-5m", "5-15m", "15-30m", "30-60m", "60m+"]
UPPER_Q, DEPTH_Q = 0.975, 0.025


def day_path(root: Path, symbol: str, day: str) -> Path:
    return root / f"symbol={symbol}" / f"date={day}" / "book_1s.parquet"


def load_features(path: Path, notional: float, tick: float) -> pd.DataFrame:
    grid = pd.read_parquet(path)
    keep_extra = [c for c in ("book_age_seconds", "book_source") if c in grid]
    features = build_classification_features(grid, notional, tick)
    for column in keep_extra:
        if column not in features:
            features[column] = grid.set_index(pd.to_datetime(grid["timestamp_utc"], utc=True))[column].reindex(features["timestamp_utc"]).to_numpy()
    return features


def top_levels_equal(a: pd.DataFrame, b: pd.DataFrame, levels: int = 10) -> np.ndarray:
    equal = np.ones(len(a), dtype=bool)
    for side in ("bid", "ask"):
        for level in range(1, levels + 1):
            for kind in ("price", "qty"):
                name = f"{side}_{kind}_{level}"
                x = pd.to_numeric(a[name], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(b[name], errors="coerce").to_numpy(dtype=float)
                equal &= np.isclose(x, y, rtol=0, atol=1e-12, equal_nan=True)
    return equal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", required=True, help="ngày có grid chính xác, cách nhau bằng dấu phẩy")
    parser.add_argument("--symbol", default="BTCUSDT", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--exact-root", type=Path, default=Path("data_processed/cryptohftdata_6m/grid"))
    parser.add_argument("--warm-root", type=Path, default=Path("data_processed/validation_warm/grid"))
    parser.add_argument("--restage-interval-seconds", type=int, default=7200, help="quên book mỗi N giây để lấy nhiều mẫu khởi động")
    parser.add_argument("--skip-replay", action="store_true", help="dùng lại grid warm-start đã tạo")
    parser.add_argument("--min-agreement", type=float, default=0.99, help="ngưỡng khớp nhãn stress (nhóm tự đặt TRƯỚC khi xem kết quả)")
    parser.add_argument("--max-depth-error", type=float, default=0.05, help="sai số tương đối tối đa của depth_notional_10")
    args = parser.parse_args()

    exact_root = args.exact_root if args.exact_root.is_absolute() else ROOT / args.exact_root
    warm_root = args.warm_root if args.warm_root.is_absolute() else ROOT / args.warm_root
    days = [d.strip() for d in args.dates.split(",") if d.strip()]
    settings = load_settings(ROOT / "config" / "settings.yaml")
    tick = settings.tick_size_for(args.symbol)
    notional = settings.impact_notional_for(args.symbol)

    for day in days:
        if not day_path(exact_root, args.symbol, day).is_file():
            raise SystemExit(f"Thiếu grid chính xác: {day_path(exact_root, args.symbol, day)}")
        if args.skip_replay:
            continue
        print(f"[warm-start] {args.symbol} {day} ...", flush=True)
        code = subprocess.run([
            sys.executable, str(ROOT / "backend" / "scripts" / "replay_cryptohftdata_archive.py"),
            "--start-date", day, "--end-date", day, "--symbol", args.symbol,
            "--output-root", str(warm_root), "--manifest-root", str(ROOT / "outputs" / "manifests" / "validation_warm"),
            "--warm-start-seconds", "0", "--ignore-snapshots",
            "--restage-interval-seconds", str(args.restage_interval_seconds),
        ], cwd=ROOT, stdout=subprocess.DEVNULL).returncode
        if code != 0:
            raise SystemExit(f"Replay warm-start lỗi (exit {code}) cho {day}")

    merged = []
    for day in days:
        exact = load_features(day_path(exact_root, args.symbol, day), notional, tick)
        warm = load_features(day_path(warm_root, args.symbol, day), notional, tick)
        both = exact.merge(warm, on="timestamp_utc", suffixes=("", "_w"), how="inner")
        both["top10_equal"] = top_levels_equal(
            both[[c for c in both if not c.endswith("_w")]],
            both[[c for c in both if c.endswith("_w")]].rename(columns=lambda c: c[:-2]),
        )
        merged.append(both)
    data = pd.concat(merged, ignore_index=True)
    # Grid chính xác không có cột tuổi book, nên pandas không thêm hậu tố _w cho cột của grid warm-start.
    age_column = "book_age_seconds_w" if "book_age_seconds_w" in data else "book_age_seconds"
    if age_column not in data:
        raise SystemExit("Grid warm-start thiếu cột book_age_seconds: hãy replay với --warm-start-seconds.")
    data = data[data[age_column].notna()].copy()
    if data.empty:
        raise SystemExit("Không có giây nào khớp giữa hai grid.")

    spread_t = np.nanquantile(data["spread_ticks"], UPPER_Q)
    impact_t = np.nanquantile(data["price_impact_bps"], UPPER_Q)
    depth_t = np.nanquantile(data["depth_notional_10"], DEPTH_Q)
    data["flag_exact"] = (data["spread_ticks"] > spread_t) | (data["price_impact_bps"] > impact_t) | (data["depth_notional_10"] < depth_t)
    data["flag_warm"] = (data["spread_ticks_w"] > spread_t) | (data["price_impact_bps_w"] > impact_t) | (data["depth_notional_10_w"] < depth_t)
    data["spread_err"] = (data["spread_ticks_w"] - data["spread_ticks"]).abs()
    data["mid_err_bps"] = ((data["mid_price_w"] - data["mid_price"]) / data["mid_price"]).abs() * 1e4
    data["depth_rel_err"] = data["depth_notional_10_w"] / data["depth_notional_10"] - 1.0
    data["impact_err_bps"] = (data["price_impact_bps_w"] - data["price_impact_bps"]).abs()
    data["age_bin"] = pd.cut(data[age_column], BIN_EDGES, labels=BIN_LABELS, right=False)

    rows = []
    for label, part in data.groupby("age_bin", observed=True):
        negatives = ~part["flag_exact"]
        positives = part["flag_exact"]
        rows.append({
            "age": label,
            "n": len(part),
            "top10_equal_%": 100 * part["top10_equal"].mean(),
            "spread_err_p95_ticks": part["spread_err"].quantile(0.95),
            "mid_err_p95_bps": part["mid_err_bps"].quantile(0.95),
            "depth_err_median_%": 100 * part["depth_rel_err"].median(),
            "depth_err_p5_%": 100 * part["depth_rel_err"].quantile(0.05),
            "impact_err_p95_bps": part["impact_err_bps"].quantile(0.95),
            "label_agree_%": 100 * (part["flag_warm"] == part["flag_exact"]).mean(),
            "false_pos_%": 100 * part.loc[negatives, "flag_warm"].mean() if negatives.any() else np.nan,
            "false_neg_%": 100 * (~part.loc[positives, "flag_warm"]).mean() if positives.any() else np.nan,
        })
    table = pd.DataFrame(rows)
    out = ROOT / "data_interim" / "validation_warm"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"validation_{args.symbol}.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n{args.symbol}: {len(data):,} giây so sánh trên {len(days)} ngày; ngưỡng stress tức thời "
          f"spread>{spread_t:.2f} ticks | impact>{impact_t:.2f} bps | depth_10<{depth_t:,.0f}\n")
    print(table.round(3).to_string(index=False))
    ok = table[(table["label_agree_%"] >= 100 * args.min_agreement) & (table["depth_err_median_%"].abs() <= 100 * args.max_depth_error)]
    print()
    if ok.empty:
        print(f"Không nhóm tuổi nào đạt (khớp nhãn >= {args.min_agreement:.0%} và |sai số depth trung vị| <= {args.max_depth_error:.0%}).")
        print("=> Warm-start không đủ chính xác với ngưỡng này; KHÔNG dùng dữ liệu xấp xỉ cho train.")
    else:
        first = ok.iloc[0]["age"]
        print(f"Nhóm tuổi đầu tiên đạt ngưỡng: {first}. Đây là gợi ý cho --warm-start-seconds "
              f"(cận dưới của nhóm); ngưỡng do nhóm tự đặt, không phải kết luận của script.")
    print(f"\nĐã ghi {out / f'validation_{args.symbol}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())