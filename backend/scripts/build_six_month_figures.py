"""Build the six-month study figures from saved outputs only.

Inputs are the walk-forward fold metrics, the once-scored test summary and per-second test
predictions in ``outputs/tables/final_test_6m/``, and the test bootstrap tables in its
``regime_bootstrap/`` folder. No model is refitted and the test is not rescored; weekly test
metrics and confusion matrices come from the saved predictions and summary.

The two descriptive book figures (LOB heatmap, OBI vs forward return) read feature partitions of
the development validation weeks only; any partition dated on or after the test start is refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_settings
from scripts.build_research_figures import (  # also applies the shared rcParams
    BREACH, DATA, INK, MUTE, RULE, SURFACE, _confusion_figure, _lob_heatmap, _obi_forward_return_figure,
)

ROOT = Path(__file__).resolve().parents[2]
HEATMAP_DAY = "2026-08-17"  # Monday of the last validation week
HEATMAP_NOT_BEFORE = pd.Timestamp("2026-08-17T14:00:00Z")  # European/US overlap, an active book
BOOK_COLUMNS = ["timestamp_utc", "book_valid", "mid_price", "tick_size", "obi_qty_10"] + [
    f"{side}_{field}_{level}" for side in ("bid", "ask") for level in range(1, 11) for field in ("price", "qty")
]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PRIMARY_HORIZON = 5
LABELS = ("down", "flat", "up")
REGIME_NAMES = {
    "all": "Tất cả", "vol_low": "Biến động thấp", "vol_mid": "Biến động TB", "vol_high": "Biến động cao",
    "session_asia": "Phiên Á", "session_europe": "Phiên Âu", "session_americas": "Phiên Mỹ",
}


def _summary(metrics_dir: Path, symbol: str) -> dict:
    path = metrics_dir / symbol / "classification_metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("final_test_scored_once") or not (metrics_dir / symbol / "final_test_opened.json").is_file():
        raise RuntimeError(f"{path} is not a once-scored final-test summary")
    return value


def _weekly_test(metrics_dir: Path, symbol: str, horizon: dict, test_start: pd.Timestamp) -> pd.DataFrame:
    """Per-week test MCC (direction) and PR-AUC (stress) for the selected model, uncalibrated as in walk-forward."""
    files = horizon["test_prediction_files"]
    rows = []
    direction = pd.read_parquet(metrics_dir / symbol / files["direction"])
    direction["week"] = (pd.to_datetime(direction["timestamp_utc"], utc=True) - test_start) // pd.Timedelta(days=7)
    name = horizon["direction_model"]
    columns = [f"{name}_p_{label}" for label in LABELS]
    for week, group in direction.groupby("week"):
        predicted = np.asarray(LABELS)[group[columns].to_numpy().argmax(axis=1)]
        rows.append({"task": "direction", "week": int(week), "model": "selected", "value": matthews_corrcoef(group["y_true"], predicted)})
        rows.append({"task": "direction", "week": int(week), "model": "persistence", "value": matthews_corrcoef(group["y_true"], group["direction_persistence"])})
    stress = pd.read_parquet(metrics_dir / symbol / files["stress"])
    stress = stress.loc[stress["stress_persistence"] >= 0]
    stress["week"] = (pd.to_datetime(stress["timestamp_utc"], utc=True) - test_start) // pd.Timedelta(days=7)
    name = horizon["stress_model"]
    for week, group in stress.groupby("week"):
        rows.append({"task": "stress", "week": int(week), "model": "selected", "value": average_precision_score(group["y_true"], group[f"{name}_p_stress"])})
        rows.append({"task": "stress", "week": int(week), "model": "persistence", "value": average_precision_score(group["y_true"], group["stress_persistence"])})
    return pd.DataFrame(rows)


def _weekly_figure(output: Path, metrics_dir: Path, summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True)
    for row, symbol in enumerate(SYMBOLS):
        horizon = summaries[symbol]["horizons"][str(PRIMARY_HORIZON)]
        test_start = pd.Timestamp(summaries[symbol]["reproducibility"]["final_test_start_utc"])
        folds = pd.read_csv(metrics_dir / symbol / "fold_metrics.csv")
        folds = folds.loc[folds["horizon_seconds"] == PRIMARY_HORIZON]
        weeks = sorted(folds["week_utc"].unique())
        test = _weekly_test(metrics_dir, symbol, horizon, test_start)
        n_test = int(test["week"].max()) + 1
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            metric = "mcc" if task == "direction" else "pr_auc"
            selected = horizon[f"{task}_model"]
            persistence = f"{task}_persistence"
            task_folds = folds.loc[folds["task"] == task].set_index("model")
            for model, label, color, key in ((selected, selected, DATA, "selected"), (persistence, "persistence", MUTE, "persistence")):
                development = task_folds.loc[[model]].set_index("week_utc").loc[weeks, metric].to_numpy(dtype=float)
                tested = test.loc[(test["task"] == task) & (test["model"] == key)].sort_values("week")["value"].to_numpy(dtype=float)
                ax.plot(range(len(weeks)), development, marker="o", markersize=3, color=color, label=label)
                ax.plot(range(len(weeks), len(weeks) + n_test), tested, marker="s", markersize=4, color=color)
            ax.axvspan(len(weeks) - 0.5, len(weeks) + n_test - 0.5, color=BREACH, alpha=0.08, linewidth=0)
            ax.text(len(weeks) + (n_test - 1) / 2, 0.97, "TEST (chấm 1 lần)", ha="center", va="top", fontsize=8, color=BREACH, transform=ax.get_xaxis_transform())
            ticks = [pd.Timestamp(week).strftime("%d/%m") for week in weeks] + [(test_start + pd.Timedelta(days=7 * k)).strftime("%d/%m") for k in range(n_test)]
            ax.set_xticks(range(len(ticks)), ticks, rotation=60, fontsize=7)
            ax.set_ylabel("MCC hướng" if task == "direction" else "PR-AUC stress")
            ax.set_ylim(0, 1.0 if task == "stress" else 0.5)
            ax.grid(alpha=0.2)
            ax.set_title(f"{symbol} · {'Hướng' if task == 'direction' else 'Stress'} · h={PRIMARY_HORIZON}s")
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Hiệu năng theo tuần: 12 tuần validation walk-forward và 5 tuần test", fontsize=13, weight="bold")
    fig.text(0.5, 0.005, "Tuần test tính lại từ dự báo đã lưu của mô hình được chọn, chưa hiệu chỉnh như trong walk-forward. Stress chỉ tính các giây persistence xác định được.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / "weekly_performance_6m.png", dpi=180)
    plt.close(fig)


def _calibration_figure(output: Path, summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    colors = {"down": DATA, "flat": MUTE, "up": BREACH}
    for row, symbol in enumerate(SYMBOLS):
        horizon = summaries[symbol]["horizons"][str(PRIMARY_HORIZON)]
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            result = horizon[f"{task}_test"]
            ax.plot([0, 1], [0, 1], linestyle="--", color=MUTE, linewidth=1, label="Hiệu chỉnh hoàn hảo")
            if task == "direction":
                for label, bins in result["calibration_by_class"].items():
                    ax.plot([b["mean_probability"] for b in bins], [b["observed_rate"] for b in bins], marker="o", markersize=3, color=colors.get(label), label=label)
                ece = max(result["ece_by_class"].values())
            else:
                bins = result["calibration_bins"]
                ax.plot([b["mean_probability"] for b in bins], [b["observed_rate"] for b in bins], marker="o", markersize=4, color=BREACH, label="stress")
                ece = result["ece"]
            ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Tần suất thực tế")
            if row == 1:
                ax.set_xlabel("Xác suất dự báo trung bình (đã hiệu chỉnh isotonic)")
            ax.grid(alpha=0.2)
            ax.set_title(f"{symbol} · {task} · {horizon[f'{task}_model']} · Brier={result['brier']:.3f} · ECE≤{ece:.3f}", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Calibration trên test 24/08–27/09 · h={PRIMARY_HORIZON}s", fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.1, right=0.98, hspace=0.38, wspace=0.28)
    fig.savefig(output / "calibration_test_6m.png", dpi=180)
    plt.close(fig)


def _contrast_figure(output: Path, bootstrap_dir: Path) -> None:
    rows = [
        ("Headline: selected model vs persistence", "direction", "mcc", "Mô hình chọn − persistence · MCC hướng"),
        ("RQ1: OBI vs persistence", "direction", "mcc", "RQ1 · OBI − persistence · MCC"),
        ("RQ2: adding OFI to OBI", "direction", "mcc", "RQ2 · (OBI+OFI) − OBI · MCC"),
        ("Full feature set vs OBI+OFI", "direction", "mcc", "Đủ feature − (OBI+OFI) · MCC"),
        ("Headline: selected model vs persistence", "stress", "pr_auc", "Stress · mô hình chọn − persistence · PR-AUC"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    offsets = {1: -0.22, 5: 0.0, 10: 0.22}
    shades = {1: MUTE, 5: DATA, 10: INK}
    for ax, symbol in zip(axes, SYMBOLS):
        contrasts = pd.read_csv(bootstrap_dir / f"{symbol}_contrasts.csv")
        contrasts = contrasts.loc[contrasts["regime"] == "all"]
        for index, (question, task, metric, _) in enumerate(rows):
            for horizon, offset in offsets.items():
                item = contrasts.loc[(contrasts["question"] == question) & (contrasts["task"] == task) & (contrasts["metric"] == metric) & (contrasts["horizon_seconds"] == horizon)].iloc[0]
                y = index + offset
                ax.errorbar(item["difference"], y, xerr=[[item["difference"] - item["ci_low"]], [item["ci_high"] - item["difference"]]],
                            fmt="o", markersize=4, color=shades[horizon], capsize=2, label=f"h={horizon}s" if index == 0 else None)
        ax.axvline(0, color=BREACH, linewidth=1)
        ax.set_yticks(range(len(rows)), [row[3] for row in rows], fontsize=8)
        ax.grid(alpha=0.2, axis="x")
        ax.set_xlabel("Chênh lệch (khoảng tin cậy bootstrap 95%)")
        ax.set_title(symbol)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].invert_yaxis()  # shared y: inverting once flips both panels
    fig.suptitle("Các so sánh cặp trên test · bootstrap khối 1 giờ trong từng tuần, 2.000 lần", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "paired_contrasts_test_6m.png", dpi=180)
    plt.close(fig)


def _regime_figure(output: Path, bootstrap_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    regimes = list(REGIME_NAMES)
    for ax, symbol in zip(axes, SYMBOLS):
        models = pd.read_csv(bootstrap_dir / f"{symbol}_model_metrics.csv")
        models = models.loc[(models["horizon_seconds"] == PRIMARY_HORIZON) & (models["task"] == "stress") & (models["metric"] == "pr_auc")].set_index(["model", "regime"])
        x = np.arange(len(regimes))
        for model, label, color, shift in (("selected_calibrated", "Mô hình chọn (XGBoost)", BREACH, -0.2), ("stress_persistence", "Persistence", MUTE, 0.2)):
            values = models.loc[model].loc[regimes]
            ax.bar(x + shift, values["estimate"], width=0.38, color=color, label=label,
                   yerr=[values["estimate"] - values["ci_low"], values["ci_high"] - values["estimate"]], capsize=2, error_kw={"elinewidth": 0.8})
        ax.set_xticks(x, [REGIME_NAMES[r] for r in regimes], rotation=30, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("PR-AUC stress")
        ax.grid(alpha=0.2, axis="y")
        ax.set_title(f"{symbol} · h={PRIMARY_HORIZON}s")
        ax.legend(fontsize=8)
    fig.suptitle("PR-AUC stress theo regime trên test · tercile biến động lấy ngưỡng từ train development", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "stress_by_regime_test_6m.png", dpi=180)
    plt.close(fig)


def _development_partitions(settings, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """Feature partitions with ``start <= date < end``; refuses any date at or after the untouched test start."""
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    if end > test_start:
        raise RuntimeError("descriptive figures may only read development data")
    paths = []
    for path in sorted((settings.data_dir / "features" / f"symbol={symbol}").glob("date=*/features_1s.parquet")):
        day = pd.Timestamp(path.parent.name.removeprefix("date="), tz="UTC")
        if day >= test_start:
            continue
        if start <= day < end:
            paths.append(path)
    return paths


def _read_book(paths: list[Path], columns: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.loc[(frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)]
    frame["book_valid"] = frame["book_valid"].fillna(False).astype(bool)
    for column in columns:
        if column not in {"timestamp_utc", "book_valid"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
    return frame.sort_values("timestamp_utc", kind="mergesort").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _book_figures(output: Path, settings_path: Path) -> dict[str, dict]:
    settings = load_settings(settings_path)
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    first_week = pd.Timestamp(settings.lob_first_validation_week_utc)
    day = pd.Timestamp(HEATMAP_DAY, tz="UTC")
    heat = {s: _read_book(_development_partitions(settings, s, day, day + pd.Timedelta(days=1)), BOOK_COLUMNS, day, day + pd.Timedelta(days=1)) for s in SYMBOLS}
    _lob_heatmap(output, heat, not_before=HEATMAP_NOT_BEFORE, filename="lob_heatmap_development_6m.png",
                 xlabel=f"Thời gian UTC · {HEATMAP_DAY} (tuần validation cuối của development)",
                 title="LOB heatmap · bid/ask top-10 · dữ liệu development")
    columns = ["timestamp_utc", "book_valid", "mid_price", "obi_qty_10"]
    obi = {s: _read_book(_development_partitions(settings, s, first_week, test_start), columns, first_week, test_start) for s in SYMBOLS}
    return _obi_forward_return_figure(output, obi, filename="obi_vs_forward_return_development_6m.png",
                                      title=f"OBI L10 và lợi suất mid-price {PRIMARY_HORIZON}s · 12 tuần validation development")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs" / "tables" / "final_test_6m")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "research_figures_6m")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--frontend-dir", type=Path, default=ROOT / "frontend" / "public" / "research" / "6m",
                        help="Copy the PNGs here for the dashboard research page")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {symbol: _summary(args.metrics_dir, symbol) for symbol in SYMBOLS}
    bootstrap_dir = args.metrics_dir / "regime_bootstrap"
    _weekly_figure(args.output_dir, args.metrics_dir, summaries)
    _calibration_figure(args.output_dir, summaries)
    _contrast_figure(args.output_dir, bootstrap_dir)
    _regime_figure(args.output_dir, bootstrap_dir)
    test_window = summaries[SYMBOLS[0]]["reproducibility"]
    _confusion_figure(args.output_dir, summaries, filename="confusion_matrices_test_6m.png",
                      title=f"Ma trận nhầm lẫn trên test {test_window['final_test_start_utc'][:10]} – {test_window['final_test_end_utc'][:10]} · h={PRIMARY_HORIZON}s · chấm một lần")
    obi_summary = _book_figures(args.output_dir, args.settings)
    (args.output_dir / "obi_vs_forward_return_development_6m.json").write_text(json.dumps(obi_summary, indent=2), encoding="utf-8")
    args.frontend_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.output_dir.glob("*.png")):
        shutil.copy2(path, args.frontend_dir / path.name)
        print(f"wrote {path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
