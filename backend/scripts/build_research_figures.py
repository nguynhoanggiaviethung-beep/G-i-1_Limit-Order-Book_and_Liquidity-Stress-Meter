"""Build the research figures from local pilot metrics and pre-test data.

The script reads saved aggregate metrics for figures that include the already
scored holdout. It does not load the final-test Parquet partitions, retrain a
model, or rescore the holdout. LOB and OBI descriptive plots use validation
week data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler


ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
PRIMARY_HORIZON = 5
PAPER = "#f4f4f0"
SURFACE = "#ffffff"
INK = "#17181c"
MUTE = "#6b6c70"
RULE = "#d4d4ce"
DATA = "#3b4fd6"
BREACH = "#ad4a2c"
VALIDATION_START = pd.Timestamp("2026-10-05T00:00:00Z")
VALIDATION_END = pd.Timestamp("2026-10-12T00:00:00Z")
TEST_START = pd.Timestamp("2026-10-12T00:00:00Z")

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "font.family": "monospace",
    "font.monospace": ["Space Mono", "DejaVu Sans Mono"],
    "axes.prop_cycle": cycler(color=[DATA, INK, BREACH, MUTE]),
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight",
})


def _metrics(symbol: str) -> dict:
    path = ROOT / "outputs" / "tables" / "historical_archive_pilot" / symbol / "classification_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Saved pilot metrics not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("final_test_scored_once"):
        raise RuntimeError(f"Refusing to use unmarked final-test summary in {path}")
    return value


def _feature_paths(symbol: str, dates: set[str] | None = None) -> list[Path]:
    root = ROOT / "data_processed" / "cryptohftdata_pilot" / "features"
    paths = sorted(root.rglob("features_1s.parquet"))
    selected = []
    for path in paths:
        if f"symbol={symbol}" not in path.parts:
            continue
        date_keys = [part.removeprefix("date=") for part in path.parts if part.startswith("date=")]
        if not date_keys:
            raise RuntimeError(f"Refusing to load a feature file without a date partition: {path}")
        date = date_keys[-1]
        if pd.Timestamp(date, tz="UTC") >= TEST_START:
            continue
        if dates is None or date in dates:
            selected.append(path)
    return selected


def _read_validation(symbol: str, *, with_book: bool = False) -> pd.DataFrame:
    paths = _feature_paths(symbol)
    if with_book:
        paths = [path for path in paths if any(f"date={date}" in path.parts for date in ("2026-10-05",))]
        columns = ["timestamp_utc", "book_valid", "mid_price", "tick_size", "obi_qty_10"]
        for side in ("bid", "ask"):
            for level in range(1, 11):
                columns.extend([f"{side}_price_{level}", f"{side}_qty_{level}"])
        frame = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
    else:
        columns = ["timestamp_utc", "book_valid", "mid_price", "obi_qty_10"]
        frame = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"])
    frame = frame.loc[(frame["timestamp_utc"] >= VALIDATION_START) & (frame["timestamp_utc"] < VALIDATION_END)]
    frame["book_valid"] = frame["book_valid"].fillna(False).astype(bool)
    for column in frame.columns:
        if column not in {"timestamp_utc", "book_valid"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp_utc", kind="mergesort").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _confusion_figure(output: Path, metrics: dict[str, dict], *, filename: str = "confusion_matrices_holdout.png",
                      title: str = "Ma trận nhầm lẫn trên holdout đã chấm một lần") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    for row, symbol in enumerate(SYMBOLS):
        h = metrics[symbol]["horizons"][str(PRIMARY_HORIZON)]
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            result = h[f"{task}_test"]
            cm = np.asarray(result["confusion_matrix"], dtype=float)
            labels = [str(x) for x in result["labels"]]
            row_totals = cm.sum(axis=1, keepdims=True)
            normalized = np.divide(cm, row_totals, out=np.zeros_like(cm), where=row_totals > 0)
            image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(labels)), labels=labels)
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xlabel("Dự đoán")
            ax.set_ylabel("Thực tế")
            task_label = "Hướng" if task == "direction" else "Stress"
            ax.set_title(f"{symbol} · {task_label} · {h[f'{task}_model']} · n={int(result['n']):,}", fontsize=9)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]:,.0f}\n{normalized[i, j]:.1%}", ha="center", va="center",
                            color=INK if normalized[i, j] < 0.55 else SURFACE, fontsize=8)
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.text(0.5, 0.94, "Mỗi ô: số lượng dự báo và tỷ lệ theo hàng thực tế", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.88, hspace=0.36, wspace=0.34)
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.018, 0.60])
    fig.colorbar(image, cax=colorbar_axis, label="Tỷ lệ trong lớp thực tế")
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def _calibration_figure(output: Path, metrics: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    for row, symbol in enumerate(SYMBOLS):
        h = metrics[symbol]["horizons"][str(PRIMARY_HORIZON)]
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            result = h[f"{task}_test"]
            ax.plot([0, 1], [0, 1], linestyle="--", color=MUTE, linewidth=1, label="Hiệu chỉnh hoàn hảo")
            if task == "direction":
                bins_by_class = result.get("calibration_by_class", {})
                colors = {"down": DATA, "flat": MUTE, "up": BREACH}
                for label, bins in bins_by_class.items():
                    if bins:
                        ax.plot([item["mean_probability"] for item in bins],
                                [item["observed_rate"] for item in bins], marker="o", markersize=3,
                                color=colors.get(label, None), label=label)
            else:
                bins = result.get("calibration_bins", [])
                if bins:
                    ax.plot([item["mean_probability"] for item in bins],
                            [item["observed_rate"] for item in bins], marker="o", markersize=4,
                            color=BREACH, label="stress")
            ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Tần suất thực tế")
            if row == 1:
                ax.set_xlabel("Xác suất dự báo trung bình")
            ax.grid(alpha=0.2)
            ax.set_title(f"{symbol} · {task} · Brier={result.get('brier', float('nan')):.3f}")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Calibration · holdout đã chấm một lần · biểu đồ từ các bin đã lưu", fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.12, right=0.98, hspace=0.54, wspace=0.34)
    fig.savefig(output / "calibration_curves_holdout.png", dpi=180)
    plt.close(fig)


def _performance_over_time_figure(output: Path, metrics: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    periods = ["Validation\n05–11 Oct", "Holdout\n12–18 Oct"]
    for row, symbol in enumerate(SYMBOLS):
        h = metrics[symbol]["horizons"][str(PRIMARY_HORIZON)]
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            selector = h[f"{task}_model"]
            fold_rows = h[f"walk_forward_{task}"]
            validation = next((x for x in fold_rows if x.get("model") == selector), None)
            holdout = h[f"{task}_test"]
            metric_key = "macro_f1" if task == "direction" else "pr_auc"
            values = [validation.get(metric_key) if validation else None, holdout.get(metric_key)]
            y = [np.nan if value is None else float(value) for value in values]
            ax.plot(range(2), y, marker="o", linewidth=2, color=DATA if task == "direction" else BREACH)
            ax.set_xticks(range(2), periods)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Macro-F1" if task == "direction" else "PR-AUC")
            ax.grid(alpha=0.2)
            ax.set_title(f"{symbol} · {task} · {selector}")
            for x, value in enumerate(y):
                if np.isfinite(value):
                    ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center")
    fig.suptitle("Hiệu năng theo giai đoạn · pilot chỉ có 1 tuần validation và 1 tuần holdout", fontsize=13, weight="bold")
    fig.text(0.5, 0.01, "Các điểm holdout lấy từ kết quả aggregate đã lưu; không chấm lại holdout. Hai giai đoạn không đủ để kết luận ổn định dài hạn.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output / "performance_over_time_pilot.png", dpi=180)
    plt.close(fig)


def _lob_heatmap(output: Path, frames: dict[str, pd.DataFrame] | None = None, *, not_before: pd.Timestamp | None = None,
                 filename: str = "lob_heatmap_validation.png", xlabel: str = "Thời gian UTC · tuần validation 2026-10-05",
                 title: str = "LOB heatmap · bid/ask top-10 · dữ liệu validation trước holdout") -> None:
    # Supplied frames may give each symbol a different window, so only the pilot shares its time axis.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=frames is None)
    plotted = []
    for ax, symbol in zip(axes, SYMBOLS):
        frame = frames[symbol] if frames is not None else _read_validation(symbol, with_book=True)
        frame = frame.loc[frame["book_valid"]].reset_index(drop=True)
        if len(frame) < 300:
            raise RuntimeError(f"Not enough valid pre-test snapshots for a five-minute {symbol} heatmap")
        times_ns = frame["timestamp_utc"].astype("int64").to_numpy()
        consecutive = np.diff(times_ns) == 1_000_000_000
        window = 300
        run_starts = np.flatnonzero(np.convolve(consecutive.astype(np.int16), np.ones(window - 1, dtype=np.int16), mode="valid") == window - 1)
        if not len(run_starts):
            raise RuntimeError(f"No uninterrupted five-minute validation window found for {symbol}")
        if not_before is not None:
            run_starts = run_starts[(frame["timestamp_utc"].iloc[run_starts] >= not_before).to_numpy()]
            if not len(run_starts):
                raise RuntimeError(f"No uninterrupted five-minute window for {symbol} after {not_before}")
        start = int(run_starts[0])
        sample = frame.iloc[start : start + window]
        tick = float(sample["tick_size"].dropna().median())
        if not np.isfinite(tick) or tick <= 0:
            raise RuntimeError(f"Invalid configured tick size in {symbol} validation sample")
        offsets = np.arange(-10.5, 10.51, 0.5)
        matrix = np.zeros((len(offsets), len(sample)), dtype=float)
        mid = sample["mid_price"].to_numpy(dtype=float)
        for side, levels in (("bid", range(1, 11)), ("ask", range(1, 11))):
            for level in levels:
                price = sample[f"{side}_price_{level}"].to_numpy(dtype=float)
                quantity = sample[f"{side}_qty_{level}"].to_numpy(dtype=float)
                relative_ticks = (price - mid) / tick
                row_index = np.abs(offsets[:, None] - relative_ticks[None, :]).argmin(axis=0)
                # Levels beyond the plotted ±10.5 ticks are dropped rather than piled onto the edge rows.
                valid = np.isfinite(relative_ticks) & np.isfinite(quantity) & (quantity > 0) & (np.abs(relative_ticks) <= offsets[-1] + 0.25)
                matrix[row_index[valid], np.flatnonzero(valid)] += quantity[valid]
        edges = np.concatenate(([offsets[0] - 0.25], (offsets[:-1] + offsets[1:]) / 2, [offsets[-1] + 0.25]))
        times = mdates.date2num(sample["timestamp_utc"].dt.tz_convert(None).to_numpy())
        time_edges = np.concatenate(([times[0] - (times[1] - times[0]) / 2], (times[:-1] + times[1:]) / 2,
                                      [times[-1] + (times[-1] - times[-2]) / 2]))
        mesh = ax.pcolormesh(time_edges, edges, np.log1p(matrix), shading="auto", cmap="Blues")
        ax.axhline(0, color=INK, linewidth=0.8, alpha=0.8)
        ax.set_ylabel("Giá tương đối với mid (tick)")
        ax.set_title(f"{symbol} · 5 phút liên tục · bắt đầu {sample['timestamp_utc'].iloc[0].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=mdates.UTC))
        plotted.append(mesh)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.10, right=0.89, hspace=0.32)
    colorbar_axis = fig.add_axes([0.92, 0.18, 0.018, 0.68])
    fig.colorbar(plotted[-1], cax=colorbar_axis, label="log(1 + khối lượng tại mức giá)")
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def _obi_forward_return_figure(output: Path, frames: dict[str, pd.DataFrame] | None = None, *,
                               filename: str = "obi_vs_forward_return_validation.png",
                               title: str = "OBI L10 và lợi suất mid-price tương lai · mô tả trên tuần validation") -> dict[str, dict]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    summary: dict[str, dict] = {}
    for ax, symbol in zip(axes, SYMBOLS):
        frame = frames[symbol] if frames is not None else _read_validation(symbol)
        frame = frame.loc[frame["book_valid"] & frame["obi_qty_10"].notna() & frame["mid_price"].notna()].copy()
        future = frame[["timestamp_utc", "mid_price", "book_valid"]].copy()
        future["timestamp_utc"] -= pd.to_timedelta(PRIMARY_HORIZON, unit="s")
        future = future.rename(columns={"mid_price": "future_mid", "book_valid": "future_valid"})
        joined = frame.merge(future, on="timestamp_utc", how="inner", validate="one_to_one")
        joined = joined.loc[joined["future_valid"] & (joined["future_mid"] > 0) & (joined["mid_price"] > 0)]
        joined["forward_return_bps"] = 10000.0 * np.log(joined["future_mid"] / joined["mid_price"])
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=["obi_qty_10", "forward_return_bps"])
        if len(joined) < 100:
            raise RuntimeError(f"Too few paired validation observations for {symbol} OBI plot")
        joined["obi_decile"] = pd.qcut(joined["obi_qty_10"], 10, labels=False, duplicates="drop")
        grouped = joined.groupby("obi_decile")["forward_return_bps"].agg(["mean", "median", "count"])
        summary[symbol] = {"n": int(len(joined)), "spearman": float(joined["obi_qty_10"].corr(joined["forward_return_bps"], method="spearman")),
                           "decile_mean_bps": [float(v) for v in grouped["mean"]], "decile_median_bps": [float(v) for v in grouped["median"]]}
        x = grouped.index.to_numpy(dtype=float) + 1
        ax.plot(x, grouped["mean"], marker="o", color=DATA, label="Trung bình")
        ax.plot(x, grouped["median"], marker="s", color=BREACH, linewidth=1, label="Trung vị")
        ax.axhline(0, color=MUTE, linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xlabel("Nhóm OBI L10 theo decile")
        ax.set_title(f"{symbol} · n={len(joined):,} · t+{PRIMARY_HORIZON}s")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mid-price forward return (bps)")
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.text(0.5, 0.01, "Decile được xác định trên chính mẫu mô tả; không có dải tin cậy vì chuỗi 1 giây tự tương quan mạnh.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)
    return summary


def _deeplob_comparison(output: Path, predictions_path: Path, metrics_path: Path) -> bool:
    if not predictions_path.is_file() or not metrics_path.is_file():
        return False
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    metric_key = {"direction": "macro_f1", "stress": "pr_auc"}
    for ax, task in zip(axes, ("direction", "stress")):
        task_rows = [entry for entry in summary["results"] if entry["task"] == task]
        names = ["deeplob", "logistic", "xgboost"]
        labels = ["DeepLOB", "Logistic", "XGBoost"]
        colors = [DATA, INK, BREACH]
        x = np.arange(len(SYMBOLS))
        width = 0.23
        for idx, name in enumerate(names):
            values = []
            for symbol in SYMBOLS:
                row = next(entry for entry in task_rows if entry["symbol"] == symbol)
                values.append(row["metrics"][name].get(metric_key[task]))
            ax.bar(x + (idx - 1) * width, values, width, label=labels[idx], color=colors[idx])
        ax.set_xticks(x, SYMBOLS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1" if task == "direction" else "PR-AUC")
        ax.set_title(f"{task} · validation only")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("DeepLOB so với Logistic/XGBoost · validation pilot · cùng điểm dự báo", fontsize=13, weight="bold")
    fig.text(0.5, 0.01, f"DeepLOB train cap={summary['train_cap_per_fit']:,}; validation cap/task={summary['validation_row_cap_per_task']:,}. Không dùng holdout.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(output / "deeplob_vs_baselines_validation.png", dpi=180)
    plt.close(fig)
    return True


def _deeplob_diagnostics(output: Path, predictions_path: Path, metrics_path: Path) -> bool:
    if not predictions_path.is_file() or not metrics_path.is_file():
        return False
    from sklearn.metrics import confusion_matrix

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    last_image = None
    for row, symbol in enumerate(SYMBOLS):
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            subset = predictions.loc[
                (predictions["symbol"] == symbol)
                & (predictions["task"] == task)
                & (predictions["split"] == "development_validation_only")
            ]
            labels = ["down", "flat", "up"] if task == "direction" else [0, 1]
            model_column = "deeplob_prediction"
            true = subset["y_true"].astype(str if task == "direction" else int)
            predicted = subset[model_column].astype(str if task == "direction" else int)
            typed_labels = labels if task == "direction" else [0, 1]
            cm = confusion_matrix(true, predicted, labels=typed_labels)
            denom = cm.sum(axis=1, keepdims=True)
            norm = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)
            last_image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(labels)), labels=[str(x) for x in labels])
            ax.set_yticks(range(len(labels)), labels=[str(x) for x in labels])
            ax.set_xlabel("Dự đoán")
            ax.set_ylabel("Thực tế")
            ax.set_title(f"{symbol} · {task} · validation n={len(subset):,}")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center",
                            color=INK if norm[i, j] < 0.55 else SURFACE, fontsize=8)
    fig.suptitle("DeepLOB · ma trận nhầm lẫn trên validation (tối đa 2.500 điểm/mục tiêu)", fontsize=12, weight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.10, right=0.88, hspace=0.40, wspace=0.34)
    cax = fig.add_axes([0.91, 0.20, 0.018, 0.60])
    fig.colorbar(last_image, cax=cax, label="Tỷ lệ trong lớp thực tế")
    fig.savefig(output / "deeplob_confusion_validation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for row, symbol in enumerate(SYMBOLS):
        for col, task in enumerate(("direction", "stress")):
            ax = axes[row, col]
            result = next(
                item for item in summary["results"]
                if item["symbol"] == symbol and item["task"] == task
            )["metrics"]["deeplob"]
            ax.plot([0, 1], [0, 1], linestyle="--", color=MUTE, linewidth=1, label="Hoàn hảo")
            if task == "direction":
                colors = {"down": DATA, "flat": MUTE, "up": BREACH}
                for label, bins in result.get("calibration_by_class", {}).items():
                    if bins:
                        ax.plot([point["mean_probability"] for point in bins],
                                [point["observed_rate"] for point in bins], marker="o", markersize=3,
                                color=colors.get(label), label=label)
            else:
                bins = result.get("calibration_bins", [])
                if bins:
                    ax.plot([point["mean_probability"] for point in bins],
                            [point["observed_rate"] for point in bins], marker="o", markersize=4,
                            color=BREACH, label="stress")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Xác suất dự báo trung bình", ylabel="Tần suất thực tế")
            ax.grid(alpha=0.2)
            ax.set_title(f"{symbol} · {task} · n={result['n']:,}")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("DeepLOB · calibration trên validation, không dùng holdout", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "deeplob_calibration_validation.png", dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = ROOT / "outputs" / "research_figures"
    output.mkdir(parents=True, exist_ok=True)
    metrics = {symbol: _metrics(symbol) for symbol in SYMBOLS}
    _confusion_figure(output, metrics)
    _calibration_figure(output, metrics)
    _performance_over_time_figure(output, metrics)
    _lob_heatmap(output)
    _obi_forward_return_figure(output)
    deeplob_metrics = ROOT / "outputs" / "deeplob" / f"validation_metrics_h{PRIMARY_HORIZON}.json"
    deeplob_predictions = ROOT / "outputs" / "deeplob" / f"validation_predictions_h{PRIMARY_HORIZON}.csv"
    has_deeplob = _deeplob_comparison(output, deeplob_predictions, deeplob_metrics)
    has_deeplob_diagnostics = _deeplob_diagnostics(output, deeplob_predictions, deeplob_metrics)
    index = [
        "# Research figures — historical archive pilot",
        "",
        "All figures use locally stored licensed archive data or saved aggregate evaluation outputs. They are for internal research/report preparation; follow the provider's redistribution and product-use restrictions.",
        "",
        "- `lob_heatmap_validation.png`: BTC/ETH five-minute contiguous LOB top-10 window from validation week 2026-10-05.",
        "- `obi_vs_forward_return_validation.png`: OBI L10 deciles versus five-second mid-price forward return, validation week only; descriptive, no iid confidence bands.",
        "- `confusion_matrices_holdout.png`: confusion matrices from saved aggregate holdout summaries; no holdout data re-read or rescoring.",
        "- `calibration_curves_holdout.png`: calibration bins from saved aggregate holdout summaries.",
        "- `performance_over_time_pilot.png`: validation-week versus once-scored holdout performance; this two-period pilot cannot support a long-term stability claim.",
    ]
    index.append("- `deeplob_vs_baselines_validation.png`: DeepLOB/Logistic/XGBoost comparison on development validation only." if has_deeplob else "- DeepLOB comparison is omitted until validation-only metrics are available.")
    if has_deeplob_diagnostics:
        index += [
            "- `deeplob_confusion_validation.png`: DeepLOB confusion matrices from saved pointwise development-validation predictions.",
            "- `deeplob_calibration_validation.png`: DeepLOB class-specific calibration curves on development validation.",
        ]
    index += [
        "",
        "The holdout test remains scored once. Confusion/calibration/performance figures read only the aggregate JSON metrics saved by that run; DeepLOB uses development validation and does not score the holdout.",
    ]
    (output / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    for path in sorted(output.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
