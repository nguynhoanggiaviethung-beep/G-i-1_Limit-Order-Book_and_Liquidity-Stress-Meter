"""Summarize development walk-forward fold metrics into research tables (no model is fitted).

Reads ``<metrics-dir>/<SYMBOL>/fold_metrics.csv`` written by ``run_lob_classification.py
--development-only`` and writes:

* ``model_summary.csv``: mean and standard deviation over validation weeks per model.
* ``paired_fold_comparisons.csv``: per-week paired differences for the RQ1/RQ2 contrasts, with the
  number of weeks won, a paired t-test and a Wilcoxon signed-rank test across weeks.
* ``depth_level_sensitivity.csv`` and ``delta_sensitivity.csv``: the L and δ ablations.
* ``walk_forward_summary.md``: the same tables for reading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT

DIRECTION_METRICS = ("macro_f1", "mcc", "balanced_accuracy", "brier")
STRESS_METRICS = ("pr_auc", "mcc", "macro_f1", "brier", "ece")
# (task, challenger, reference, research question)
CONTRASTS = (
    ("direction", "logistic_obi_only", "class_frequency", "RQ1: OBI vs class frequency"),
    ("direction", "logistic_obi_only", "direction_persistence", "RQ1: OBI vs persistence"),
    ("direction", "logistic_obi_plus_ofi", "logistic_obi_only", "RQ2: adding OFI to OBI"),
    ("direction", "logistic", "logistic_obi_plus_ofi", "Full feature set vs OBI+OFI"),
    ("direction", "xgboost", "logistic", "XGBoost vs Logistic"),
    ("direction", "random_forest", "logistic", "Random forest vs Logistic"),
    ("stress", "logistic", "stress_persistence", "Stress: Logistic vs persistence"),
    ("stress", "random_forest", "stress_persistence", "Stress: random forest vs persistence"),
    ("stress", "xgboost", "stress_persistence", "Stress: XGBoost vs persistence"),
)
LOWER_IS_BETTER = {"brier", "ece"}


def load_fold_metrics(metrics_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(metrics_dir.glob("*/fold_metrics.csv"))]
    if not frames:
        raise FileNotFoundError(f"No */fold_metrics.csv under {metrics_dir}; run run_lob_classification.py --development-only first")
    return pd.concat(frames, ignore_index=True)


def model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, horizon, task, model), group in metrics.groupby(["symbol", "horizon_seconds", "task", "model"]):
        if model.startswith(("logistic_delta_quantile", "logistic_depth_levels")):
            continue
        row = {"symbol": symbol, "horizon_seconds": horizon, "task": task, "model": model, "weeks": int(group["week_utc"].nunique())}
        for metric in DIRECTION_METRICS if task == "direction" else STRESS_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna() if metric in group else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, horizon), group in metrics.groupby(["symbol", "horizon_seconds"]):
        for task, challenger, reference, question in CONTRASTS:
            subset = group.loc[group["task"] == task]
            for metric in DIRECTION_METRICS if task == "direction" else STRESS_METRICS:
                if subset.empty or metric not in subset:
                    continue
                wide = subset.pivot_table(index="week_utc", columns="model", values=metric, aggfunc="first")
                if challenger not in wide or reference not in wide:
                    continue
                pair = wide[[challenger, reference]].apply(pd.to_numeric, errors="coerce").dropna()
                if pair.empty:
                    continue
                difference = pair[challenger] - pair[reference]
                better = -difference if metric in LOWER_IS_BETTER else difference
                row = {
                    "symbol": symbol, "horizon_seconds": horizon, "question": question, "task": task,
                    "challenger": challenger, "reference": reference, "metric": metric, "weeks": int(len(pair)),
                    "challenger_mean": float(pair[challenger].mean()), "reference_mean": float(pair[reference].mean()),
                    "mean_difference": float(difference.mean()),
                    "weeks_challenger_better": int((better > 0).sum()),
                    "paired_t_p": np.nan, "wilcoxon_p": np.nan,
                }
                if len(pair) >= 3 and difference.std(ddof=1) > 0:
                    row["paired_t_p"] = float(ttest_rel(pair[challenger], pair[reference]).pvalue)
                    if (difference != 0).sum() >= 3:
                        row["wilcoxon_p"] = float(wilcoxon(difference).pvalue)
                rows.append(row)
    return pd.DataFrame(rows)


def sensitivity(metrics: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    subset = metrics.loc[metrics["model"].str.startswith(prefix)].copy()
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["symbol", "horizon_seconds", column])[["macro_f1", "mcc", "balanced_accuracy", "brier"]]
        .agg(["mean", "std"]).round(4).reset_index()
    )


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._\n"
    frame = frame.copy()
    frame.columns = [" ".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column) for column in frame.columns]
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    body = "".join(
        "| " + " | ".join(f"{value:.{digits}f}" if isinstance(value, float) else str(value) for value in row) + " |\n"
        for row in frame.itertuples(index=False)
    )
    return header + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "walk_forward_6m")
    args = parser.parse_args()
    metrics_dir = args.metrics_dir if args.metrics_dir.is_absolute() else REPOSITORY_ROOT / args.metrics_dir
    metrics = load_fold_metrics(metrics_dir)
    summary = model_summary(metrics)
    paired = paired_comparisons(metrics)
    levels = sensitivity(metrics, "depth_levels_L", "logistic_depth_levels")
    deltas = sensitivity(metrics, "delta_quantile", "logistic_delta_quantile")
    summary.to_csv(metrics_dir / "model_summary.csv", index=False)
    paired.to_csv(metrics_dir / "paired_fold_comparisons.csv", index=False)
    levels.to_csv(metrics_dir / "depth_level_sensitivity.csv", index=False)
    deltas.to_csv(metrics_dir / "delta_sensitivity.csv", index=False)
    primary = paired.loc[paired["metric"].isin(["macro_f1", "mcc", "pr_auc"])]
    (metrics_dir / "walk_forward_summary.md").write_text(
        "# Development walk-forward summary\n\n"
        "Development validation weeks only; the untouched test was not opened. "
        "Means are over weekly folds. Week-level tests treat each validation week as one paired observation.\n\n"
        "## Models\n\n" + _markdown(summary)
        + "\n## Paired contrasts (macro F1, MCC, PR-AUC)\n\n"
        + _markdown(primary[["symbol", "horizon_seconds", "question", "metric", "weeks", "challenger_mean", "reference_mean", "mean_difference", "weeks_challenger_better", "paired_t_p", "wilcoxon_p"]])
        + "\n## Depth levels L (Logistic on one OBI level)\n\n" + _markdown(levels)
        + "\n## Flat-band δ quantile (Logistic, all features)\n\n" + _markdown(deltas),
        encoding="utf-8",
    )
    print(f"Wrote walk-forward tables for {metrics['symbol'].nunique()} symbol(s), {metrics['week_utc'].nunique()} week(s) to {metrics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
