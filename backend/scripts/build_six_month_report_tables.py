"""Build the six report tables of the six-month study from saved outputs.

1. Class distribution: direction classes and stress prevalence, development training vs test.
2. Feature summary: the 27 model features on valid development seconds (test partitions are not read).
3. Baseline metrics: every classical model and baseline on the once-scored test.
4. Deep model metrics: DeepLOB vs Logistic and XGBoost over the 12 development validation weeks.
5. Performance by volatility regime: selected model vs persistence on the test, with bootstrap CIs.
6. Latency and model complexity: served artifacts, plus DeepLOB timed on synthetic input.

Nothing is refitted on real data and the test is not rescored. Each table is written as CSV, and all
six together as Markdown, to ``outputs/tables/report_6m/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES

SYMBOLS = ("BTCUSDT", "ETHUSDT")
DIRECTION_LABELS = ("down", "flat", "up")
REGIMES = ("all", "vol_low", "vol_mid", "vol_high", "session_asia", "session_europe", "session_americas")
FEATURE_QUANTILES = (0.01, 0.5, 0.99)


def _summary(metrics_dir: Path, symbol: str) -> dict:
    path = metrics_dir / symbol / "classification_metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("final_test_scored_once") or not (metrics_dir / symbol / "final_test_opened.json").is_file():
        raise RuntimeError(f"{path} is not a once-scored final-test summary")
    return value


def class_distribution(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for symbol, summary in summaries.items():
        for horizon, result in sorted(summary["horizons"].items(), key=lambda item: int(item[0])):
            train = result["direction_class_distribution_train"]
            confusion = np.asarray(result["direction_test"]["confusion_matrix"], dtype=float)
            test = dict(zip(result["direction_test"]["labels"], confusion.sum(axis=1) / confusion.sum()))
            for label in DIRECTION_LABELS:
                rows.append({"symbol": symbol, "horizon_seconds": int(horizon), "task": "direction", "class": label,
                             "development_train_share": train[label], "test_share": test[label], "test_n": int(confusion.sum())})
            rows.append({"symbol": symbol, "horizon_seconds": int(horizon), "task": "stress", "class": "stress (composite)",
                         "development_train_share": result["stress_positive_rate_train"],
                         "test_share": result["stress_test"]["positive_rate"], "test_n": int(result["stress_test"]["n"])})
            spread = result["stress_spread_only_sensitivity"]
            rows.append({"symbol": symbol, "horizon_seconds": int(horizon), "task": "stress", "class": "stress (spread-only Q95, sensitivity)",
                         "development_train_share": spread["thresholds_train"]["positive_rate_train"],
                         "test_share": spread["test_metrics"]["positive_rate"], "test_n": int(spread["test_metrics"]["n"])})
    return pd.DataFrame(rows)


def stress_prevalence_by_month(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for symbol, summary in summaries.items():
        result = summary["horizons"]["5"]
        for split, key in (("development", "stress_positive_rate_by_month_development"), ("test", "stress_positive_rate_by_month_test")):
            for month, item in result[key].items():
                rows.append({"symbol": symbol, "horizon_seconds": 5, "split": split, "month": month, "n": item["n"], "positive_rate": item["positive_rate"]})
    return pd.DataFrame(rows)


def feature_summary(settings) -> pd.DataFrame:
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    rows = []
    columns = ["timestamp_utc", "book_valid", *LOB_CLASSIFICATION_FEATURES]
    for symbol in SYMBOLS:
        parts = []
        for path in sorted((settings.data_dir / "features" / f"symbol={symbol}").glob("date=*/features_1s.parquet")):
            if pd.Timestamp(path.parent.name.removeprefix("date="), tz="UTC") >= test_start:
                continue  # untouched test and later
            frame = pd.read_parquet(path, columns=columns)
            frame = frame.loc[frame["book_valid"].fillna(False).astype(bool) & (pd.to_datetime(frame["timestamp_utc"], utc=True) < test_start)]
            parts.append(frame[list(LOB_CLASSIFICATION_FEATURES)].apply(pd.to_numeric, errors="coerce").astype(np.float32))
        values = pd.concat(parts, ignore_index=True)
        del parts
        for feature in LOB_CLASSIFICATION_FEATURES:
            column = values[feature].to_numpy(dtype=np.float64)
            finite = column[np.isfinite(column)]
            quantiles = np.quantile(finite, FEATURE_QUANTILES) if len(finite) else [np.nan] * len(FEATURE_QUANTILES)
            rows.append({"symbol": symbol, "feature": feature, "n_valid_seconds": int(len(column)),
                         "missing_share": float(1 - len(finite) / len(column)) if len(column) else np.nan,
                         "mean": float(finite.mean()) if len(finite) else np.nan, "sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                         **{f"p{int(q * 100)}": float(v) for q, v in zip(FEATURE_QUANTILES, quantiles)}})
        print(f"{symbol}: summarized {len(values):,} valid development seconds", flush=True)
        del values
    return pd.DataFrame(rows)


def baseline_metrics(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for symbol, summary in summaries.items():
        for horizon, result in sorted(summary["horizons"].items(), key=lambda item: int(item[0])):
            direction = {f"selected_calibrated ({result['direction_model']})": result["direction_test"], **result["direction_test_comparisons"]}
            for model, metrics in direction.items():
                rows.append({"symbol": symbol, "horizon_seconds": int(horizon), "task": "direction", "model": model, "n": metrics["n"],
                             "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"], "mcc": metrics["mcc"],
                             "brier": metrics["brier"], "ece_max_class": max(metrics["ece_by_class"].values()), "accuracy": metrics["accuracy"]})
            threshold = result["stress_decision_threshold"]
            stress = {f"selected_calibrated ({result['stress_model']}, threshold {threshold:.2f})": result["stress_test"], **result["stress_test_comparisons"]}
            for model, metrics in stress.items():
                rows.append({"symbol": symbol, "horizon_seconds": int(horizon), "task": "stress", "model": model, "n": metrics["n"],
                             "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"], "mcc": metrics["mcc"],
                             "brier": metrics["brier"], "ece": metrics["ece"], "pr_auc": metrics["pr_auc"],
                             "precision_stress": metrics["precision_by_class"]["1"], "recall_stress": metrics["recall_by_class"]["1"],
                             "accuracy": metrics["accuracy"]})
    return pd.DataFrame(rows)


def deep_model_metrics(deeplob_path: Path) -> pd.DataFrame:
    payload = json.loads(deeplob_path.read_text(encoding="utf-8"))
    if payload.get("holdout_accessed") or payload.get("holdout_scored"):
        raise RuntimeError("DeepLOB results must be development-only")
    records = []
    for result in payload["results"]:
        for model, metrics in result["metrics"].items():
            records.append({"symbol": result["symbol"], "task": result["task"], "week_utc": result["week_utc"], "model": model,
                            **{key: metrics.get(key) for key in ("macro_f1", "balanced_accuracy", "mcc", "brier", "pr_auc")}})
    weekly = pd.DataFrame(records)
    rows = []
    for (symbol, task), group in weekly.groupby(["symbol", "task"]):
        primary = "pr_auc" if task == "stress" else "mcc"
        deeplob = group.loc[group["model"] == "deeplob"].set_index("week_utc")[primary]
        for model, model_rows in group.groupby("model"):
            row = {"symbol": symbol, "task": task, "model": model, "weeks": int(len(model_rows))}
            for metric in ("macro_f1", "balanced_accuracy", "mcc", "brier", "pr_auc"):
                values = pd.to_numeric(model_rows[metric], errors="coerce").dropna()
                if len(values):
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_sd"] = float(values.std(ddof=1))
            if model != "deeplob":
                other = model_rows.set_index("week_utc")[primary]
                row[f"weeks_deeplob_better_on_{primary}"] = int((deeplob > other.reindex(deeplob.index)).sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs.update(train_cap=payload["train_cap_per_fit"], validation_cap=payload["validation_row_cap_per_task"])
    return table


def regime_performance(bootstrap_dir: Path) -> pd.DataFrame:
    rows = []
    for symbol in SYMBOLS:
        models = pd.read_csv(bootstrap_dir / f"{symbol}_model_metrics.csv")
        keep = ((models["task"] == "direction") & (models["metric"] == "mcc")) | ((models["task"] == "stress") & models["metric"].isin(["pr_auc", "mcc"]))
        keep &= models["model"].isin(["selected_calibrated", "direction_persistence", "stress_persistence"])
        rows.append(models.loc[keep])
    table = pd.concat(rows, ignore_index=True)
    table["model"] = table["model"].replace({"direction_persistence": "persistence", "stress_persistence": "persistence"})
    table["regime"] = pd.Categorical(table["regime"], REGIMES, ordered=True)
    return table.sort_values(["symbol", "task", "metric", "horizon_seconds", "regime", "model"]).reset_index(drop=True)


def _deeplob_complexity() -> dict:
    """Parameter count and CPU inference latency of the DeepLOB network, on synthetic input."""
    try:
        import torch
    except ImportError:
        return {"status": "unavailable", "reason": "PyTorch is not installed"}
    from torch import nn

    from app.models.deeplob_classifier import DeepLOBClassifier

    classifier = DeepLOBClassifier()
    rows = []
    for task, classes in (("direction", 3), ("stress", 2)):
        network = classifier._make_network(nn, 40, classes).eval()
        parameters = int(sum(p.numel() for p in network.parameters()))
        batch = torch.from_numpy(np.random.default_rng(0).standard_normal((1000, classifier.lookback, 40)).astype(np.float32))
        with torch.no_grad():
            network(batch[:10])
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                network(batch)
                timings.append((time.perf_counter() - started) / len(batch) * 1000)
        rows.append({"task": task, "parameters": parameters, "mean_ms_per_row": float(np.mean(timings)), "lookback_seconds": classifier.lookback})
    return {"status": "measured", "rows": rows}


def latency_complexity(summaries: dict[str, dict], artifact_dir: Path) -> pd.DataFrame:
    rows = []
    for symbol, summary in summaries.items():
        for horizon, result in sorted(summary["horizons"].items(), key=lambda item: int(item[0])):
            artifact = artifact_dir / symbol / f"horizon_{horizon}.joblib"
            for task in ("direction", "stress"):
                profile = result[f"{task}_runtime_profile"]
                complexity = profile["model_complexity"]
                rows.append({
                    "symbol": symbol, "horizon_seconds": int(horizon), "task": task, "model": result[f"{task}_model"],
                    "estimator": complexity.get("estimator"), "input_features": complexity.get("input_features"),
                    "trees_or_rounds": complexity.get("tree_count") or complexity.get("boosted_rounds"),
                    "tree_nodes": complexity.get("tree_node_count"),
                    "mean_ms_per_row": profile["prediction_latency"]["mean_ms_per_row"],
                    "p95_ms_per_row": profile["prediction_latency"]["p95_of_repeats_ms_per_row"],
                    "artifact_mb_both_tasks": round(artifact.stat().st_size / 1e6, 1) if artifact.is_file() else None,
                    "training_rows_per_fit": summary["max_train_rows_per_fit"],
                })
    deep = _deeplob_complexity()
    for row in deep.get("rows", []):
        rows.append({"symbol": "both", "horizon_seconds": 5, "task": row["task"], "model": "deeplob (development only)",
                     "estimator": "CNN + LSTM", "input_features": f"{row['lookback_seconds']} s x 40 book channels",
                     "parameters": row["parameters"], "mean_ms_per_row": row["mean_ms_per_row"],
                     "note": "latency on synthetic input, batch of 1,000, CPU"})
    return pd.DataFrame(rows)


def _markdown_table(table: pd.DataFrame) -> str:
    """Pipe table without an extra dependency; floats use 4 significant digits."""
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _markdown(tables: list[tuple[str, str, pd.DataFrame]]) -> str:
    parts = ["# Six-month study: report tables", "",
             "Generated by `backend/scripts/build_six_month_report_tables.py` from saved outputs; the untouched test was not rescored.", ""]
    for title, note, table in tables:
        parts += [f"## {title}", "", note, "", _markdown_table(table), ""]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings_6m.yaml")
    parser.add_argument("--metrics-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "final_test_6m")
    parser.add_argument("--deeplob", type=Path, default=REPOSITORY_ROOT / "outputs" / "deeplob_6m" / "validation_metrics_h5.json")
    parser.add_argument("--artifact-dir", type=Path, default=REPOSITORY_ROOT / "backend" / "artifacts" / "meter_6m")
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "report_6m")
    parser.add_argument("--skip-feature-summary", action="store_true", help="Reuse an existing table2 instead of rereading development partitions")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {symbol: _summary(args.metrics_dir, symbol) for symbol in SYMBOLS}

    table1 = class_distribution(summaries)
    table1b = stress_prevalence_by_month(summaries)
    feature_path = args.out_dir / "table2_feature_summary_development.csv"
    table2 = pd.read_csv(feature_path) if args.skip_feature_summary and feature_path.is_file() else feature_summary(settings)
    table3 = baseline_metrics(summaries)
    table4 = deep_model_metrics(args.deeplob)
    table5 = regime_performance(args.metrics_dir / "regime_bootstrap")
    table6 = latency_complexity(summaries, args.artifact_dir)

    outputs = [
        ("table1_class_distribution.csv", table1), ("table1b_stress_prevalence_by_month.csv", table1b),
        ("table2_feature_summary_development.csv", table2), ("table3_baseline_metrics_test.csv", table3),
        ("table4_deep_model_metrics_development.csv", table4), ("table5_performance_by_regime_test.csv", table5),
        ("table6_latency_complexity.csv", table6),
    ]
    for name, table in outputs:
        table.to_csv(args.out_dir / name, index=False)
        print(f"wrote {name} ({len(table)} rows)", flush=True)
    window = summaries[SYMBOLS[0]]["reproducibility"]
    test_label = f"test {window['final_test_start_utc'][:10]} – {window['final_test_end_utc'][:10]}"
    markdown = _markdown([
        ("Table 1. Class distribution", f"Shares of each class in development training rows and in the {test_label}.", table1),
        ("Table 1b. Stress prevalence by month (h = 5 s)", "Composite label; the pre-registered range is 2–10%.", table1b),
        ("Table 2. Feature summary", "Valid development seconds only; the untouched test was not read.", table2),
        ("Table 3. Baseline and classical model metrics", f"{test_label}, scored once. Uncalibrated candidates use argmax / a 0.5 threshold.", table3),
        ("Table 4. Deep model metrics", f"DeepLOB vs Logistic and XGBoost, h = 5 s, 12 development validation weeks, "
         f"{table4.attrs['train_cap']:,} training sequences per fit and up to {table4.attrs['validation_cap']:,} shared validation origins per week. Not scored on the test.", table4),
        ("Table 5. Performance by volatility regime", f"{test_label}; 95% CIs from a 1-hour block bootstrap within weeks (2,000 replicates). "
         "Volatility terciles use rv_300s cut-offs from development training rows.", table5),
        ("Table 6. Latency and model complexity", "Served artifacts: mean CPU prediction time per row over 1,000 test rows. DeepLOB: synthetic input.", table6),
    ])
    (args.out_dir / "report_tables.md").write_text(markdown, encoding="utf-8")
    print(f"wrote report_tables.md to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
