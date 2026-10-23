"""Run the DeepLOB comparison on development validation weeks only.

This entry point deliberately loads only partitions dated before the frozen
final-test week. It writes pointwise validation predictions for research plots
and never trains on, predicts, or scores the once-used holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import classification_metrics, expanding_week_folds
from app.labels.direction import fit_direction_delta, make_direction_labels
from app.labels.stress import fit_primary_stress_thresholds, make_stress_labels
from app.models.deeplob_classifier import DeepLOBClassifier, build_lob_sequences
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, make_classifiers
from scripts.run_lob_classification import (
    DIRECTION_CLASSES,
    STRESS_CLASSES,
    _align_probabilities,
    _stress_threshold_rows,
)


DEFAULT_VALIDATION_ROWS = 2500


def _load_pretest_features(symbol: str, settings, test_start: pd.Timestamp) -> pd.DataFrame:
    root = settings.data_dir / "features"
    paths = sorted(root.rglob("features_1s.parquet"))
    selected = []
    for path in paths:
        if f"symbol={symbol}" not in path.parts:
            continue
        date_parts = [part.removeprefix("date=") for part in path.parts if part.startswith("date=")]
        if not date_parts:
            raise RuntimeError(f"Refusing to load unpartitioned feature file without a date key: {path}")
        if pd.Timestamp(date_parts[-1], tz="UTC") < test_start:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No pre-test feature partitions found for {symbol} under {root}")
    frame = pd.concat((pd.read_parquet(path) for path in selected), ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"])
    frame = frame.loc[frame["timestamp_utc"] < test_start]
    return frame.sort_values("timestamp_utc", kind="mergesort").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _aligned_deep_probabilities(model, x: np.ndarray, class_order: tuple[object, ...]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    aligned = np.zeros((len(x), len(class_order)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        if label in class_order:
            aligned[:, class_order.index(label)] = raw[:, source_index]
    return aligned


def _sequence_training_sample(labels: pd.Series, indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    selected_labels = labels.iloc[indices].dropna()
    if len(selected_labels) <= maximum:
        return selected_labels.index.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label, group in selected_labels.groupby(selected_labels):
        target = max(1, int(round(maximum * len(group) / len(selected_labels))))
        count = min(len(group), target)
        chosen.extend(rng.choice(group.index.to_numpy(dtype=int), count, replace=False).tolist())
    chosen = sorted(set(chosen))
    if len(chosen) > maximum:
        chosen = np.asarray(chosen)[np.linspace(0, len(chosen) - 1, maximum, dtype=int)].tolist()
    if len(chosen) < maximum:
        remaining = np.setdiff1d(selected_labels.index.to_numpy(dtype=int), np.asarray(chosen, dtype=int))
        take = min(maximum - len(chosen), len(remaining))
        if take:
            chosen.extend(rng.choice(remaining, take, replace=False).tolist())
    return np.asarray(sorted(set(chosen)), dtype=int)


def _build_sequence_batch(frame: pd.DataFrame, labels: pd.Series, endpoints: np.ndarray, lookback: int):
    x, valid_endpoints = build_lob_sequences(frame, endpoints, lookback)
    y = labels.loc[valid_endpoints].to_numpy()
    if str(labels.dtype) == "Int64":
        y = y.astype(np.int64)
    return x, y, valid_endpoints


def _fit_deeplob(frame, labels, train_indices, lookback, cap, seed):
    sampled = _sequence_training_sample(labels, train_indices, cap, seed)
    x, y, endpoints = _build_sequence_batch(frame, labels, sampled, lookback)
    if len(y) < 60 or len(np.unique(y)) < 2:
        raise RuntimeError(f"Only {len(y)} contiguous DeepLOB training sequences remain")
    model = DeepLOBClassifier(lookback=lookback, random_state=seed)
    model.fit(x, y)
    return model, endpoints


def _predict_task(
    *, symbol: str, task: str, frame: pd.DataFrame, labels: pd.Series,
    train_indices: np.ndarray, validation_indices: np.ndarray, horizon: int,
    week: str, settings, validation_cap: int, thresholds: dict | None = None,
):
    class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
    train_cap = int(settings.lob_deeplob_max_train_rows)
    lookback = int(settings.lob_deeplob_lookback_seconds)
    sampled_train = _sequence_training_sample(labels, train_indices, train_cap, 1000 + horizon + (1000 if task == "stress" else 0))
    x_train_sequence, y_train, train_endpoints = _build_sequence_batch(frame, labels, sampled_train, lookback)
    if len(y_train) < 60 or len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{symbol} h={horizon} {task}: insufficient contiguous DeepLOB training sequences")
    deep = DeepLOBClassifier(lookback=lookback, random_state=(1000 if task == "direction" else 2000) + horizon)
    deep.fit(x_train_sequence, y_train)

    valid = validation_indices
    if len(valid) > validation_cap:
        valid = valid[np.linspace(0, len(valid) - 1, validation_cap, dtype=int)]
    x_valid_sequence, y_valid, endpoints = _build_sequence_batch(frame, labels, valid, lookback)
    if len(y_valid) < 30:
        raise RuntimeError(f"{symbol} h={horizon} {task}: fewer than 30 valid sequences remain")
    x_train_table = frame.loc[train_endpoints, LOB_CLASSIFICATION_FEATURES]
    x_valid_table = frame.loc[endpoints, LOB_CLASSIFICATION_FEATURES]
    y_train_aligned = labels.loc[train_endpoints].to_numpy()

    deep_probability = _aligned_deep_probabilities(deep, x_valid_sequence, class_order)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "deeplob": (deep.predict(x_valid_sequence), deep_probability),
    }
    for model_name in ("logistic", "xgboost"):
        model = make_classifiers(task).get(model_name)
        if model is None:
            raise RuntimeError(f"The {model_name} comparison model is unavailable; install project ML requirements")
        model.fit(x_train_table, y_train_aligned)
        predictions[model_name] = (
            np.asarray(model.predict(x_valid_table)),
            _align_probabilities(model, x_valid_table, class_order),
        )

    result = {
        "symbol": symbol,
        "task": task,
        "horizon_seconds": horizon,
        "week_utc": week,
        "split": "development_validation_only",
        "n_train_after_sequence_filter": int(len(y_train)),
        "n_validation": int(len(y_valid)),
        "class_labels": [str(label) for label in class_order],
        "thresholds": thresholds,
        "metrics": {
            name: classification_metrics(
                y_valid, prediction, probability,
                labels=list(class_order), stress=(task == "stress"),
            )
            for name, (prediction, probability) in predictions.items()
        },
    }
    output = pd.DataFrame({
        "symbol": symbol,
        "task": task,
        "horizon_seconds": horizon,
        "week_utc": week,
        "split": "development_validation_only",
        "timestamp_utc": pd.to_datetime(frame.loc[endpoints, "timestamp_utc"], utc=True).astype(str).to_numpy(),
        "y_true": y_valid,
    })
    label_names = ("down", "flat", "up") if task == "direction" else ("no_stress", "stress")
    for model_name, (prediction, probability) in predictions.items():
        output[f"{model_name}_prediction"] = prediction
        for index, label_name in enumerate(label_names):
            output[f"{model_name}_p_{label_name}"] = probability[:, index]
    return result, output


def run_validation(settings, symbols: tuple[str, ...], horizon: int, validation_cap: int) -> tuple[list[dict], list[pd.DataFrame]]:
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc).tz_convert("UTC")
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    if not folds:
        raise RuntimeError("No full walk-forward validation weeks are configured")
    all_metrics, all_predictions = [], []
    for symbol in symbols:
        frame = _load_pretest_features(symbol, settings, test_start)
        times = pd.to_datetime(frame["timestamp_utc"], utc=True)
        first_start = pd.Timestamp(f"{folds[0]}T00:00:00Z")
        embargo = pd.to_timedelta(settings.lob_embargo_seconds, unit="s")
        first_cutoff = first_start - embargo
        first_train = frame.loc[times < first_cutoff]
        delta = fit_direction_delta(first_train, horizon, settings.lob_direction_delta_quantile)
        direction_labels = make_direction_labels(frame, horizon, delta)
        for week in folds:
            start = pd.Timestamp(f"{week}T00:00:00Z")
            end = start + pd.to_timedelta(7, unit="D")
            cutoff = start - pd.to_timedelta(settings.lob_embargo_seconds, unit="s")
            train_direction = np.flatnonzero((times < cutoff) & direction_labels.notna())
            valid_direction = np.flatnonzero(
                (times >= start) & (times < end - pd.to_timedelta(horizon, unit="s")) & direction_labels.notna()
            )
            direction_metrics, direction_predictions = _predict_task(
                symbol=symbol, task="direction", frame=frame, labels=direction_labels,
                train_indices=train_direction, validation_indices=valid_direction,
                horizon=horizon, week=week, settings=settings, validation_cap=validation_cap,
            )
            all_metrics.append(direction_metrics)
            all_predictions.append(direction_predictions)

            thresholds = fit_primary_stress_thresholds(
                _stress_threshold_rows(frame, times < cutoff, settings.lob_stress_definition),
                settings.lob_stress_definition,
                settings.lob_stress_quantile,
                settings.lob_stress_depth_quantile,
            )
            stress_labels = make_stress_labels(frame, horizon, thresholds).astype("Int64")
            train_stress = np.flatnonzero((times < cutoff) & stress_labels.notna())
            valid_stress = np.flatnonzero(
                (times >= start) & (times < end - pd.to_timedelta(horizon, unit="s")) & stress_labels.notna()
            )
            stress_metrics, stress_predictions = _predict_task(
                symbol=symbol, task="stress", frame=frame, labels=stress_labels,
                train_indices=train_stress, validation_indices=valid_stress,
                horizon=horizon, week=week, settings=settings, validation_cap=validation_cap,
                thresholds=thresholds.as_dict(),
            )
            all_metrics.append(stress_metrics)
            all_predictions.append(stress_predictions)
            print(
                f"{symbol} h={horizon} {week}: direction n={direction_metrics['n_validation']}, "
                f"stress n={stress_metrics['n_validation']} (validation only; final test untouched)",
                flush=True,
            )
    return all_metrics, all_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.yaml")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10))
    parser.add_argument("--validation-max-rows", type=int, default=DEFAULT_VALIDATION_ROWS)
    args = parser.parse_args()
    if args.validation_max_rows < 30:
        parser.error("--validation-max-rows must be at least 30")
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    symbols = (args.symbol,) if args.symbol else settings.symbols
    horizon = args.horizon or settings.lob_primary_horizon_seconds
    metrics, predictions = run_validation(settings, symbols, horizon, args.validation_max_rows)
    out_dir = REPOSITORY_ROOT / "outputs" / "deeplob"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"validation_metrics_h{horizon}.json"
    predictions_path = out_dir / f"validation_predictions_h{horizon}.csv"
    metrics_path.write_text(json.dumps({
        "schema": "deeplob_validation.v1",
        "status": "completed_development_validation_only",
        "study_label": "historical_archive_pilot",
        "holdout_scored": False,
        "holdout_accessed": False,
        "comparison_models": ["deeplob", "logistic", "xgboost"],
        "train_cap_per_fit": int(settings.lob_deeplob_max_train_rows),
        "validation_row_cap_per_task": int(args.validation_max_rows),
        "results": metrics,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(predictions_path, index=False)
    status_path = out_dir / "status.json"
    status_path.write_text(json.dumps({
        "status": "validated_on_development_only",
        "metrics_file": metrics_path.name,
        "predictions_file": predictions_path.name,
        "holdout_scored": False,
        "do_not_run_against_holdout": True,
        "configured_train_cap": int(settings.lob_deeplob_max_train_rows),
        "horizon_seconds": horizon,
        "note": "Validation evidence only; the historical pilot final holdout remains scored exactly once by the classical-model run.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Pointwise validation predictions: {predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
