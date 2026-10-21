"""Walk-forward and one-time held-out evaluation of the DeepLOB CNN-LSTM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import classification_metrics, paired_classification_tests
from app.labels.direction import fit_direction_delta, make_direction_labels
from app.labels.stress import fit_primary_stress_thresholds, make_stress_labels, monthly_positive_rates
from app.models.deeplob_classifier import DeepLOBClassifier, build_lob_sequences
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, make_classifiers
from scripts.run_lob_classification import (
    DIRECTION_CLASSES,
    STRESS_CLASSES,
    _align_probabilities,
    _bounded_sample,
    _current_stress_state,
    _direction_persistence_predictions,
    _load_features,
    _stress_threshold_rows,
)


MAX_DEEPLOB_TRAIN_ROWS = 8000
MAX_FOLD_VALIDATION_ROWS = 2500
MAX_FINAL_TEST_ROWS = 10000


def _sample_indices(labels: pd.Series, indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    series = labels.iloc[indices].dropna()
    if len(series) <= maximum:
        return series.index.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for label, group in series.groupby(series):
        count = min(len(group), max(1, int(round(maximum * len(group) / len(series)))))
        selected.extend(rng.choice(group.index.to_numpy(dtype=int), count, replace=False).tolist())
    if len(selected) > maximum:
        selected = sorted(selected)
        selected = np.asarray(selected)[np.linspace(0, len(selected) - 1, maximum, dtype=int)].tolist()
    elif len(selected) < maximum:
        remainder = np.setdiff1d(series.index.to_numpy(dtype=int), np.asarray(selected, dtype=int))
        take = min(maximum - len(selected), len(remainder))
        if take:
            selected.extend(rng.choice(remainder, take, replace=False).tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def _evenly_spaced(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def _build_samples(frame: pd.DataFrame, labels: pd.Series, indices: np.ndarray, lookback: int):
    x, endpoints = build_lob_sequences(frame, indices, lookback)
    y = labels.loc[endpoints].to_numpy()
    if str(labels.dtype) == "Int64":
        y = y.astype(np.int64)
    return x, y, endpoints


def _fit_deeplob(frame, labels, indices, lookback, seed):
    sampled = _sample_indices(labels, indices, MAX_DEEPLOB_TRAIN_ROWS, seed)
    x, y, endpoints = _build_samples(frame, labels, sampled, lookback)
    if len(y) < 60 or len(np.unique(y)) < 2:
        raise RuntimeError(f"Only {len(y)} usable DeepLOB training sequences remain")
    model = DeepLOBClassifier(lookback=lookback, random_state=seed)
    model.fit(x, y)
    return model, endpoints


def _evaluate_fold(task, frame, labels, train_indices, validation_indices, lookback, month, seed):
    deep_model, _ = _fit_deeplob(frame, labels, train_indices, lookback, seed)
    validation_indices = _evenly_spaced(validation_indices, MAX_FOLD_VALIDATION_ROWS)
    x_sequence, y_valid, endpoints = _build_samples(frame, labels, validation_indices, lookback)
    if len(y_valid) < 30:
        return None
    x_table = frame.loc[endpoints, LOB_CLASSIFICATION_FEATURES]
    logistic_model = make_classifiers(task)["logistic"]
    train_sample = _sample_indices(labels, train_indices, MAX_DEEPLOB_TRAIN_ROWS, seed)
    x_train = frame.loc[train_sample, LOB_CLASSIFICATION_FEATURES]
    y_train = labels.loc[train_sample]
    logistic_model.fit(x_train, y_train.to_numpy())
    class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
    deep_prediction = deep_model.predict(x_sequence)
    deep_probability = deep_model.predict_proba(x_sequence)
    deep_classes = list(deep_model.classes_)
    aligned = np.zeros((len(x_sequence), len(class_order)), dtype=float)
    for column, label in enumerate(deep_classes):
        if label in class_order:
            aligned[:, class_order.index(label)] = deep_probability[:, column]
    logistic_prediction = logistic_model.predict(x_table)
    logistic_probability = _align_probabilities(logistic_model, x_table, class_order)
    return {
        "month": month,
        "n": int(len(y_valid)),
        "deeplob": classification_metrics(y_valid, deep_prediction, aligned, labels=list(class_order), stress=(task == "stress")),
        "logistic": classification_metrics(y_valid, logistic_prediction, logistic_probability, labels=list(class_order), stress=(task == "stress")),
    }


def _runtime_ms(model, sequences: np.ndarray) -> dict[str, float | int]:
    sample = sequences[: min(len(sequences), 1000)]
    if not len(sample):
        return {"rows": 0, "mean_ms_per_row": float("nan")}
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        model.predict_proba(sample)
        durations.append((time.perf_counter() - started) / len(sample) * 1000)
    return {"rows": int(len(sample)), "repeats": 3, "mean_ms_per_row": float(np.mean(durations)), "p95_of_repeats_ms_per_row": float(np.quantile(durations, 0.95))}


def run_symbol(symbol: str, horizons: tuple[int, ...], settings, allow_repeat: bool) -> dict:
    frame = _load_features(symbol)
    times = pd.to_datetime(frame["timestamp_utc"], utc=True)
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    test_end = pd.Timestamp(settings.lob_untouched_test_end_utc) + pd.Timedelta(seconds=1)
    development_end = pd.Timestamp(settings.lob_development_end_utc)
    folds = tuple(str(month) for month in pd.period_range(settings.lob_first_validation_month, development_end.to_period("M"), freq="M"))
    dev_cutoff = test_start - pd.Timedelta(seconds=settings.lob_embargo_seconds)
    first_cutoff = pd.Timestamp(f"{folds[0]}-01T00:00:00Z") - pd.Timedelta(seconds=settings.lob_embargo_seconds)
    first_train = frame.loc[times < first_cutoff]
    if first_train.empty:
        raise RuntimeError(f"Insufficient development history to fit {symbol} DeepLOB")
    result: dict[str, object] = {"symbol": symbol, "data_rows": int(len(frame)), "horizons": {}}

    for horizon in horizons:
        delta = fit_direction_delta(first_train, horizon, settings.lob_direction_delta_quantile)
        direction_labels = make_direction_labels(frame, horizon, delta)
        direction_folds = []
        stress_folds = []
        for month in folds:
            start = pd.Timestamp(f"{month}-01T00:00:00Z")
            end = start + pd.offsets.MonthBegin(1)
            cutoff = start - pd.Timedelta(seconds=settings.lob_embargo_seconds)
            train_indices = np.flatnonzero((times < cutoff) & direction_labels.notna())
            valid_indices = np.flatnonzero((times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & direction_labels.notna())
            try:
                score = _evaluate_fold("direction", frame, direction_labels, train_indices, valid_indices, settings.lob_deeplob_lookback_seconds, month, 1000 + horizon)
                if score:
                    direction_folds.append(score)
            except RuntimeError as exc:
                direction_folds.append({"month": month, "status": "skipped", "reason": str(exc)})

            stress_thresholds = fit_primary_stress_thresholds(
                _stress_threshold_rows(frame, times < cutoff, settings.lob_stress_definition),
                settings.lob_stress_definition,
                settings.lob_stress_quantile,
                settings.lob_stress_depth_quantile,
            )
            stress_labels = make_stress_labels(frame, horizon, stress_thresholds)
            train_indices = np.flatnonzero((times < cutoff) & stress_labels.notna())
            valid_indices = np.flatnonzero((times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & stress_labels.notna())
            try:
                score = _evaluate_fold("stress", frame, stress_labels.astype("Int64"), train_indices, valid_indices, settings.lob_deeplob_lookback_seconds, month, 2000 + horizon)
                if score:
                    stress_folds.append({**score, "thresholds": stress_thresholds.as_dict()})
            except RuntimeError as exc:
                stress_folds.append({"month": month, "status": "skipped", "reason": str(exc), "thresholds": stress_thresholds.as_dict()})

        final_thresholds = fit_primary_stress_thresholds(
            _stress_threshold_rows(frame, times < dev_cutoff, settings.lob_stress_definition),
            settings.lob_stress_definition,
            settings.lob_stress_quantile,
            settings.lob_stress_depth_quantile,
        )
        stress_labels = make_stress_labels(frame, horizon, final_thresholds).astype("Int64")
        direction_train = np.flatnonzero((times < dev_cutoff) & direction_labels.notna())
        stress_train = np.flatnonzero((times < dev_cutoff) & stress_labels.notna())
        direction_test = np.flatnonzero((times >= test_start) & (times < test_end - pd.Timedelta(seconds=horizon)) & direction_labels.notna())
        stress_test = np.flatnonzero((times >= test_start) & (times < test_end - pd.Timedelta(seconds=horizon)) & stress_labels.notna())
        train_positive_rate = float(stress_labels.iloc[stress_train].mean()) if len(stress_train) else float("nan")
        if not settings.lob_stress_min_positive_rate <= train_positive_rate <= settings.lob_stress_max_positive_rate:
            raise RuntimeError(f"{symbol} h={horizon}: development stress prevalence {train_positive_rate:.4f} is outside the configured range")

        task_results = {}
        final_models = {}
        for task, labels, train_indices, test_indices in (
            ("direction", direction_labels, direction_train, direction_test),
            ("stress", stress_labels, stress_train, stress_test),
        ):
            selected_test = _evenly_spaced(test_indices, MAX_FINAL_TEST_ROWS)
            model, training_endpoints = _fit_deeplob(frame, labels, train_indices, settings.lob_deeplob_lookback_seconds, 3000 + horizon)
            x_test, y_test, test_endpoints = _build_samples(frame, labels, selected_test, settings.lob_deeplob_lookback_seconds)
            if len(y_test) < 30:
                raise RuntimeError(f"Only {len(y_test)} final-test {task} sequences remain for {symbol} h={horizon}")
            class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
            raw_probability = model.predict_proba(x_test)
            probability = np.zeros((len(x_test), len(class_order)), dtype=float)
            for column, label in enumerate(model.classes_):
                if label in class_order:
                    probability[:, class_order.index(label)] = raw_probability[:, column]
            prediction = model.predict(x_test)
            logistic = make_classifiers(task)["logistic"]
            logistic_train = _sample_indices(labels, train_indices, MAX_DEEPLOB_TRAIN_ROWS, 3000 + horizon)
            x_logistic_train = frame.loc[logistic_train, LOB_CLASSIFICATION_FEATURES]
            logistic.fit(x_logistic_train, labels.loc[logistic_train].to_numpy())
            x_logistic_test = frame.loc[test_endpoints, LOB_CLASSIFICATION_FEATURES]
            logistic_prediction = logistic.predict(x_logistic_test)
            logistic_probability = _align_probabilities(logistic, x_logistic_test, class_order)
            metrics = classification_metrics(y_test, prediction, probability, labels=list(class_order), stress=(task == "stress"))
            logistic_metrics = classification_metrics(y_test, logistic_prediction, logistic_probability, labels=list(class_order), stress=(task == "stress"))
            paired_vs_logistic = paired_classification_tests(
                y_test, prediction, logistic_prediction, probability, logistic_probability,
                stress=(task == "stress"), horizon_seconds=horizon, baseline_name="logistic",
            )
            if task == "direction":
                persistence_all = _direction_persistence_predictions(frame, delta)
                persistence = persistence_all[test_endpoints]
                persistence_probability = np.column_stack([persistence == label for label in DIRECTION_CLASSES]).astype(float)
            else:
                persistence = _current_stress_state(frame.iloc[test_endpoints], final_thresholds)
                persistence_probability = np.column_stack([1 - persistence, persistence]).astype(float)
            paired_vs_persistence = paired_classification_tests(
                y_test, prediction, persistence, probability, persistence_probability,
                stress=(task == "stress"), horizon_seconds=horizon,
            )
            metrics.update({
                "logistic_baseline": logistic_metrics,
                "paired_tests_vs_logistic": paired_vs_logistic,
                "paired_tests_vs_persistence": paired_vs_persistence,
                "test_rows_sampled": int(len(y_test)),
                "test_rows_available": int(len(test_indices)),
                "runtime": _runtime_ms(model, x_test),
                "complexity": {"parameters": model.parameter_count_, "input_features_per_second": int(x_test.shape[2]), "lookback_seconds": int(model.lookback)},
            })
            task_results[task] = metrics
            final_models[task] = model

        directory = REPOSITORY_ROOT / "backend" / "artifacts" / "deeplob" / symbol / f"horizon_{horizon}"
        directory.mkdir(parents=True, exist_ok=True)
        final_models["direction"].save_state(str(directory / "direction.pt"))
        final_models["stress"].save_state(str(directory / "stress.pt"))
        result["horizons"][str(horizon)] = {
            "delta": delta,
            "lookback_seconds": settings.lob_deeplob_lookback_seconds,
            "walk_forward_direction": direction_folds,
            "walk_forward_stress": stress_folds,
            "stress_thresholds": final_thresholds.as_dict(),
            "stress_positive_rate_train": train_positive_rate,
            "stress_positive_rate_by_month_development": monthly_positive_rates(stress_labels.loc[times < dev_cutoff], times.loc[times < dev_cutoff]),
            "stress_positive_rate_by_month_test": monthly_positive_rates(stress_labels.loc[stress_test], times.loc[stress_test]),
            "final_test": task_results,
            "artifact_directory": str(directory.relative_to(REPOSITORY_ROOT)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10))
    parser.add_argument("--allow-repeat-final-test", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    symbols = (args.symbol,) if args.symbol else settings.symbols
    horizons = (args.horizon,) if args.horizon else settings.lob_classification_horizons_seconds
    summaries = {}
    for symbol in symbols:
        output_path = REPOSITORY_ROOT / "outputs" / "tables" / symbol / "deeplob_metrics.json"
        if output_path.is_file() and not args.allow_repeat_final_test:
            raise SystemExit(f"DeepLOB final-test output already exists for {symbol}; use --allow-repeat-final-test only for an intentional re-score")
        summaries[symbol] = run_symbol(symbol, horizons, settings, args.allow_repeat_final_test)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries[symbol], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{symbol}: wrote DeepLOB walk-forward and final-test metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
