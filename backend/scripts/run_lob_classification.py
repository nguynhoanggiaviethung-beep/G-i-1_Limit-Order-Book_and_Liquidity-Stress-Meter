"""Walk-forward train and final-test the LOB direction and stress classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import classification_metrics, expanding_week_folds, paired_classification_tests
from app.labels.direction import fit_direction_delta, make_direction_labels
from app.labels.stress import (
    fit_primary_stress_thresholds,
    make_stress_labels,
    monthly_positive_rates,
)
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, make_classifiers


DIRECTION_CLASSES = ("down", "flat", "up")
STRESS_CLASSES = (0, 1)


def _load_features(symbol: str) -> pd.DataFrame:
    root = load_settings().data_dir / "features"
    paths = sorted(root.rglob("features_1s.parquet"))
    paths = [path for path in paths if f"symbol={symbol}" in path.parts]
    if not paths:
        raise FileNotFoundError(f"No feature partitions for {symbol} under {root}; run build_lob_classification_dataset.py first")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
    return frame


def _align_probabilities(model, x: pd.DataFrame, class_order: tuple[object, ...]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(list(model.named_steps.values())[-1].classes_)
    aligned = np.zeros((len(x), len(class_order)), dtype=float)
    positions = {label: index for index, label in enumerate(class_order)}
    for column, label in enumerate(classes):
        if label in positions and column < raw.shape[1]:
            aligned[:, positions[label]] = raw[:, column]
    return aligned


def _threshold_for_stress(probabilities: np.ndarray, truth: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -2.0
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = (probabilities >= threshold).astype(int)
        score = matthews_corrcoef(truth, prediction) if len(np.unique(prediction)) > 1 else 0.0
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def _direction_persistence_predictions(frame: pd.DataFrame, delta: float) -> np.ndarray:
    """Repeat the latest valid one-second price direction as a simple baseline."""
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").astype("int64").to_numpy()
    mid = pd.to_numeric(frame["mid_price"], errors="coerce").to_numpy(dtype=float)
    previous_mid = np.r_[np.nan, mid[:-1]]
    contiguous = np.r_[False, np.diff(times) == 1_000_000_000]
    valid = frame.get("book_valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool).to_numpy()
    valid_previous = np.r_[False, valid[:-1]]
    change = mid - previous_mid
    predictions = np.full(len(frame), "flat", dtype=object)
    predictions[change > delta] = "up"
    predictions[change < -delta] = "down"
    predictions[~(contiguous & valid & valid_previous & np.isfinite(change))] = "flat"
    return predictions


def _current_stress_state(frame: pd.DataFrame, thresholds) -> np.ndarray:
    spread_name = thresholds.spread_metric if thresholds.spread_metric in frame else "spread_ticks"
    spread_stress = pd.to_numeric(frame[spread_name], errors="coerce") > thresholds.spread_upper
    if thresholds.definition == "spread_only":
        return spread_stress.fillna(False).to_numpy(dtype=int)
    impact_name = "price_impact_bps" if "price_impact_bps" in frame else "impact_bps"
    depth_name = thresholds.depth_metric if thresholds.depth_metric in frame else "depth_notional_10"
    return (
        spread_stress
        | (pd.to_numeric(frame[impact_name], errors="coerce") > thresholds.impact_upper)
        | (pd.to_numeric(frame[depth_name], errors="coerce") < thresholds.depth_lower)
        | (pd.to_numeric(frame["rv_60s"], errors="coerce") > thresholds.rv_60_upper)
    ).fillna(False).to_numpy(dtype=int)


def _stress_threshold_rows(frame: pd.DataFrame, time_mask: pd.Series, definition: str = "composite") -> pd.DataFrame:
    valid = time_mask.copy()
    if "book_valid" in frame:
        book_valid = frame["book_valid"]
        if book_valid.dtype == object:
            book_valid = book_valid.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        else:
            book_valid = book_valid.fillna(False).astype(bool)
        valid &= book_valid
    spread_name = next((name for name in ("spread_ticks", "absolute_spread", "relative_spread_bps") if name in frame), None)
    if definition == "spread_only":
        if spread_name is None:
            raise ValueError("Spread-only stress thresholds require a spread feature")
        return frame.loc[valid & pd.to_numeric(frame[spread_name], errors="coerce").notna()]
    depth_name = next((name for name in ("depth_notional_10", "depth_qty_10") if name in frame), None)
    impact_name = next((name for name in ("price_impact_bps", "impact_bps") if name in frame), None)
    for name in (spread_name, impact_name, depth_name, "rv_60s"):
        if name is None:
            continue
        if name in frame:
            valid &= pd.to_numeric(frame[name], errors="coerce").notna()
    return frame.loc[valid]


def _bounded_sample(x: pd.DataFrame, y: pd.Series, maximum: int, seed: int = 42):
    if maximum < 1:
        raise ValueError("max_train_rows must be a positive integer")
    if len(y) <= maximum:
        return x, y
    rng = np.random.default_rng(seed)
    counts = y.value_counts()
    quotas = {label: max(1, int(maximum * count / len(y))) for label, count in counts.items()}
    while sum(quotas.values()) > maximum:
        label = max((key for key in quotas if quotas[key] > 1), key=lambda key: quotas[key], default=None)
        if label is None:
            break
        quotas[label] -= 1
    selected = []
    for label, group in y.groupby(y):
        indices = group.index.to_numpy()
        quota = min(len(indices), quotas.get(label, 0))
        selected.extend(rng.choice(indices, size=quota, replace=False).tolist())
    if len(selected) < maximum:
        remaining = np.setdiff1d(y.index.to_numpy(), np.asarray(selected, dtype=y.index.dtype), assume_unique=False)
        fill = min(maximum - len(selected), len(remaining))
        if fill:
            selected.extend(rng.choice(remaining, size=fill, replace=False).tolist())
    selected = sorted(selected)
    return x.loc[selected], y.loc[selected]


def _runtime_profile(model, x: pd.DataFrame) -> dict[str, object]:
    sample = x.iloc[: min(len(x), 1000)]
    if sample.empty:
        latency = None
    else:
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            model.predict_proba(sample)
            timings.append((time.perf_counter() - started) / len(sample) * 1000.0)
        latency = {"rows": int(len(sample)), "repeats": len(timings), "mean_ms_per_row": float(np.mean(timings)), "p95_of_repeats_ms_per_row": float(np.quantile(timings, 0.95))}
    estimator = model
    if hasattr(estimator, "estimator"):
        estimator = estimator.estimator
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    complexity: dict[str, object] = {"estimator": type(estimator).__name__, "input_features": int(x.shape[1])}
    if hasattr(estimator, "coef_"):
        complexity["coefficient_count"] = int(np.asarray(estimator.coef_).size)
    if hasattr(estimator, "n_estimators"):
        complexity["n_estimators"] = int(estimator.n_estimators)
    if hasattr(estimator, "max_depth"):
        complexity["max_depth"] = estimator.max_depth
    if hasattr(estimator, "get_booster"):
        complexity["boosted_rounds"] = int(estimator.get_booster().num_boosted_rounds())
    if hasattr(estimator, "estimators_"):
        trees = np.asarray(estimator.estimators_, dtype=object).reshape(-1)
        complexity["tree_count"] = int(len(trees))
        complexity["tree_node_count"] = int(sum(getattr(tree, "tree_", None).node_count for tree in trees if getattr(tree, "tree_", None) is not None))
    return {"prediction_latency": latency, "model_complexity": complexity}


def _fit_candidates(task: str, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series, max_train_rows: int):
    class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
    scores: dict[str, list[dict]] = {}
    predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, estimator in make_classifiers(task).items():
        if y_train.nunique() < 2:
            continue
        try:
            x_fit, y_fit = _bounded_sample(x_train, y_train, max_train_rows)
            estimator.fit(x_fit, y_fit.to_numpy())
            y_pred = estimator.predict(x_valid)
            probabilities = _align_probabilities(estimator, x_valid, class_order)
            metrics = classification_metrics(
                y_valid.to_numpy(), y_pred, probabilities,
                labels=list(class_order), stress=(task == "stress"),
            )
            scores.setdefault(name, []).append(metrics)
            predictions.setdefault(name, []).append((y_valid.to_numpy(), probabilities))
        except Exception as exc:
            print(f"skip {task}/{name}: {type(exc).__name__}: {exc}")
    return scores, predictions


def _select_model(task: str, scores: dict[str, list[dict]]) -> str:
    if not scores:
        raise RuntimeError(f"No {task} model completed any walk-forward fold")
    if task == "direction":
        return max(scores, key=lambda name: np.nanmean([row["macro_f1"] for row in scores[name]]))
    return max(scores, key=lambda name: np.nanmean([
        row["pr_auc"] if row.get("pr_auc") is not None else -1.0 for row in scores[name]
    ]))


def _fit_final(task: str, name: str, x: pd.DataFrame, y: pd.Series, max_train_rows: int):
    estimator = make_classifiers(task)[name]
    x_fit, y_fit = _bounded_sample(x, y, max_train_rows)
    estimator.fit(x_fit, y_fit.to_numpy())
    return estimator


def _collect_selected_oos(task: str, model_name: str, frame: pd.DataFrame, times: pd.Series, horizon: int, delta: float, embargo_seconds: int, max_train_rows: int, folds: tuple[str, ...], stress_quantile: float, stress_depth_quantile: float, stress_definition: str = "spread_only"):
    class_order = DIRECTION_CLASSES if task == "direction" else STRESS_CLASSES
    truth_parts, probability_parts = [], []
    for week in folds:
        start = pd.Timestamp(f"{week}T00:00:00Z")
        end = start + pd.Timedelta(days=7)
        cutoff = start - pd.Timedelta(seconds=embargo_seconds)
        if task == "direction":
            labels = make_direction_labels(frame, horizon, delta)
        else:
            thresholds = fit_primary_stress_thresholds(
                _stress_threshold_rows(frame, times < cutoff, stress_definition),
                stress_definition, stress_quantile, stress_depth_quantile,
            )
            labels = make_stress_labels(frame, horizon, thresholds)
        train_mask = (times < cutoff) & labels.notna()
        valid_mask = (times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & labels.notna()
        y_train, y_valid = labels.loc[train_mask], labels.loc[valid_mask]
        if y_train.nunique() < 2 or y_valid.empty:
            continue
        estimator = _fit_final(task, model_name, frame.loc[train_mask, LOB_CLASSIFICATION_FEATURES], y_train.astype(int) if task == "stress" else y_train, max_train_rows)
        x_valid = frame.loc[valid_mask, LOB_CLASSIFICATION_FEATURES]
        truth_parts.append(y_valid.to_numpy())
        probability_parts.append(_align_probabilities(estimator, x_valid, class_order))
    if not truth_parts:
        return np.asarray([]), np.empty((0, len(class_order)))
    return np.concatenate(truth_parts), np.concatenate(probability_parts)


def _fit_isotonic(truth: np.ndarray, probabilities: np.ndarray, class_order: tuple[object, ...]):
    calibrators = []
    for index, label in enumerate(class_order):
        observed = (truth == label).astype(int)
        if len(np.unique(observed)) < 2:
            calibrators.append(None)
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities[:, index], observed)
        calibrators.append(calibrator)
    return calibrators


def _fit_stress_isotonic(truth: np.ndarray, probabilities: np.ndarray):
    """Calibrate the positive stress probability directly for API/evaluation parity."""
    if not len(truth) or len(np.unique(truth == 1)) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities[:, 1], (truth == 1).astype(int))
    return calibrator


def _apply_isotonic(probabilities: np.ndarray, calibrators) -> np.ndarray:
    output = probabilities.copy()
    if calibrators:
        for index, calibrator in enumerate(calibrators):
            if calibrator is not None and index < output.shape[1]:
                output[:, index] = calibrator.predict(output[:, index])
    totals = output.sum(axis=1, keepdims=True)
    return np.divide(output, totals, out=probabilities.copy(), where=totals > 0)


def _apply_stress_isotonic(probability: np.ndarray, calibrator) -> np.ndarray:
    return calibrator.predict(probability) if calibrator is not None else probability.copy()


def run_symbol(symbol: str, horizons: tuple[int, ...], embargo_seconds: int, max_train_rows: int, settings) -> dict:
    frame = _load_features(symbol)
    if "depth_qty_10" not in frame:
        frame["depth_qty_10"] = pd.to_numeric(frame.get("bid_depth_qty_10"), errors="coerce") + pd.to_numeric(frame.get("ask_depth_qty_10"), errors="coerce")
    times = pd.to_datetime(frame["timestamp_utc"], utc=True)
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    test_end = pd.Timestamp(settings.lob_untouched_test_end_utc) + pd.Timedelta(seconds=1)
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    if not folds:
        raise RuntimeError("No full weekly walk-forward validation folds are configured")
    dev_cutoff = test_start - pd.Timedelta(seconds=embargo_seconds)
    first_cutoff = pd.Timestamp(f"{folds[0]}T00:00:00Z") - pd.Timedelta(seconds=embargo_seconds)
    first_train = frame.loc[times < first_cutoff].copy()
    if first_train.empty:
        raise RuntimeError(f"Insufficient data before validation week {folds[0]} for the first training fold")

    result: dict[str, object] = {"symbol": symbol, "data_rows": int(len(frame)), "horizons": {}}
    test_columns = ["timestamp_utc", "mid_price", *LOB_CLASSIFICATION_FEATURES]
    test_slice = frame.loc[(times >= test_start) & (times < test_end), test_columns]
    if test_slice.empty:
        raise RuntimeError(
            f"No data exists in the frozen final-test week {settings.lob_untouched_test_start_utc[:10]}"
            f" through {settings.lob_untouched_test_end_utc[:10]}; collect the configured capture window first"
        )
    result["final_test_scored_once"] = True
    result["final_test_data_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(test_slice, index=False).values.tobytes()
    ).hexdigest()
    for horizon in horizons:
        delta = fit_direction_delta(first_train, horizon, quantile=settings.lob_direction_delta_quantile)
        direction_labels = make_direction_labels(frame, horizon, delta)
        direction_scores: dict[str, list[dict]] = {}
        stress_scores: dict[str, list[dict]] = {}
        stress_threshold_history = []
        direction_fold_metrics, stress_fold_metrics = [], []
        delta_sensitivity_metrics, level_sensitivity_metrics = [], []
        delta_quantiles = sorted({max(0.1, settings.lob_direction_delta_quantile - 0.1), settings.lob_direction_delta_quantile, min(0.9, settings.lob_direction_delta_quantile + 0.1)})

        for week in folds:
            validation_start = pd.Timestamp(f"{week}T00:00:00Z")
            validation_end = validation_start + pd.Timedelta(days=7)
            train_end = validation_start - pd.Timedelta(seconds=embargo_seconds)
            train_mask = (times < train_end) & direction_labels.notna()
            valid_mask = (
                (times >= validation_start)
                & (times < validation_end - pd.Timedelta(seconds=horizon))
                & direction_labels.notna()
            )
            stress_thresholds = fit_primary_stress_thresholds(
                _stress_threshold_rows(frame, times < train_end, settings.lob_stress_definition),
                settings.lob_stress_definition,
                settings.lob_stress_quantile,
                settings.lob_stress_depth_quantile,
            )
            stress_threshold_history.append({"validation_week_utc": week, **stress_thresholds.as_dict()})
            stress_labels = make_stress_labels(frame, horizon, stress_thresholds)
            stress_train_mask = (times < train_end) & stress_labels.notna()
            stress_valid_mask = (
                (times >= validation_start)
                & (times < validation_end - pd.Timedelta(seconds=horizon))
                & stress_labels.notna()
            )
            x_direction = frame.loc[train_mask, LOB_CLASSIFICATION_FEATURES]
            xv_direction = frame.loc[valid_mask, LOB_CLASSIFICATION_FEATURES]
            y_direction = direction_labels.loc[train_mask]
            yv_direction = direction_labels.loc[valid_mask]
            dscores, _ = _fit_candidates("direction", x_direction, y_direction, xv_direction, yv_direction, max_train_rows)
            if not y_direction.empty and not yv_direction.empty:
                prior = y_direction.value_counts(normalize=True)
                majority = prior.index[0]
                prior_probability = np.asarray([prior.get(label, 0.0) for label in DIRECTION_CLASSES])
                baseline_probability = np.tile(prior_probability, (len(yv_direction), 1))
                baseline_metrics = classification_metrics(
                    yv_direction.to_numpy(), np.repeat(majority, len(yv_direction)), baseline_probability,
                    labels=list(DIRECTION_CLASSES), stress=False,
                )
                direction_fold_metrics.append({"week_utc": week, "model": "class_frequency", **baseline_metrics})
                persistence_prediction = _direction_persistence_predictions(frame.loc[valid_mask], delta)
                persistence_probability = np.column_stack([
                    persistence_prediction == label for label in DIRECTION_CLASSES
                ]).astype(float)
                persistence_metrics = classification_metrics(
                    yv_direction.to_numpy(), persistence_prediction, persistence_probability,
                    labels=list(DIRECTION_CLASSES), stress=False,
                )
                direction_fold_metrics.append({"week_utc": week, "model": "direction_persistence", **persistence_metrics})
            for name, values in dscores.items():
                direction_scores.setdefault(name, []).extend(values)
                direction_fold_metrics.append({"week_utc": week, "model": name, **values[-1]})
            obi_only = ("obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20")
            obi_plus_ofi = (*obi_only, "ofi_1s", "ofi_5s", "ofi_10s")
            for comparison_name, columns in (("logistic_obi_only", obi_only), ("logistic_obi_plus_ofi", obi_plus_ofi)):
                if not y_direction.empty and not yv_direction.empty:
                    comparator = make_classifiers("direction")["logistic"]
                    comparison_x, comparison_y = _bounded_sample(x_direction.loc[:, columns], y_direction, max_train_rows)
                    comparator.fit(comparison_x, comparison_y.to_numpy())
                    comparison_prediction = comparator.predict(xv_direction.loc[:, columns])
                    comparison_probabilities = _align_probabilities(comparator, xv_direction.loc[:, columns], DIRECTION_CLASSES)
                    comparison_metrics = classification_metrics(
                        yv_direction.to_numpy(), comparison_prediction, comparison_probabilities,
                        labels=list(DIRECTION_CLASSES), stress=False,
                    )
                    direction_fold_metrics.append({"week_utc": week, "model": comparison_name, **comparison_metrics})

            # δ sensitivity is fitted from the first training segment only;
            # each variant then uses the same expanding walk-forward folds.
            for quantile in delta_quantiles:
                delta_variant = fit_direction_delta(first_train, horizon, quantile=quantile)
                variant_labels = make_direction_labels(frame, horizon, delta_variant)
                variant_train = (times < train_end) & variant_labels.notna()
                variant_valid = (times >= validation_start) & (times < validation_end - pd.Timedelta(seconds=horizon)) & variant_labels.notna()
                variant_y = variant_labels.loc[variant_train]
                variant_yv = variant_labels.loc[variant_valid]
                if variant_y.nunique() < 2 or variant_yv.empty:
                    continue
                comparator = make_classifiers("direction")["logistic"]
                comparison_x, comparison_y = _bounded_sample(frame.loc[variant_train, LOB_CLASSIFICATION_FEATURES], variant_y, max_train_rows)
                comparator.fit(comparison_x, comparison_y.to_numpy())
                xv = frame.loc[variant_valid, LOB_CLASSIFICATION_FEATURES]
                pred = comparator.predict(xv)
                prob = _align_probabilities(comparator, xv, DIRECTION_CLASSES)
                delta_sensitivity_metrics.append({"week_utc": week, "delta_quantile": float(quantile), "delta": float(delta_variant), **classification_metrics(variant_yv.to_numpy(), pred, prob, labels=list(DIRECTION_CLASSES))})

            # L sensitivity keeps the classifier fixed and varies the static
            # depth imbalance lookback, for every requested forecast horizon.
            for depth in (1, 5, 10, 20):
                column = f"obi_qty_{depth}"
                if y_direction.nunique() < 2 or yv_direction.empty:
                    continue
                comparator = make_classifiers("direction")["logistic"]
                comparison_x, comparison_y = _bounded_sample(x_direction.loc[:, [column]], y_direction, max_train_rows)
                comparator.fit(comparison_x, comparison_y.to_numpy())
                xv = xv_direction.loc[:, [column]]
                pred = comparator.predict(xv)
                prob = _align_probabilities(comparator, xv, DIRECTION_CLASSES)
                level_sensitivity_metrics.append({"week_utc": week, "depth_levels_L": depth, **classification_metrics(yv_direction.to_numpy(), pred, prob, labels=list(DIRECTION_CLASSES))})

            x_stress = frame.loc[stress_train_mask, LOB_CLASSIFICATION_FEATURES]
            xv_stress = frame.loc[stress_valid_mask, LOB_CLASSIFICATION_FEATURES]
            y_stress = stress_labels.loc[stress_train_mask].astype(int)
            yv_stress = stress_labels.loc[stress_valid_mask].astype(int)
            sscores, _ = _fit_candidates("stress", x_stress, y_stress, xv_stress, yv_stress, max_train_rows)
            if not y_stress.empty and not yv_stress.empty:
                persistent = _current_stress_state(frame.loc[stress_valid_mask], stress_thresholds)
                persistence_metrics = classification_metrics(
                    yv_stress.to_numpy(), persistent,
                    np.column_stack([1 - persistent, persistent]),
                    labels=list(STRESS_CLASSES), stress=True,
                )
                stress_fold_metrics.append({"week_utc": week, "model": "stress_persistence", **persistence_metrics})
            for name, values in sscores.items():
                stress_scores.setdefault(name, []).extend(values)
                stress_fold_metrics.append({"week_utc": week, "model": name, **values[-1]})

        direction_name = _select_model("direction", direction_scores)
        stress_name = _select_model("stress", stress_scores)
        final_stress_thresholds = fit_primary_stress_thresholds(
            _stress_threshold_rows(frame, times < dev_cutoff, settings.lob_stress_definition),
            settings.lob_stress_definition,
            settings.lob_stress_quantile,
            settings.lob_stress_depth_quantile,
        )
        final_stress_labels = make_stress_labels(frame, horizon, final_stress_thresholds)
        direction_train_mask = (times < dev_cutoff) & direction_labels.notna()
        stress_train_mask = (times < dev_cutoff) & final_stress_labels.notna()
        test_mask = (times >= test_start) & (times < test_end - pd.Timedelta(seconds=horizon)) & direction_labels.notna()
        stress_test_mask = (times >= test_start) & (times < test_end - pd.Timedelta(seconds=horizon)) & final_stress_labels.notna()

        x_train = frame.loc[direction_train_mask, LOB_CLASSIFICATION_FEATURES]
        y_train = direction_labels.loc[direction_train_mask]
        x_test = frame.loc[test_mask, LOB_CLASSIFICATION_FEATURES]
        y_test = direction_labels.loc[test_mask]
        direction_model = _fit_final("direction", direction_name, x_train, y_train, max_train_rows)
        direction_prediction = direction_model.predict(x_test)
        direction_probability_raw = _align_probabilities(direction_model, x_test, DIRECTION_CLASSES)
        direction_oos_truth, direction_oos_probability = _collect_selected_oos(
            "direction", direction_name, frame, times, horizon, delta, embargo_seconds,
            max_train_rows, folds, settings.lob_stress_quantile, settings.lob_stress_depth_quantile,
            settings.lob_stress_definition,
        )
        direction_calibrators = _fit_isotonic(direction_oos_truth, direction_oos_probability, DIRECTION_CLASSES)
        direction_probability = _apply_isotonic(direction_probability_raw, direction_calibrators)
        direction_test_metrics_raw = classification_metrics(
            y_test.to_numpy(), direction_prediction, direction_probability_raw,
            labels=list(DIRECTION_CLASSES), stress=False,
        )
        direction_prediction_calibrated = np.asarray(DIRECTION_CLASSES)[np.argmax(direction_probability, axis=1)]
        direction_test_metrics = classification_metrics(
            y_test.to_numpy(), direction_prediction_calibrated, direction_probability,
            labels=list(DIRECTION_CLASSES), stress=False,
        )
        direction_persistence_all = _direction_persistence_predictions(frame, delta)
        direction_persistence_test = direction_persistence_all[test_mask.to_numpy()]
        direction_persistence_probability = np.column_stack([
            direction_persistence_test == label for label in DIRECTION_CLASSES
        ]).astype(float)
        direction_pairwise = paired_classification_tests(
            y_test.to_numpy(), direction_prediction_calibrated, direction_persistence_test,
            direction_probability, direction_persistence_probability,
            stress=False, horizon_seconds=horizon, timestamps=times.loc[test_mask],
        )

        xs_train = frame.loc[stress_train_mask, LOB_CLASSIFICATION_FEATURES]
        ys_train = final_stress_labels.loc[stress_train_mask].astype(int)
        xs_test = frame.loc[stress_test_mask, LOB_CLASSIFICATION_FEATURES]
        ys_test = final_stress_labels.loc[stress_test_mask].astype(int)
        stress_train_positive_rate = float(ys_train.mean()) if len(ys_train) else float("nan")
        prevalence_min = settings.lob_stress_min_positive_rate
        prevalence_max = settings.lob_stress_max_positive_rate
        if not np.isfinite(stress_train_positive_rate) or not prevalence_min <= stress_train_positive_rate <= prevalence_max:
            raise RuntimeError(
                f"{symbol} h={horizon}: development stress-label prevalence {stress_train_positive_rate:.4f} "
                f"is outside [{prevalence_min:.2%}, {prevalence_max:.2%}]; revise the train-only label definition before scoring the frozen test week"
            )
        stress_model = _fit_final("stress", stress_name, xs_train, ys_train, max_train_rows)
        stress_probability_raw = _align_probabilities(stress_model, xs_test, STRESS_CLASSES)
        stress_oos_truth, stress_oos_probability = _collect_selected_oos(
            "stress", stress_name, frame, times, horizon, delta, embargo_seconds,
            max_train_rows, folds, settings.lob_stress_quantile, settings.lob_stress_depth_quantile,
            settings.lob_stress_definition,
        )
        stress_calibrator = _fit_stress_isotonic(stress_oos_truth, stress_oos_probability)
        stress_threshold_raw = _threshold_for_stress(stress_oos_probability[:, 1], stress_oos_truth) if len(stress_oos_truth) else 0.5
        stress_oos_probability_calibrated = _apply_stress_isotonic(stress_oos_probability[:, 1], stress_calibrator)
        stress_threshold = _threshold_for_stress(stress_oos_probability_calibrated, stress_oos_truth) if len(stress_oos_truth) else 0.5
        stress_probability = _apply_stress_isotonic(stress_probability_raw[:, 1], stress_calibrator)
        stress_prediction = (stress_probability >= stress_threshold).astype(int)
        stress_test_metrics_raw = classification_metrics(
            ys_test.to_numpy(), (stress_probability_raw[:, 1] >= stress_threshold_raw).astype(int),
            stress_probability_raw, labels=list(STRESS_CLASSES), stress=True,
        )
        stress_test_metrics = classification_metrics(
            ys_test.to_numpy(), stress_prediction,
            np.column_stack([1.0 - stress_probability, stress_probability]),
            labels=list(STRESS_CLASSES), stress=True,
        )
        stress_persistence_test = _current_stress_state(frame.loc[stress_test_mask], final_stress_thresholds)
        stress_persistence_probability = np.column_stack([1 - stress_persistence_test, stress_persistence_test]).astype(float)
        stress_pairwise = paired_classification_tests(
            ys_test.to_numpy(), stress_prediction, stress_persistence_test,
            np.column_stack([1.0 - stress_probability, stress_probability]), stress_persistence_probability,
            stress=True, horizon_seconds=horizon, timestamps=times.loc[stress_test_mask],
        )
        spread_sensitivity_thresholds = fit_primary_stress_thresholds(
            _stress_threshold_rows(frame, times < dev_cutoff, "spread_only"),
            "spread_only",
            settings.lob_spread_sensitivity_quantile,
            settings.lob_stress_depth_quantile,
        )
        spread_sensitivity_labels = make_stress_labels(frame, horizon, spread_sensitivity_thresholds)
        spread_sensitivity_train_mask = (times < dev_cutoff) & spread_sensitivity_labels.notna()
        spread_sensitivity_test_mask = (times >= test_start) & (times < test_end - pd.Timedelta(seconds=horizon)) & spread_sensitivity_labels.notna()
        spread_sensitivity_train_y = spread_sensitivity_labels.loc[spread_sensitivity_train_mask].astype(int)
        spread_sensitivity_test_y = spread_sensitivity_labels.loc[spread_sensitivity_test_mask].astype(int)
        spread_sensitivity: dict[str, object] = {
            "status": "unavailable",
            "definition": f"spread_only_train_q{settings.lob_spread_sensitivity_quantile * 100:g}_sensitivity",
            "thresholds_train": spread_sensitivity_thresholds.as_dict(),
            "positive_rate_by_month_development": monthly_positive_rates(
                spread_sensitivity_labels.loc[times < dev_cutoff], times.loc[times < dev_cutoff]
            ),
            "positive_rate_by_month_test": monthly_positive_rates(
                spread_sensitivity_labels.loc[spread_sensitivity_test_mask], times.loc[spread_sensitivity_test_mask]
            ),
        }
        if spread_sensitivity_train_y.nunique() == 2 and not spread_sensitivity_test_y.empty:
            spread_sensitivity_model = _fit_final(
                "stress", stress_name,
                frame.loc[spread_sensitivity_train_mask, LOB_CLASSIFICATION_FEATURES],
                spread_sensitivity_train_y,
                max_train_rows,
            )
            spread_sensitivity_x_test = frame.loc[spread_sensitivity_test_mask, LOB_CLASSIFICATION_FEATURES]
            spread_sensitivity_probability = _align_probabilities(spread_sensitivity_model, spread_sensitivity_x_test, STRESS_CLASSES)
            spread_sensitivity_prediction = (spread_sensitivity_probability[:, 1] >= 0.5).astype(int)
            spread_sensitivity.update({
                "status": "scored",
                "test_metrics": classification_metrics(
                    spread_sensitivity_test_y.to_numpy(), spread_sensitivity_prediction,
                    spread_sensitivity_probability, labels=list(STRESS_CLASSES), stress=True,
                ),
            })
        elif spread_sensitivity_train_y.nunique() < 2:
            spread_sensitivity["reason"] = "Spread-only train thresholds produced a single class; sensitivity label is degenerate"
        else:
            spread_sensitivity["reason"] = "No valid untouched-test spread-only labels are available"
        regime_training_rv = pd.to_numeric(frame.loc[times < dev_cutoff, "rv_300s"], errors="coerce").dropna()
        if len(regime_training_rv) and regime_training_rv.quantile(1 / 3) < regime_training_rv.quantile(2 / 3):
            regime_cutoffs = [float(regime_training_rv.quantile(1 / 3)), float(regime_training_rv.quantile(2 / 3))]
            test_regimes = pd.cut(
                pd.to_numeric(frame.loc[test_mask, "rv_300s"], errors="coerce"),
                bins=[-np.inf, regime_cutoffs[0], regime_cutoffs[1], np.inf],
                labels=["low_volatility", "medium_volatility", "high_volatility"],
                include_lowest=True,
            )
        else:
            regime_cutoffs = None
            test_regimes = pd.Series([None] * int(test_mask.sum()), index=frame.index[test_mask])
        direction_by_regime, stress_by_regime = {}, {}
        test_positions = np.flatnonzero(test_mask.to_numpy())
        stress_position_lookup = {index: position for position, index in enumerate(np.flatnonzero(stress_test_mask.to_numpy()))}
        for regime in ("low_volatility", "medium_volatility", "high_volatility"):
            selected_direction = np.asarray(test_regimes == regime)
            if selected_direction.any():
                direction_by_regime[regime] = classification_metrics(
                    y_test.to_numpy()[selected_direction],
                    direction_prediction_calibrated[selected_direction],
                    direction_probability[selected_direction],
                    labels=list(DIRECTION_CLASSES), stress=False,
                )
            selected_stress_rows = [
                (index, stress_position_lookup[index]) for index, selected in zip(test_positions, selected_direction)
                if selected and index in stress_position_lookup
            ]
            if selected_stress_rows:
                selected_indices, selected_positions = zip(*selected_stress_rows)
                stress_targets = final_stress_labels.loc[list(selected_indices)].astype(int).to_numpy()
                stress_probs = stress_probability[list(selected_positions)]
                stress_preds = (stress_probs >= stress_threshold).astype(int)
                stress_by_regime[regime] = classification_metrics(
                    stress_targets, stress_preds,
                    np.column_stack([1 - stress_probs, stress_probs]),
                    labels=list(STRESS_CLASSES), stress=True,
                )
        direction_counts = y_train.value_counts(normalize=True).to_dict()
        stress_counts = ys_train.value_counts(normalize=True).to_dict()
        artifact = {
            "symbol": symbol,
            "horizon_seconds": horizon,
            "model_version": f"{direction_name}-{stress_name}-h{horizon}-{test_start.date().isoformat()}",
            "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "feature_columns": list(LOB_CLASSIFICATION_FEATURES),
            "direction_model": direction_model,
            "stress_model": stress_model,
            "direction_class_order": list(DIRECTION_CLASSES),
            "direction_calibrators": direction_calibrators,
            "stress_calibrator": stress_calibrator,
            "direction_delta": delta,
            "thresholds": final_stress_thresholds.as_dict(),
            "stress_probability_threshold": stress_threshold,
            "stress_probability_threshold_raw": stress_threshold_raw,
            "direction_calibration_method": "one-vs-rest isotonic fitted on development walk-forward predictions",
            "stress_calibration_method": "isotonic fitted on development walk-forward predictions",
            "metrics": {"direction": direction_test_metrics, "stress": stress_test_metrics},
        }
        artifact_path = REPOSITORY_ROOT / "backend" / "artifacts" / "meter" / symbol / f"horizon_{horizon}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path)
        direction_runtime = _runtime_profile(direction_model, x_test)
        stress_runtime = _runtime_profile(stress_model, xs_test)
        result["horizons"][str(horizon)] = {
            "delta": delta,
            "direction_model": direction_name,
            "stress_model": stress_name,
            "walk_forward_direction": direction_fold_metrics,
            "walk_forward_stress": stress_fold_metrics,
            "direction_delta_sensitivity": delta_sensitivity_metrics,
            "direction_L_sensitivity": level_sensitivity_metrics,
            "stress_thresholds_by_fold": stress_threshold_history,
            "final_train_stress_thresholds": final_stress_thresholds.as_dict(),
            "stress_positive_rate_train": final_stress_thresholds.positive_rate_train,
            "stress_positive_rate_forward_labels_train": stress_train_positive_rate,
            "stress_positive_rate_by_month_development": monthly_positive_rates(
                final_stress_labels.loc[times < dev_cutoff], times.loc[times < dev_cutoff]
            ),
            "stress_positive_rate_by_month_test": monthly_positive_rates(
                final_stress_labels.loc[stress_test_mask], times.loc[stress_test_mask]
            ),
            "stress_prevalence_check": "within_configured_range",
            "direction_class_distribution_train": {str(key): float(value) for key, value in direction_counts.items()},
            "stress_class_distribution_train": {str(key): float(value) for key, value in stress_counts.items()},
            "direction_test": direction_test_metrics,
            "direction_test_raw": direction_test_metrics_raw,
            "direction_paired_tests_vs_persistence": direction_pairwise,
            "stress_test": stress_test_metrics,
            "stress_test_raw": stress_test_metrics_raw,
            "stress_paired_tests_vs_persistence": stress_pairwise,
            "stress_spread_only_sensitivity": spread_sensitivity,
            "stress_decision_threshold": stress_threshold,
            "stress_decision_threshold_raw": stress_threshold_raw,
            "regime_cutoffs_rv_300s_train_terciles": regime_cutoffs,
            "direction_test_by_regime": direction_by_regime,
            "stress_test_by_regime": stress_by_regime,
            "test_rows_direction": int(len(y_test)),
            "test_rows_stress": int(len(ys_test)),
            "artifact": str(artifact_path.relative_to(REPOSITORY_ROOT)),
            "artifact_bytes": int(artifact_path.stat().st_size),
            "direction_runtime_profile": direction_runtime,
            "stress_runtime_profile": stress_runtime,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10))
    parser.add_argument("--embargo-seconds", type=int)
    parser.add_argument("--max-train-rows", type=int, default=250000, help="Deterministic class-stratified cap for each model fit")
    parser.add_argument("--allow-repeat-final-test", action="store_true", help="Explicitly allow scoring the frozen final-test week again after a prior run")
    args = parser.parse_args()
    settings = load_settings()
    embargo_seconds = args.embargo_seconds if args.embargo_seconds is not None else settings.lob_embargo_seconds
    if args.max_train_rows < len(DIRECTION_CLASSES) or embargo_seconds < 1:
        raise SystemExit("--max-train-rows must be at least 3 and --embargo-seconds must be positive")
    symbols = (args.symbol,) if args.symbol else settings.symbols
    horizons = (args.horizon,) if args.horizon else settings.lob_classification_horizons_seconds
    minimum_embargo = max(max(horizons), settings.lob_feature_lookback_seconds, settings.lob_deeplob_lookback_seconds)
    if embargo_seconds < minimum_embargo:
        raise SystemExit(f"Embargo must be at least {minimum_embargo} seconds for the configured horizon and feature/sequence windows")
    prior_outputs = [REPOSITORY_ROOT / "outputs" / "tables" / symbol / "classification_metrics.json" for symbol in symbols]
    already_scored = [path for path in prior_outputs if path.is_file()]
    if already_scored and not args.allow_repeat_final_test:
        paths = ", ".join(str(path.relative_to(REPOSITORY_ROOT)) for path in already_scored)
        raise SystemExit(f"Final-test output already exists for {paths}. Use --allow-repeat-final-test only when an intentional re-score is approved.")
    output = {}
    for symbol in symbols:
        output[symbol] = run_symbol(symbol, horizons, embargo_seconds, args.max_train_rows, settings)
    result_dir = REPOSITORY_ROOT / "outputs" / "tables"
    result_dir.mkdir(parents=True, exist_ok=True)
    for symbol, payload in output.items():
        symbol_dir = result_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        destination = symbol_dir / "classification_metrics.json"
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{symbol}: wrote walk-forward and final-test metrics to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

