"""Walk-forward train and final-test the LOB direction and stress classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import matthews_corrcoef
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.evaluation.lob_classification import classification_metrics, expanding_week_folds, paired_classification_tests
from app.labels.direction import DIRECTION_LABEL_SCHEMA_VERSION, fit_direction_delta, make_direction_labels
from app.labels.stress import (
    fit_primary_stress_thresholds,
    make_stress_labels,
    monthly_positive_rates,
    STRESS_LABEL_SCHEMA_VERSION,
)
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION, make_classifiers


DIRECTION_CLASSES = ("down", "flat", "up")
STRESS_CLASSES = (0, 1)
# Columns that define labels, thresholds, baselines, or regimes stay float64; every other
# model feature is stored as float32 so six months of one-second rows fit in memory.
LABEL_INPUT_COLUMNS = (
    "mid_price", "spread_ticks", "absolute_spread", "relative_spread_bps", "price_impact_bps",
    "depth_notional_10", "depth_qty_10", "rv_60s", "rv_300s",
)
RUN_COLUMNS = tuple(dict.fromkeys((
    "timestamp_utc", "symbol", "book_valid", "book_source", *LABEL_INPUT_COLUMNS, *LOB_CLASSIFICATION_FEATURES,
)))
STRESS_THRESHOLD_COLUMNS = (
    "timestamp_utc", "book_valid", "spread_ticks", "absolute_spread", "relative_spread_bps",
    "price_impact_bps", "depth_notional_10", "depth_qty_10", "rv_60s",
)
OBI_ONLY_COLUMNS = ("obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20")
OBI_PLUS_OFI_COLUMNS = (*OBI_ONLY_COLUMNS, "ofi_1s", "ofi_5s", "ofi_10s")
FINAL_TEST_OPENED_MARKER = "final_test_opened.json"
logger = logging.getLogger(__name__)


def _study_provenance(
    frame: pd.DataFrame,
    settings,
    horizons: tuple[int, ...],
    settings_path: Path,
    study_label: str,
) -> dict[str, object]:
    settings_hash = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    manifest_path = REPOSITORY_ROOT / "outputs" / "data_quality" / "data_manifest.csv"
    dependency_versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "pyarrow"):
        try:
            dependency_versions[package] = package_version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dropna()
    return {
        "configuration_file": settings_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "configuration_sha256": settings_hash,
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None,
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions,
        "study_label": study_label,
        "feature_schema_version": LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION,
        "direction_label_schema_version": DIRECTION_LABEL_SCHEMA_VERSION,
        "stress_label_schema_version": STRESS_LABEL_SCHEMA_VERSION,
        "observed_data_start_utc": timestamps.min().isoformat() if len(timestamps) else None,
        "observed_data_end_utc": timestamps.max().isoformat() if len(timestamps) else None,
        "symbols": sorted(frame["symbol"].dropna().astype(str).str.upper().unique().tolist()) if "symbol" in frame else [],
        "horizons_seconds": list(horizons),
        "training_period_utc": {
            "start": timestamps.min().isoformat() if len(timestamps) else None,
            "end": settings.lob_first_validation_week_utc,
        },
        "validation_period_utc": {
            "start": settings.lob_first_validation_week_utc,
            "end": settings.lob_development_end_utc,
        },
        "development_end_utc": settings.lob_development_end_utc,
        "final_test_start_utc": settings.lob_untouched_test_start_utc,
        "final_test_end_utc": settings.lob_untouched_test_end_utc,
        "random_seeds": {"sklearn_estimators": 42, "classical_train_row_sampling": 42},
    }


def _partition_date(path: Path) -> str | None:
    return next((part[len("date="):] for part in path.parts if part.startswith("date=")), None)


def _read_run_columns(path: Path, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = dict.fromkeys((*RUN_COLUMNS, *extra_columns))
    frame = pd.read_parquet(path, columns=[name for name in wanted if name in available])
    for name in extra_columns:
        # Raw book levels are stored as decimal strings; parse them the way build_lob_sequences does.
        if name in frame and frame[name].dtype == object:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name in frame.columns:
        if name not in LABEL_INPUT_COLUMNS and frame[name].dtype == np.float64:
            frame[name] = frame[name].astype(np.float32)
    return frame


def _load_features(
    symbol: str,
    settings,
    end_exclusive: pd.Timestamp | None = None,
    extra_columns: tuple[str, ...] = (),
    start_inclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load the run columns (plus ``extra_columns``); with ``end_exclusive`` no partition on or after that date is opened.

    ``start_inclusive`` likewise skips partitions dated before it.
    """
    root = settings.data_dir / "features"
    paths = sorted(root.rglob("features_1s.parquet"))
    paths = [path for path in paths if f"symbol={symbol}" in path.parts]
    if end_exclusive is not None:
        last_date = end_exclusive.tz_convert("UTC").date().isoformat()
        paths = [path for path in paths if (_partition_date(path) or "9999-12-31") < last_date]
    if start_inclusive is not None:
        first_date = start_inclusive.tz_convert("UTC").date().isoformat()
        paths = [path for path in paths if (_partition_date(path) or "0000-01-01") >= first_date]
    if not paths:
        raise FileNotFoundError(f"No feature partitions for {symbol} under {root}; run build_lob_classification_dataset.py first")
    frame = pd.concat((_read_run_columns(path, extra_columns) for path in paths), ignore_index=True)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc", kind="mergesort")
    duplicate_count = int(frame.duplicated("timestamp_utc", keep="last").sum())
    if duplicate_count:
        logger.warning(
            "Resolved %d duplicate feature timestamps for %s by keeping the last input row",
            duplicate_count,
            symbol,
        )
    frame = frame.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
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


def _predict_with_probabilities(model, x: pd.DataFrame, class_order: tuple[object, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Score once: the label is the argmax of the aligned probabilities.

    Random-forest, XGBoost and multinomial Logistic ``predict`` are this argmax (first class
    wins ties); binary Logistic differs only if a probability rounds to exactly 0.5.
    This saves a second pass of the pipeline over the validation rows.
    """
    probabilities = _align_probabilities(model, x, class_order)
    labels = np.asarray(class_order)[np.argmax(probabilities, axis=1)]
    return labels, probabilities


def _threshold_for_stress(probabilities: np.ndarray, truth: np.ndarray) -> float:
    if len(probabilities) != len(truth) or not len(truth) or len(np.unique(truth)) < 2:
        raise ValueError("Stress decision threshold needs paired development predictions with both classes")
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


def _current_stress_state(frame: pd.DataFrame, thresholds) -> pd.Series:
    spread_name = thresholds.spread_metric if thresholds.spread_metric in frame else "spread_ticks"
    spread = pd.to_numeric(frame[spread_name], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(spread)
    if "book_valid" in frame:
        book_valid = frame["book_valid"]
        if book_valid.dtype == object:
            valid &= book_valid.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy()
        else:
            valid &= book_valid.fillna(False).astype(bool).to_numpy()
    states = np.full(len(frame), pd.NA, dtype=object)
    if thresholds.definition == "spread_only":
        states[valid] = (spread[valid] > thresholds.spread_upper).astype(int)
        return pd.Series(states, index=frame.index, dtype="Int64")
    impact_name = "price_impact_bps" if "price_impact_bps" in frame else "impact_bps"
    depth_name = thresholds.depth_metric if thresholds.depth_metric in frame else "depth_notional_10"
    impact = pd.to_numeric(frame[impact_name], errors="coerce").to_numpy(dtype=float)
    depth = pd.to_numeric(frame[depth_name], errors="coerce").to_numpy(dtype=float)
    rv = pd.to_numeric(frame["rv_60s"], errors="coerce").to_numpy(dtype=float)
    valid &= np.isfinite(impact) & np.isfinite(depth) & np.isfinite(rv)
    states[valid] = (
        (spread[valid] > thresholds.spread_upper)
        | (impact[valid] > thresholds.impact_upper)
        | (depth[valid] < thresholds.depth_lower)
        | (rv[valid] > thresholds.rv_60_upper)
    ).astype(int)
    return pd.Series(states, index=frame.index, dtype="Int64")


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
        finite_spread = np.isfinite(pd.to_numeric(frame[spread_name], errors="coerce").to_numpy(dtype=float))
        return frame.loc[valid & finite_spread]
    depth_name = next((name for name in ("depth_notional_10", "depth_qty_10") if name in frame), None)
    impact_name = next((name for name in ("price_impact_bps", "impact_bps") if name in frame), None)
    required = (spread_name, impact_name, depth_name, "rv_60s")
    if any(name is None or name not in frame for name in required):
        raise ValueError("Composite stress thresholds need spread, price impact, top-10 depth, and rv_60s features")
    for name in required:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(values)
    return frame.loc[valid]


def _bounded_sample(x: pd.DataFrame, y: pd.Series, maximum: int, seed: int = 42):
    if maximum < 1:
        raise ValueError("max_train_rows must be a positive integer")
    if len(y) <= maximum:
        return x, y
    rng = np.random.default_rng(seed)
    # Work on integer codes: sorted codes give groupby's label order, and counting codes
    # keeps value_counts' order, so the random draws match the label-based version.
    codes, uniques = pd.factorize(y, sort=True)
    counts = pd.Series(codes[codes >= 0]).value_counts()
    quotas = {int(code): max(1, int(maximum * count / len(y))) for code, count in counts.items()}
    while sum(quotas.values()) > maximum:
        code = max((key for key in quotas if quotas[key] > 1), key=lambda key: quotas[key], default=None)
        if code is None:
            break
        quotas[code] -= 1
    index_values = y.index.to_numpy()
    parts = []
    for code in range(len(uniques)):
        indices = index_values[codes == code]
        quota = min(len(indices), quotas.get(code, 0))
        parts.append(rng.choice(indices, size=quota, replace=False))
    selected = np.concatenate(parts) if parts else np.asarray([], dtype=index_values.dtype)
    if len(selected) < maximum:
        remaining = np.setdiff1d(index_values, selected.astype(y.index.dtype), assume_unique=False)
        fill = min(maximum - len(selected), len(remaining))
        if fill:
            selected = np.concatenate([selected, rng.choice(remaining, size=fill, replace=False)])
    selected = np.sort(selected)
    return x.loc[selected], y.loc[selected]


def _sampled_training(frame: pd.DataFrame, labels: pd.Series, mask: pd.Series, columns, max_train_rows: int):
    """Sample training origins from labels first, then slice only those feature rows.

    The class-stratified selection depends on labels alone, so this picks exactly the rows
    ``_bounded_sample`` would pick from the full slice without materializing it.
    """
    y = labels.loc[mask]
    _, y = _bounded_sample(y, y, max_train_rows)
    return frame.loc[y.index, list(columns)], y


def _with_depth_qty_10(frame: pd.DataFrame) -> pd.DataFrame:
    if "depth_qty_10" not in frame:
        frame["depth_qty_10"] = pd.to_numeric(frame.get("bid_depth_qty_10"), errors="coerce") + pd.to_numeric(frame.get("ask_depth_qty_10"), errors="coerce")
    return frame


def _threshold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, [name for name in STRESS_THRESHOLD_COLUMNS if name in frame]]


def _warm_share(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    if "book_source" not in frame or not mask.any():
        return None
    return float((frame.loc[mask, "book_source"] == "warm_start").mean())


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
            y_pred, probabilities = _predict_with_probabilities(estimator, x_valid, class_order)
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
    primary_metric = "macro_f1" if task == "direction" else "pr_auc"

    def selection_key(name: str) -> tuple[float, float]:
        rows = scores[name]
        primary = [float(row[primary_metric]) for row in rows if row.get(primary_metric) is not None and np.isfinite(row[primary_metric])]
        secondary = [float(row["mcc"]) for row in rows if row.get("mcc") is not None and np.isfinite(row["mcc"])]
        return (
            float(np.mean(primary)) if primary else -1.0,
            float(np.mean(secondary)) if secondary else -1.0,
        )

    # Primary metric selects the model; validation MCC is a predeclared tie-breaker.
    return max(scores, key=selection_key)


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
                _stress_threshold_rows(_threshold_frame(frame), times < cutoff, stress_definition),
                stress_definition, stress_quantile, stress_depth_quantile,
            )
            labels = make_stress_labels(frame, horizon, thresholds)
        train_mask = (times < cutoff) & labels.notna()
        valid_mask = (times >= start) & (times < end - pd.Timedelta(seconds=horizon)) & labels.notna()
        y_valid = labels.loc[valid_mask]
        if labels.loc[train_mask].nunique() < 2 or y_valid.empty:
            continue
        x_train, y_train = _sampled_training(frame, labels, train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
        estimator = _fit_final(task, model_name, x_train, y_train.astype(int) if task == "stress" else y_train, max_train_rows)
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


def run_symbol(
    symbol: str,
    horizons: tuple[int, ...],
    embargo_seconds: int,
    max_train_rows: int,
    settings,
    settings_path: Path,
    artifact_root: Path,
    study_label: str,
    score_final_test: bool = True,
    output_dir: Path | None = None,
) -> dict:
    """Walk-forward on development data and, unless ``score_final_test`` is False, score the untouched test.

    With ``output_dir``, an opened-marker is written there before any test partition is read, and
    per-second test predictions of every pre-specified model are saved there as Parquet.
    """
    test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
    test_end = pd.Timestamp(settings.lob_untouched_test_end_utc) + pd.Timedelta(seconds=1)
    # No partition dated on or after the untouched test start is opened here. The test is loaded
    # separately below, once models, calibration and decision thresholds are final.
    frame = _with_depth_qty_10(_load_features(symbol, settings, end_exclusive=test_start))
    frame = frame.loc[pd.to_datetime(frame["timestamp_utc"], utc=True) < test_start].reset_index(drop=True)
    test_frame: pd.DataFrame | None = None
    times = pd.to_datetime(frame["timestamp_utc"], utc=True)
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    if not folds:
        raise RuntimeError("No full weekly walk-forward validation folds are configured")
    dev_cutoff = test_start - pd.Timedelta(seconds=embargo_seconds)
    first_cutoff = pd.Timestamp(f"{folds[0]}T00:00:00Z") - pd.Timedelta(seconds=embargo_seconds)
    # δ needs only the price path and validity of the first training segment.
    first_train = frame.loc[times < first_cutoff, ["timestamp_utc", "book_valid", "mid_price"]].copy()
    if first_train.empty:
        raise RuntimeError(f"Insufficient data before validation week {folds[0]} for the first training fold")

    result: dict[str, object] = {
        "symbol": symbol,
        "data_rows": int(len(frame)),
        "development_rows": int(len(frame)),
        "max_train_rows_per_fit": int(max_train_rows),
        "reproducibility": _study_provenance(frame, settings, horizons, settings_path, study_label),
        "study_label": study_label,
        "final_test_scored": bool(score_final_test),
        "validation_weeks_utc": list(folds),
        "embargo_seconds": int(embargo_seconds),
        "model_selection_criteria": {
            "direction": {"primary": "mean_validation_macro_f1", "tie_breaker": "mean_validation_mcc"},
            "stress": {"primary": "mean_validation_pr_auc", "tie_breaker": "mean_validation_mcc"},
        },
        "horizons": {},
    }
    # Stress thresholds depend on the fold's training window, not on the horizon.
    stress_threshold_cache: dict[str, object] = {}
    for horizon in horizons:
        horizon_started = time.perf_counter()
        delta = fit_direction_delta(first_train, horizon, quantile=settings.lob_direction_delta_quantile)
        direction_labels = make_direction_labels(frame, horizon, delta)
        direction_scores: dict[str, list[dict]] = {}
        stress_scores: dict[str, list[dict]] = {}
        stress_threshold_history = []
        fold_manifest = []
        direction_fold_metrics, stress_fold_metrics = [], []
        delta_sensitivity_metrics, level_sensitivity_metrics = [], []
        delta_quantiles = sorted({max(0.1, settings.lob_direction_delta_quantile - 0.1), settings.lob_direction_delta_quantile, min(0.9, settings.lob_direction_delta_quantile + 0.1)})
        # δ variants come from the first training segment, so their labels are fold-independent.
        delta_variants = {}
        for quantile in delta_quantiles:
            delta_variant = fit_direction_delta(first_train, horizon, quantile=quantile)
            delta_variants[quantile] = (delta_variant, make_direction_labels(frame, horizon, delta_variant))

        for week in folds:
            fold_started = time.perf_counter()
            validation_start = pd.Timestamp(f"{week}T00:00:00Z")
            validation_end = validation_start + pd.Timedelta(days=7)
            train_end = validation_start - pd.Timedelta(seconds=embargo_seconds)
            train_mask = (times < train_end) & direction_labels.notna()
            valid_mask = (
                (times >= validation_start)
                & (times < validation_end - pd.Timedelta(seconds=horizon))
                & direction_labels.notna()
            )
            if week not in stress_threshold_cache:
                stress_threshold_cache[week] = fit_primary_stress_thresholds(
                    _stress_threshold_rows(_threshold_frame(frame), times < train_end, settings.lob_stress_definition),
                    settings.lob_stress_definition,
                    settings.lob_stress_quantile,
                    settings.lob_stress_depth_quantile,
                )
            stress_thresholds = stress_threshold_cache[week]
            stress_threshold_history.append({"validation_week_utc": week, **stress_thresholds.as_dict()})
            stress_labels = make_stress_labels(frame, horizon, stress_thresholds)
            stress_train_mask = (times < train_end) & stress_labels.notna()
            stress_valid_mask = (
                (times >= validation_start)
                & (times < validation_end - pd.Timedelta(seconds=horizon))
                & stress_labels.notna()
            )
            x_direction, y_direction = _sampled_training(frame, direction_labels, train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
            xv_direction = frame.loc[valid_mask, LOB_CLASSIFICATION_FEATURES]
            yv_direction = direction_labels.loc[valid_mask]
            dscores, _ = _fit_candidates("direction", x_direction, y_direction, xv_direction, yv_direction, max_train_rows)
            if not y_direction.empty and not yv_direction.empty:
                # The class-frequency baseline uses the full training prior, not the sample.
                prior = direction_labels.loc[train_mask].value_counts(normalize=True)
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
            for comparison_name, columns in (("logistic_obi_only", OBI_ONLY_COLUMNS), ("logistic_obi_plus_ofi", OBI_PLUS_OFI_COLUMNS)):
                if not y_direction.empty and not yv_direction.empty:
                    comparator = make_classifiers("direction")["logistic"]
                    comparison_x, comparison_y = _bounded_sample(x_direction.loc[:, columns], y_direction, max_train_rows)
                    comparator.fit(comparison_x, comparison_y.to_numpy())
                    comparison_prediction, comparison_probabilities = _predict_with_probabilities(comparator, xv_direction.loc[:, columns], DIRECTION_CLASSES)
                    comparison_metrics = classification_metrics(
                        yv_direction.to_numpy(), comparison_prediction, comparison_probabilities,
                        labels=list(DIRECTION_CLASSES), stress=False,
                    )
                    direction_fold_metrics.append({"week_utc": week, "model": comparison_name, **comparison_metrics})

            # δ sensitivity is fitted from the first training segment only;
            # each variant then uses the same expanding walk-forward folds.
            for quantile in delta_quantiles:
                delta_variant, variant_labels = delta_variants[quantile]
                variant_train = (times < train_end) & variant_labels.notna()
                variant_valid = (times >= validation_start) & (times < validation_end - pd.Timedelta(seconds=horizon)) & variant_labels.notna()
                variant_yv = variant_labels.loc[variant_valid]
                if variant_labels.loc[variant_train].nunique() < 2 or variant_yv.empty:
                    continue
                comparator = make_classifiers("direction")["logistic"]
                comparison_x, comparison_y = _sampled_training(frame, variant_labels, variant_train, LOB_CLASSIFICATION_FEATURES, max_train_rows)
                comparator.fit(comparison_x, comparison_y.to_numpy())
                xv = frame.loc[variant_valid, LOB_CLASSIFICATION_FEATURES]
                pred, prob = _predict_with_probabilities(comparator, xv, DIRECTION_CLASSES)
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
                pred, prob = _predict_with_probabilities(comparator, xv, DIRECTION_CLASSES)
                level_sensitivity_metrics.append({"week_utc": week, "depth_levels_L": depth, **classification_metrics(yv_direction.to_numpy(), pred, prob, labels=list(DIRECTION_CLASSES))})

            x_stress, y_stress = _sampled_training(frame, stress_labels, stress_train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
            y_stress = y_stress.astype(int)
            xv_stress = frame.loc[stress_valid_mask, LOB_CLASSIFICATION_FEATURES]
            yv_stress = stress_labels.loc[stress_valid_mask].astype(int)
            sscores, _ = _fit_candidates("stress", x_stress, y_stress, xv_stress, yv_stress, max_train_rows)
            if not y_stress.empty and not yv_stress.empty:
                persistence_states = _current_stress_state(frame.loc[stress_valid_mask], stress_thresholds)
                persistence_mask = persistence_states.notna().to_numpy()
                if persistence_mask.any():
                    persistent = persistence_states.loc[persistence_states.notna()].astype(int).to_numpy()
                    persistence_metrics = classification_metrics(
                        yv_stress.to_numpy()[persistence_mask], persistent,
                        np.column_stack([1 - persistent, persistent]),
                        labels=list(STRESS_CLASSES), stress=True,
                    )
                    stress_fold_metrics.append({"week_utc": week, "model": "stress_persistence", **persistence_metrics})
                else:
                    stress_fold_metrics.append({"week_utc": week, "model": "stress_persistence", "status": "unavailable", "reason": "No complete current stress state is available"})
            for name, values in sscores.items():
                stress_scores.setdefault(name, []).extend(values)
                stress_fold_metrics.append({"week_utc": week, "model": name, **values[-1]})
            stress_positive_train = stress_labels.loc[stress_train_mask].astype(int)
            fold_manifest.append({
                "symbol": symbol,
                "horizon_seconds": horizon,
                "validation_week_utc": week,
                "train_start_utc": times.min().isoformat(),
                "train_end_exclusive_utc": train_end.isoformat(),
                "embargo_seconds": int(embargo_seconds),
                "validation_start_utc": validation_start.isoformat(),
                "validation_end_exclusive_utc": (validation_end - pd.Timedelta(seconds=horizon)).isoformat(),
                "direction_train_rows": int(train_mask.sum()),
                "direction_train_rows_fitted": int(len(y_direction)),
                "direction_validation_rows": int(valid_mask.sum()),
                "stress_train_rows": int(stress_train_mask.sum()),
                "stress_validation_rows": int(stress_valid_mask.sum()),
                "stress_positive_rate_train": float(stress_positive_train.mean()) if len(stress_positive_train) else None,
                "warm_start_share_train": _warm_share(frame, train_mask),
                "warm_start_share_validation": _warm_share(frame, valid_mask),
                "seconds": round(time.perf_counter() - fold_started, 1),
            })
            print(f"{symbol} h={horizon} week {week}: {int(valid_mask.sum()):,} validation rows, {fold_manifest[-1]['seconds']}s", flush=True)

        if not score_final_test:
            result["horizons"][str(horizon)] = {
                "delta": delta,
                "delta_fitted_on": "first training segment (before the first validation week, minus embargo)",
                "walk_forward_direction": direction_fold_metrics,
                "walk_forward_stress": stress_fold_metrics,
                "direction_delta_sensitivity": delta_sensitivity_metrics,
                "direction_L_sensitivity": level_sensitivity_metrics,
                "stress_thresholds_by_fold": stress_threshold_history,
                "fold_manifest": fold_manifest,
                "selection_preview": {
                    "direction": _select_model("direction", direction_scores) if direction_scores else None,
                    "stress": _select_model("stress", stress_scores) if stress_scores else None,
                },
                "seconds": round(time.perf_counter() - horizon_started, 1),
            }
            continue

        direction_name = _select_model("direction", direction_scores)
        stress_name = _select_model("stress", stress_scores)
        final_stress_thresholds = fit_primary_stress_thresholds(
            _stress_threshold_rows(_threshold_frame(frame), times < dev_cutoff, settings.lob_stress_definition),
            settings.lob_stress_definition,
            settings.lob_stress_quantile,
            settings.lob_stress_depth_quantile,
        )
        final_stress_labels = make_stress_labels(frame, horizon, final_stress_thresholds)
        direction_train_mask = (times < dev_cutoff) & direction_labels.notna()
        stress_train_mask = (times < dev_cutoff) & final_stress_labels.notna()
        ys_train = final_stress_labels.loc[stress_train_mask].astype(int)
        stress_train_positive_rate = float(ys_train.mean()) if len(ys_train) else float("nan")
        prevalence_min = settings.lob_stress_min_positive_rate
        prevalence_max = settings.lob_stress_max_positive_rate
        if not np.isfinite(stress_train_positive_rate) or not prevalence_min <= stress_train_positive_rate <= prevalence_max:
            raise RuntimeError(
                f"{symbol} h={horizon}: development stress-label prevalence {stress_train_positive_rate:.4f} "
                f"is outside [{prevalence_min:.2%}, {prevalence_max:.2%}]; revise the train-only label definition before scoring the frozen test week"
            )
        x_train, y_train = _sampled_training(frame, direction_labels, direction_train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
        direction_model = _fit_final("direction", direction_name, x_train, y_train, max_train_rows)
        direction_oos_truth, direction_oos_probability = _collect_selected_oos(
            "direction", direction_name, frame, times, horizon, delta, embargo_seconds,
            max_train_rows, folds, settings.lob_stress_quantile, settings.lob_stress_depth_quantile,
            settings.lob_stress_definition,
        )
        direction_calibrators = _fit_isotonic(direction_oos_truth, direction_oos_probability, DIRECTION_CLASSES)
        xs_train, ys_train_fit = _sampled_training(frame, final_stress_labels, stress_train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows)
        stress_model = _fit_final("stress", stress_name, xs_train, ys_train_fit.astype(int), max_train_rows)
        stress_oos_truth, stress_oos_probability = _collect_selected_oos(
            "stress", stress_name, frame, times, horizon, delta, embargo_seconds,
            max_train_rows, folds, settings.lob_stress_quantile, settings.lob_stress_depth_quantile,
            settings.lob_stress_definition,
        )
        stress_calibrator = _fit_stress_isotonic(stress_oos_truth, stress_oos_probability)
        if not len(stress_oos_truth) or len(np.unique(stress_oos_truth)) < 2:
            raise RuntimeError(f"{symbol} h={horizon}: development walk-forward stress predictions do not contain both classes; refusing to invent a decision threshold")
        stress_threshold_raw = _threshold_for_stress(stress_oos_probability[:, 1], stress_oos_truth)
        stress_oos_probability_calibrated = _apply_stress_isotonic(stress_oos_probability[:, 1], stress_calibrator)
        stress_threshold = _threshold_for_stress(stress_oos_probability_calibrated, stress_oos_truth)

        spread_sensitivity_thresholds = fit_primary_stress_thresholds(
            _stress_threshold_rows(_threshold_frame(frame), times < dev_cutoff, "spread_only"),
            "spread_only",
            settings.lob_spread_sensitivity_quantile,
            settings.lob_stress_depth_quantile,
        )
        spread_sensitivity_labels = make_stress_labels(frame, horizon, spread_sensitivity_thresholds)
        spread_sensitivity_train_mask = (times < dev_cutoff) & spread_sensitivity_labels.notna()
        spread_sensitivity_train_y = spread_sensitivity_labels.loc[spread_sensitivity_train_mask].astype(int)
        spread_sensitivity: dict[str, object] = {
            "status": "unavailable",
            "definition": f"spread_only_train_q{settings.lob_spread_sensitivity_quantile * 100:g}_sensitivity",
            "thresholds_train": spread_sensitivity_thresholds.as_dict(),
            "positive_rate_by_month_development": monthly_positive_rates(
                spread_sensitivity_labels.loc[times < dev_cutoff], times.loc[times < dev_cutoff]
            ),
        }
        spread_sensitivity_model = None
        if spread_sensitivity_train_y.nunique() == 2:
            spread_x, spread_y = _sampled_training(
                frame, spread_sensitivity_labels, spread_sensitivity_train_mask, LOB_CLASSIFICATION_FEATURES, max_train_rows,
            )
            spread_sensitivity_model = _fit_final("stress", stress_name, spread_x, spread_y.astype(int), max_train_rows)
        else:
            spread_sensitivity["reason"] = "Spread-only train thresholds produced a single class; sensitivity label is degenerate"

        regime_training_rv = pd.to_numeric(frame.loc[times < dev_cutoff, "rv_300s"], errors="coerce").dropna()
        if len(regime_training_rv) and regime_training_rv.quantile(1 / 3) < regime_training_rv.quantile(2 / 3):
            regime_cutoffs = [float(regime_training_rv.quantile(1 / 3)), float(regime_training_rv.quantile(2 / 3))]
        else:
            regime_cutoffs = None

        # Do not slice or label the frozen test week until models, calibration,
        # decision thresholds, and the development prevalence guard are final.
        if test_frame is None:
            if output_dir is not None:
                # Recorded before the first test partition is read, so a crash after this point
                # still counts as an opened test and a rerun needs --allow-repeat-final-test.
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / FINAL_TEST_OPENED_MARKER).write_text(json.dumps({
                    "symbol": symbol,
                    "opened_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "first_horizon_seconds": horizon,
                    "test_window_utc": [settings.lob_untouched_test_start_utc, settings.lob_untouched_test_end_utc],
                }, indent=2), encoding="utf-8")
            test_frame = _with_depth_qty_10(_load_features(symbol, settings, end_exclusive=test_end, start_inclusive=test_start))
            loaded_times = pd.to_datetime(test_frame["timestamp_utc"], utc=True)
            test_frame = test_frame.loc[(loaded_times >= test_start) & (loaded_times < test_end)].reset_index(drop=True)
            result["data_rows"] = int(len(frame) + len(test_frame))
            result["test_rows_loaded"] = int(len(test_frame))
            if len(test_frame):
                result["reproducibility"]["observed_data_end_utc"] = pd.to_datetime(test_frame["timestamp_utc"], utc=True).max().isoformat()
        if test_frame.empty:
            raise RuntimeError(
                f"No data exists in the frozen final-test week {settings.lob_untouched_test_start_utc[:10]}"
                f" through {settings.lob_untouched_test_end_utc[:10]}; collect the configured capture window first"
            )
        test_columns = ["timestamp_utc", "mid_price", *LOB_CLASSIFICATION_FEATURES]
        test_slice = test_frame[test_columns]
        if "final_test_data_sha256" not in result:
            result["final_test_scored_once"] = True
            result["final_test_data_sha256"] = hashlib.sha256(
                pd.util.hash_pandas_object(test_slice, index=False).values.tobytes()
            ).hexdigest()
        test_times = pd.to_datetime(test_frame["timestamp_utc"], utc=True)
        direction_test_labels = make_direction_labels(test_frame, horizon, delta)
        stress_test_labels = make_stress_labels(test_frame, horizon, final_stress_thresholds)
        test_mask = (test_times < test_end - pd.Timedelta(seconds=horizon)) & direction_test_labels.notna()
        stress_test_mask = (test_times < test_end - pd.Timedelta(seconds=horizon)) & stress_test_labels.notna()
        if not test_mask.any() or not stress_test_mask.any():
            raise RuntimeError(f"{symbol} h={horizon}: no complete final-test labels are available")

        x_test = test_frame.loc[test_mask, LOB_CLASSIFICATION_FEATURES]
        y_test = direction_test_labels.loc[test_mask]
        direction_prediction = direction_model.predict(x_test)
        direction_probability_raw = _align_probabilities(direction_model, x_test, DIRECTION_CLASSES)
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
        direction_persistence_all = _direction_persistence_predictions(test_frame, delta)
        direction_persistence_test = direction_persistence_all[test_mask.to_numpy()]
        direction_persistence_probability = np.column_stack([
            direction_persistence_test == label for label in DIRECTION_CLASSES
        ]).astype(float)
        direction_pairwise = paired_classification_tests(
            y_test.to_numpy(), direction_prediction_calibrated, direction_persistence_test,
            direction_probability, direction_persistence_probability,
            stress=False, horizon_seconds=horizon, timestamps=test_times.loc[test_mask],
        )

        xs_test = test_frame.loc[stress_test_mask, LOB_CLASSIFICATION_FEATURES]
        ys_test = stress_test_labels.loc[stress_test_mask].astype(int)
        stress_probability_raw = _align_probabilities(stress_model, xs_test, STRESS_CLASSES)
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
        stress_persistence_states = _current_stress_state(test_frame.loc[stress_test_mask], final_stress_thresholds)
        stress_persistence_mask = stress_persistence_states.notna().to_numpy()
        stress_persistence_test = stress_persistence_states.loc[stress_persistence_states.notna()].astype(int).to_numpy()
        stress_persistence_probability = np.column_stack([1 - stress_persistence_test, stress_persistence_test]).astype(float)
        stress_test_probability = np.column_stack([1.0 - stress_probability, stress_probability])
        stress_test_timestamps = test_times.loc[stress_test_mask]
        stress_pairwise = paired_classification_tests(
            ys_test.to_numpy()[stress_persistence_mask], stress_prediction[stress_persistence_mask], stress_persistence_test,
            stress_test_probability[stress_persistence_mask], stress_persistence_probability,
            stress=True, horizon_seconds=horizon, timestamps=stress_test_timestamps.iloc[np.flatnonzero(stress_persistence_mask)],
        )
        stress_persistence_test_metrics = classification_metrics(
            ys_test.to_numpy()[stress_persistence_mask], stress_persistence_test, stress_persistence_probability,
            labels=list(STRESS_CLASSES), stress=True,
        ) if stress_persistence_mask.any() else {"n": 0, "status": "unavailable", "reason": "No complete current stress state is available"}
        spread_sensitivity_test_labels = make_stress_labels(test_frame, horizon, spread_sensitivity_thresholds)
        spread_sensitivity_test_mask = (test_times < test_end - pd.Timedelta(seconds=horizon)) & spread_sensitivity_test_labels.notna()
        spread_sensitivity_test_y = spread_sensitivity_test_labels.loc[spread_sensitivity_test_mask].astype(int)
        spread_sensitivity["positive_rate_by_month_test"] = monthly_positive_rates(
            spread_sensitivity_test_labels.loc[spread_sensitivity_test_mask], test_times.loc[spread_sensitivity_test_mask]
        )
        if spread_sensitivity_model is not None and not spread_sensitivity_test_y.empty:
            spread_sensitivity_x_test = test_frame.loc[spread_sensitivity_test_mask, LOB_CLASSIFICATION_FEATURES]
            spread_sensitivity_probability = _align_probabilities(spread_sensitivity_model, spread_sensitivity_x_test, STRESS_CLASSES)
            spread_sensitivity_prediction = (spread_sensitivity_probability[:, 1] >= 0.5).astype(int)
            spread_sensitivity.update({
                "status": "scored",
                "test_metrics": classification_metrics(
                    spread_sensitivity_test_y.to_numpy(), spread_sensitivity_prediction,
                    spread_sensitivity_probability, labels=list(STRESS_CLASSES), stress=True,
                ),
            })
        else:
            spread_sensitivity["reason"] = "No valid untouched-test spread-only labels are available"
        if regime_cutoffs is not None:
            test_regimes = pd.cut(
                pd.to_numeric(test_frame.loc[test_mask, "rv_300s"], errors="coerce"),
                bins=[-np.inf, regime_cutoffs[0], regime_cutoffs[1], np.inf],
                labels=["low_volatility", "medium_volatility", "high_volatility"],
                include_lowest=True,
            )
            stress_test_regimes = pd.cut(
                pd.to_numeric(test_frame.loc[stress_test_mask, "rv_300s"], errors="coerce"),
                bins=[-np.inf, regime_cutoffs[0], regime_cutoffs[1], np.inf],
                labels=["low_volatility", "medium_volatility", "high_volatility"],
                include_lowest=True,
            )
        else:
            test_regimes = pd.Series([None] * int(test_mask.sum()), index=test_frame.index[test_mask])
            stress_test_regimes = pd.Series([None] * int(stress_test_mask.sum()), index=test_frame.index[stress_test_mask])
        direction_by_regime, stress_by_regime = {}, {}
        for regime in ("low_volatility", "medium_volatility", "high_volatility"):
            selected_direction = np.asarray(test_regimes == regime)
            if selected_direction.any():
                direction_by_regime[regime] = classification_metrics(
                    y_test.to_numpy()[selected_direction],
                    direction_prediction_calibrated[selected_direction],
                    direction_probability[selected_direction],
                    labels=list(DIRECTION_CLASSES), stress=False,
                )
            selected_stress = np.asarray(stress_test_regimes == regime)
            if selected_stress.any():
                stress_targets = ys_test.to_numpy()[selected_stress]
                stress_probs = stress_probability[selected_stress]
                stress_preds = (stress_probs >= stress_threshold).astype(int)
                stress_by_regime[regime] = classification_metrics(
                    stress_targets, stress_preds,
                    np.column_stack([1 - stress_probs, stress_probs]),
                    labels=list(STRESS_CLASSES), stress=True,
                )
        # Pre-specified secondary comparisons (RQ1/RQ2 ablations and every candidate model), fitted
        # on the same final training sample and scored in this same single pass over the test rows.
        # They are uncalibrated, as in walk-forward; the selected, calibrated model stays the headline.
        direction_comparisons: dict[str, np.ndarray] = {}
        direction_prior = direction_labels.loc[direction_train_mask].value_counts(normalize=True)
        direction_comparisons["class_frequency"] = np.tile(
            np.asarray([direction_prior.get(label, 0.0) for label in DIRECTION_CLASSES]), (len(y_test), 1)
        )
        for comparison_name, columns in (("logistic_obi_only", OBI_ONLY_COLUMNS), ("logistic_obi_plus_ofi", OBI_PLUS_OFI_COLUMNS)):
            comparator = make_classifiers("direction")["logistic"]
            comparison_x, comparison_y = _bounded_sample(x_train.loc[:, columns], y_train, max_train_rows)
            comparator.fit(comparison_x, comparison_y.to_numpy())
            direction_comparisons[comparison_name] = _align_probabilities(comparator, x_test.loc[:, columns], DIRECTION_CLASSES)
        for candidate in make_classifiers("direction"):
            model = direction_model if candidate == direction_name else _fit_final("direction", candidate, x_train, y_train, max_train_rows)
            direction_comparisons[candidate] = _align_probabilities(model, x_test, DIRECTION_CLASSES)
        direction_test_comparisons = {}
        for name, probabilities in direction_comparisons.items():
            prediction = (
                np.repeat(direction_prior.index[0], len(y_test)) if name == "class_frequency"
                else np.asarray(DIRECTION_CLASSES)[np.argmax(probabilities, axis=1)]
            )
            direction_test_comparisons[name] = classification_metrics(
                y_test.to_numpy(), prediction, probabilities, labels=list(DIRECTION_CLASSES), stress=False,
            )
        direction_test_comparisons["direction_persistence"] = classification_metrics(
            y_test.to_numpy(), direction_persistence_test, direction_persistence_probability,
            labels=list(DIRECTION_CLASSES), stress=False,
        )
        stress_comparisons: dict[str, np.ndarray] = {}
        for candidate in make_classifiers("stress"):
            model = stress_model if candidate == stress_name else _fit_final("stress", candidate, xs_train, ys_train_fit.astype(int), max_train_rows)
            stress_comparisons[candidate] = _align_probabilities(model, xs_test, STRESS_CLASSES)[:, 1]
        stress_test_comparisons = {
            name: classification_metrics(
                ys_test.to_numpy(), (probability >= 0.5).astype(int), np.column_stack([1.0 - probability, probability]),
                labels=list(STRESS_CLASSES), stress=True,
            )
            for name, probability in stress_comparisons.items()
        }
        stress_test_comparisons["stress_persistence"] = stress_persistence_test_metrics
        prediction_files = {}
        if output_dir is not None:
            direction_rows = {
                "timestamp_utc": test_times.loc[test_mask].to_numpy(),
                "y_true": y_test.astype(str).to_numpy(),
                "rv_300s": pd.to_numeric(test_frame.loc[test_mask, "rv_300s"], errors="coerce").to_numpy(),
                "direction_persistence": direction_persistence_test.astype(str),
            }
            for index, label in enumerate(DIRECTION_CLASSES):
                direction_rows[f"selected_calibrated_p_{label}"] = direction_probability[:, index].astype(np.float32)
            for name, probabilities in direction_comparisons.items():
                for index, label in enumerate(DIRECTION_CLASSES):
                    direction_rows[f"{name}_p_{label}"] = probabilities[:, index].astype(np.float32)
            stress_persistence_column = np.full(len(ys_test), -1, dtype=np.int8)
            stress_persistence_column[stress_persistence_mask] = stress_persistence_test
            stress_rows = {
                "timestamp_utc": stress_test_timestamps.to_numpy(),
                "y_true": ys_test.to_numpy().astype(np.int8),
                "rv_300s": pd.to_numeric(test_frame.loc[stress_test_mask, "rv_300s"], errors="coerce").to_numpy(),
                "stress_persistence": stress_persistence_column,  # -1 where the current state is undefined
                "selected_calibrated_p_stress": stress_probability.astype(np.float32),
                **{f"{name}_p_stress": probability.astype(np.float32) for name, probability in stress_comparisons.items()},
            }
            for task, rows in (("direction", direction_rows), ("stress", stress_rows)):
                path = output_dir / f"test_predictions_h{horizon}_{task}.parquet"
                pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
                prediction_files[task] = path.name
        direction_counts = y_train.value_counts(normalize=True).to_dict()
        stress_counts = ys_train.value_counts(normalize=True).to_dict()
        artifact = {
            "symbol": symbol,
            "horizon_seconds": horizon,
            "target_names": {"direction": "midprice_direction", "stress": "liquidity_stress"},
            "direction_model_name": direction_name,
            "stress_model_name": stress_name,
            "study_label": study_label,
            "feature_schema_version": LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION,
            "label_schema_versions": {
                "direction": DIRECTION_LABEL_SCHEMA_VERSION,
                "stress": STRESS_LABEL_SCHEMA_VERSION,
            },
            "max_train_rows_per_fit": int(max_train_rows),
            "model_version": f"{direction_name}-{stress_name}-h{horizon}-{test_start.date().isoformat()}",
            "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "reproducibility": result["reproducibility"],
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
        artifact_path = artifact_root / symbol / f"horizon_{horizon}.joblib"
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
                stress_test_labels.loc[stress_test_mask], test_times.loc[stress_test_mask]
            ),
            "stress_prevalence_check": "within_configured_range",
            "direction_class_distribution_train": {str(key): float(value) for key, value in direction_counts.items()},
            "stress_class_distribution_train": {str(key): float(value) for key, value in stress_counts.items()},
            "direction_test": direction_test_metrics,
            "direction_test_raw": direction_test_metrics_raw,
            "direction_paired_tests_vs_persistence": direction_pairwise,
            "stress_test": stress_test_metrics,
            "stress_test_raw": stress_test_metrics_raw,
            "stress_persistence_test": stress_persistence_test_metrics,
            "stress_paired_tests_vs_persistence": stress_pairwise,
            "stress_spread_only_sensitivity": spread_sensitivity,
            "stress_decision_threshold": stress_threshold,
            "stress_decision_threshold_raw": stress_threshold_raw,
            "regime_cutoffs_rv_300s_train_terciles": regime_cutoffs,
            "direction_test_by_regime": direction_by_regime,
            "stress_test_by_regime": stress_by_regime,
            "direction_test_comparisons": direction_test_comparisons,
            "stress_test_comparisons": stress_test_comparisons,
            "stress_test_comparisons_threshold": 0.5,
            "test_prediction_files": prediction_files,
            "test_rows_direction": int(len(y_test)),
            "test_rows_stress": int(len(ys_test)),
            "artifact": str(artifact_path.relative_to(REPOSITORY_ROOT) if artifact_path.is_relative_to(REPOSITORY_ROOT) else artifact_path),
            "artifact_bytes": int(artifact_path.stat().st_size),
            "direction_runtime_profile": direction_runtime,
            "stress_runtime_profile": stress_runtime,
        }
    return result


def _scalar_metrics(row: dict) -> dict:
    return {key: value for key, value in row.items() if not isinstance(value, (dict, list))}


def write_fold_tables(payload: dict, symbol_dir: Path) -> None:
    """Flatten per-fold manifests and metrics into CSVs for tables and time plots."""
    manifest_rows, metric_rows = [], []
    for horizon, block in payload.get("horizons", {}).items():
        manifest_rows.extend(block.get("fold_manifest", []))
        sources = (
            ("direction", "walk_forward_direction", None),
            ("stress", "walk_forward_stress", None),
            ("direction", "direction_delta_sensitivity", "delta_quantile"),
            ("direction", "direction_L_sensitivity", "depth_levels_L"),
        )
        for task, key, variant in sources:
            for row in block.get(key, []):
                flat = _scalar_metrics(row)
                if variant is not None:
                    flat["model"] = f"logistic_{variant}={row.get(variant)}"
                metric_rows.append({"symbol": payload["symbol"], "horizon_seconds": int(horizon), "task": task, **flat})
    pd.DataFrame(manifest_rows).to_csv(symbol_dir / "fold_manifest.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    leading = [name for name in ("symbol", "horizon_seconds", "task", "week_utc", "model") if name in metrics]
    metrics[leading + [name for name in metrics.columns if name not in leading]].to_csv(symbol_dir / "fold_metrics.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.yaml", help="Study settings file; defaults to config/settings.yaml")
    parser.add_argument("--metrics-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "historical_archive_pilot", help="Metrics output root; defaults to the already-scored submission pilot")
    parser.add_argument("--artifact-dir", type=Path, default=REPOSITORY_ROOT / "backend" / "artifacts" / "meter", help="Model artifact root; defaults to backend/artifacts/meter")
    parser.add_argument("--study-label", default="historical_archive_pilot", help="Provenance label stored in metrics and model artifacts")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10))
    parser.add_argument("--embargo-seconds", type=int)
    parser.add_argument("--max-train-rows", type=int, default=250000, help="Deterministic class-stratified cap for each model fit")
    parser.add_argument("--allow-repeat-final-test", action="store_true", help="Explicitly allow scoring the frozen final-test week again after a prior run")
    parser.add_argument(
        "--development-only", action="store_true",
        help="Run the walk-forward validation folds only. No partition dated on or after the untouched test start "
             "is opened, no model is selected or saved, and no final-test metrics are written.",
    )
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    metrics_dir = args.metrics_dir if args.metrics_dir.is_absolute() else REPOSITORY_ROOT / args.metrics_dir
    artifact_root = args.artifact_dir if args.artifact_dir.is_absolute() else REPOSITORY_ROOT / args.artifact_dir
    settings = load_settings(settings_path)
    embargo_seconds = args.embargo_seconds if args.embargo_seconds is not None else settings.lob_embargo_seconds
    if args.max_train_rows < len(DIRECTION_CLASSES) or embargo_seconds < 1:
        raise SystemExit("--max-train-rows must be at least 3 and --embargo-seconds must be positive")
    symbols = (args.symbol,) if args.symbol else settings.symbols
    horizons = (args.horizon,) if args.horizon else settings.lob_classification_horizons_seconds
    minimum_embargo = max(max(horizons), settings.lob_feature_lookback_seconds, settings.lob_deeplob_lookback_seconds)
    if embargo_seconds < minimum_embargo:
        raise SystemExit(f"Embargo must be at least {minimum_embargo} seconds for the configured horizon and feature/sequence windows")
    # A finished score or an opened-marker left by a run that stopped after reading the test both count.
    prior_outputs = [metrics_dir / symbol / name for symbol in symbols for name in ("classification_metrics.json", FINAL_TEST_OPENED_MARKER)]
    already_scored = [path for path in prior_outputs if path.is_file()]
    if already_scored and not args.allow_repeat_final_test and not args.development_only:
        paths = ", ".join(str(path.relative_to(REPOSITORY_ROOT)) for path in already_scored)
        raise SystemExit(f"Final-test output already exists for {paths}. Use --allow-repeat-final-test only when an intentional re-score is approved.")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Each symbol is written as soon as it finishes, so a later failure cannot discard a scored test.
    for symbol in symbols:
        symbol_dir = metrics_dir / symbol
        payload = run_symbol(
            symbol, horizons, embargo_seconds, args.max_train_rows, settings,
            settings_path, artifact_root, args.study_label,
            score_final_test=not args.development_only,
            output_dir=None if args.development_only else symbol_dir,
        )
        symbol_dir.mkdir(parents=True, exist_ok=True)
        if args.development_only:
            destination = symbol_dir / "walk_forward_metrics.json"
            destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            write_fold_tables(payload, symbol_dir)
            print(f"{symbol}: wrote development walk-forward metrics and fold tables to {symbol_dir}")
            continue
        destination = symbol_dir / "classification_metrics.json"
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        write_fold_tables(payload, symbol_dir)
        print(f"{symbol}: wrote walk-forward and final-test metrics to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

