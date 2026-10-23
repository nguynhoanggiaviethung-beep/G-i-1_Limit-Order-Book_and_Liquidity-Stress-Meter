"""Classification scores and chronological expanding-window helpers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    average_precision_score,
)
from scipy.stats import binomtest, norm


def paired_classification_tests(
    y_true,
    prediction,
    baseline_prediction,
    probabilities,
    baseline_probabilities,
    *,
    stress: bool = False,
    horizon_seconds: int = 1,
    baseline_name: str = "persistence",
    timestamps=None,
) -> dict[str, Any]:
    """Compare paired forecasts with exact McNemar and minute-blocked HAC DM.

    Brier-loss differences are averaged by UTC minute before inference, then
    use an automatic Newey-West bandwidth with a ten-minute minimum. The
    bandwidth therefore accounts for multi-minute error dependence as well
    as the short overlap induced by the forecast horizon. A positive mean
    difference means ``prediction`` has larger Brier loss.
    """
    truth = np.asarray(y_true)
    predicted = np.asarray(prediction)
    baseline = np.asarray(baseline_prediction)
    p = np.asarray(probabilities, dtype=float)
    p0 = np.asarray(baseline_probabilities, dtype=float)
    if not (len(truth) == len(predicted) == len(baseline) == len(p) == len(p0)):
        raise ValueError("paired predictions and labels must have equal lengths")
    if not len(truth):
        return {"n": 0, "mcnemar": None, "diebold_mariano_brier": None}

    correct = predicted == truth
    baseline_correct = baseline == truth
    model_only = int(np.sum(correct & ~baseline_correct))
    baseline_only = int(np.sum(~correct & baseline_correct))
    discordant = model_only + baseline_only
    mcnemar_p = float(binomtest(model_only, discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    mcnemar_stat = float((abs(model_only - baseline_only) - 1) ** 2 / discordant) if discordant else 0.0

    if stress:
        actual = (truth == 1).astype(float)
        model_positive = p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.reshape(-1)
        baseline_positive = p0[:, 1] if p0.ndim == 2 and p0.shape[1] > 1 else p0.reshape(-1)
        model_loss = (model_positive - actual) ** 2
        baseline_loss = (baseline_positive - actual) ** 2
    else:
        labels = tuple(sorted(set(truth.tolist()) | set(predicted.tolist()) | set(baseline.tolist())))
        positions = {label: index for index, label in enumerate(labels)}
        one_hot = np.zeros((len(truth), len(labels)), dtype=float)
        for row, label in enumerate(truth):
            one_hot[row, positions[label]] = 1.0
        if p.ndim != 2 or p0.ndim != 2 or p.shape[1] != len(labels) or p0.shape[1] != len(labels):
            raise ValueError("multiclass probabilities must have one column per sorted class")
        model_loss = np.sum((p - one_hot) ** 2, axis=1)
        baseline_loss = np.sum((p0 - one_hot) ** 2, axis=1)

    difference = model_loss - baseline_loss
    if timestamps is None:
        time_index = pd.date_range("2000-01-01", periods=len(difference), freq="s", tz="UTC")
    else:
        if len(timestamps) != len(difference):
            raise ValueError("timestamps must have the same length as paired predictions")
        time_index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
        if time_index.isna().any():
            raise ValueError("timestamps must contain valid UTC times")
    raw_count = len(difference)
    minute_values = pd.Series(difference, index=time_index).groupby(time_index.floor("min")).mean()
    minute_keys = (minute_values.index.asi8 // (60 * 1_000_000_000)).astype(np.int64)
    block_loss = minute_values.to_numpy(dtype=float)
    n_blocks = len(block_loss)
    mean_difference = float(np.mean(block_loss))
    centered = block_loss - mean_difference
    # Andrews/Newey-West plug-in bandwidth, with a ten-minute floor for
    # short-horizon LOB loss series and a gap-aware minute-lag covariance.
    automatic_lag = int(np.floor(4 * (n_blocks / 100.0) ** (2.0 / 9.0))) if n_blocks else 0
    max_lag = min(max(10, automatic_lag, int(np.ceil(max(horizon_seconds - 1, 0) / 60))), max(n_blocks - 1, 0))
    long_run_variance = float(np.dot(centered, centered) / n_blocks) if n_blocks else 0.0
    positions_by_minute = {int(minute): index for index, minute in enumerate(minute_keys)}
    for lag in range(1, max_lag + 1):
        products = [
            centered[position] * centered[positions_by_minute[int(minute) - lag]]
            for position, minute in enumerate(minute_keys)
            if int(minute) - lag in positions_by_minute
        ]
        if not products:
            continue
        weight = 1.0 - lag / (max_lag + 1.0)
        autocovariance = float(np.sum(products) / n_blocks)
        long_run_variance += 2.0 * weight * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    if long_run_variance > 0 and n_blocks >= 30:
        dm_stat = float(mean_difference / np.sqrt(long_run_variance / n_blocks))
        dm_p = float(2 * norm.sf(abs(dm_stat)))
    else:
        dm_stat = None
        dm_p = None
    return {
        "n": int(raw_count),
        "baseline": baseline_name,
        "mcnemar": {
            "model_only_correct": model_only,
            "baseline_only_correct": baseline_only,
            "continuity_corrected_chi2": mcnemar_stat,
            "p_value_exact_two_sided": mcnemar_p,
        },
        "diebold_mariano_brier": {
            "mean_loss_difference_model_minus_baseline": mean_difference,
            "hac_lags": max_lag,
            "hac_lag_unit": "minutes",
            "block_duration_seconds": 60,
            "n_blocks": int(n_blocks),
            "minimum_blocks_for_p_value": 30,
            "statistic": dm_stat,
            "p_value_two_sided_normal": dm_p,
        },
    }


def classification_metrics(y_true, y_pred, probabilities=None, labels=None, stress: bool = False) -> dict[str, Any]:
    """Compute imbalance-aware metrics without using accuracy as the selector."""
    labels = list(labels or sorted(set(y_true)))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "precision_by_class": {str(label): float(value) for label, value in zip(labels, precision)},
        "recall_by_class": {str(label): float(value) for label, value in zip(labels, recall)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }
    if stress:
        positives = (np.asarray(y_true) == 1).astype(int)
        result["positive_rate"] = float(positives.mean()) if len(positives) else None
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if stress:
            positive_column = labels.index(1) if 1 in labels else p.shape[1] - 1
            positive = (np.asarray(y_true) == 1).astype(int)
            if positive_column < p.shape[1]:
                scores = p[:, positive_column]
                result["pr_auc"] = float(average_precision_score(positive, scores)) if positive.sum() else None
                result["brier"] = float(np.mean((scores - positive) ** 2))
                result["calibration_bins"] = _calibration_bins(positive, scores)
                result["ece"] = _expected_calibration_error(positive, scores)
        else:
            target = np.zeros_like(p)
            label_positions = {label: index for index, label in enumerate(labels)}
            for row, label in enumerate(y_true):
                if label in label_positions and label_positions[label] < p.shape[1]:
                    target[row, label_positions[label]] = 1.0
            result["brier"] = float(np.mean(np.sum((p - target) ** 2, axis=1)))
            result["calibration_by_class"] = {
                str(label): _calibration_bins(target[:, index], p[:, index])
                for index, label in enumerate(labels) if index < p.shape[1]
            }
            result["ece_by_class"] = {
                str(label): _expected_calibration_error(target[:, index], p[:, index])
                for index, label in enumerate(labels) if index < p.shape[1]
            }
    return result


def _calibration_bins(observed: np.ndarray, predicted: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for index in range(bins):
        selected = (predicted >= edges[index]) & (predicted <= edges[index + 1] if index == bins - 1 else predicted < edges[index + 1])
        if selected.any():
            output.append({
                "mean_probability": float(np.mean(predicted[selected])),
                "observed_rate": float(np.mean(observed[selected])),
                "count": int(selected.sum()),
            })
    return output


def _expected_calibration_error(observed: np.ndarray, predicted: np.ndarray, bins: int = 10) -> float | None:
    if len(observed) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (predicted >= edges[index]) & (predicted <= edges[index + 1] if index == bins - 1 else predicted < edges[index + 1])
        if selected.any():
            error += float(selected.mean()) * abs(float(np.mean(predicted[selected])) - float(np.mean(observed[selected])))
    return error


def expanding_month_folds(timestamps: pd.Series, first_validation_month: str = "2025-07"):
    """Yield chronological train/validation masks by calendar month."""
    values = pd.to_datetime(timestamps, utc=True)
    first = pd.Period(first_validation_month, freq="M")
    months = values.dt.to_period("M")
    for month in sorted(month for month in months.dropna().unique() if month >= first):
        validation = months == month
        train = months < month
        yield month, train.to_numpy(), validation.to_numpy()


def expanding_week_folds(first_validation_week_utc: str, development_end_utc: str) -> tuple[str, ...]:
    """Return full UTC Monday-to-Sunday validation weeks before the test week."""
    first = pd.Timestamp(first_validation_week_utc)
    development_end = pd.Timestamp(development_end_utc)
    if first.tzinfo is None or development_end.tzinfo is None:
        raise ValueError("Weekly split timestamps must include a UTC offset")
    first = first.tz_convert("UTC")
    development_end = development_end.tz_convert("UTC")
    if first.weekday() != 0 or first != first.normalize():
        raise ValueError("First validation week must start Monday 00:00 UTC")
    development_exclusive = development_end + timedelta(seconds=1)
    if development_exclusive.weekday() != 0 or development_exclusive != development_exclusive.normalize():
        raise ValueError("Development must end on Sunday 23:59:59 UTC")
    last_full_week = development_exclusive - timedelta(days=7)
    if first > last_full_week:
        return ()
    return tuple(pd.date_range(first, last_full_week, freq="7D").strftime("%Y-%m-%d"))


def apply_embargo(train_mask: np.ndarray, timestamps: pd.Series, validation_start: pd.Timestamp, seconds: int) -> np.ndarray:
    """Drop training origins within the configured purge/embargo interval."""
    values = pd.to_datetime(timestamps, utc=True)
    return train_mask & (values < validation_start - timedelta(seconds=seconds)).to_numpy()

