"""Classification scores and chronological expanding-window helpers."""

from __future__ import annotations

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


def apply_embargo(train_mask: np.ndarray, timestamps: pd.Series, validation_start: pd.Timestamp, seconds: int) -> np.ndarray:
    """Drop training origins within the configured purge/embargo interval."""
    values = pd.to_datetime(timestamps, utc=True)
    return train_mask & (values < validation_start - pd.Timedelta(seconds=seconds)).to_numpy()

