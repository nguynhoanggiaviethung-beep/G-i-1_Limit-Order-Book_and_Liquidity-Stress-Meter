"""The fast metric, sampling and scoring paths must reproduce the original outputs exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from app.evaluation.lob_classification import _calibration_bins, _expected_calibration_error, classification_metrics
from app.models.deeplob_classifier import DEEPLOB_INPUT_COLUMNS, build_lob_sequences
from app.models.lob_classifiers import make_classifiers
from scripts.run_lob_classification import DIRECTION_CLASSES, STRESS_CLASSES, _bounded_sample, _predict_with_probabilities


def _reference_metrics(y_true, y_pred, probabilities=None, labels=None, stress=False):
    """The sklearn implementation that classification_metrics replaced, kept verbatim."""
    labels = list(labels or sorted(set(y_true)))
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result = {
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
            result["calibration_by_class"] = {str(label): _calibration_bins(target[:, index], p[:, index]) for index, label in enumerate(labels) if index < p.shape[1]}
            result["ece_by_class"] = {str(label): _expected_calibration_error(target[:, index], p[:, index]) for index, label in enumerate(labels) if index < p.shape[1]}
    return result


def _reference_bounded_sample(x, y, maximum, seed=42):
    """The label-based stratified sampler that _bounded_sample replaced, kept verbatim."""
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


def _probabilities(rng, rows, columns):
    raw = rng.random((rows, columns))
    raw[: rows // 10] = np.round(raw[: rows // 10], 1)  # exact bin edges and ties
    return raw / raw.sum(axis=1, keepdims=True)


DIRECTION_CASES = {
    "typical": (["down", "flat", "up"], [0.2, 0.5, 0.3], [0.25, 0.45, 0.3]),
    "never_predicts_up": (["down", "flat", "up"], [0.2, 0.5, 0.3], [0.4, 0.6, 0.0]),
    "single_prediction": (["down", "flat", "up"], [0.2, 0.5, 0.3], [0.0, 1.0, 0.0]),
    "up_absent_in_truth": (["down", "flat", "up"], [0.4, 0.6, 0.0], [0.3, 0.4, 0.3]),
}


@pytest.mark.parametrize("case", sorted(DIRECTION_CASES))
def test_direction_metrics_match_sklearn(case):
    labels, truth_mix, prediction_mix = DIRECTION_CASES[case]
    rng = np.random.default_rng(7)
    truth = rng.choice(labels, size=20_000, p=truth_mix).astype(object)
    prediction = rng.choice(labels, size=20_000, p=prediction_mix).astype(object)
    probabilities = _probabilities(rng, 20_000, 3)
    assert classification_metrics(truth, prediction, probabilities, labels=list(DIRECTION_CLASSES)) == _reference_metrics(
        truth, prediction, probabilities, labels=list(DIRECTION_CLASSES)
    )


def test_metrics_match_sklearn_for_values_outside_the_label_list():
    rng = np.random.default_rng(11)
    truth = rng.choice(["down", "flat", "up", "gap"], size=5_000, p=[0.2, 0.5, 0.25, 0.05]).astype(object)
    prediction = rng.choice(["down", "flat", "up", "halt"], size=5_000, p=[0.2, 0.5, 0.25, 0.05]).astype(object)
    probabilities = _probabilities(rng, 5_000, 3)
    assert classification_metrics(truth, prediction, probabilities, labels=list(DIRECTION_CLASSES)) == _reference_metrics(
        truth, prediction, probabilities, labels=list(DIRECTION_CLASSES)
    )
    # String predictions from numpy (what _predict_with_probabilities returns) score the same.
    assert classification_metrics(truth, prediction.astype(str), labels=list(DIRECTION_CLASSES)) == _reference_metrics(
        truth, prediction, labels=list(DIRECTION_CLASSES)
    )


@pytest.mark.parametrize("prediction_rate", [0.0, 0.07, 0.5])
def test_stress_metrics_match_sklearn(prediction_rate):
    rng = np.random.default_rng(5)
    truth = (rng.random(30_000) < 0.06).astype(int)
    prediction = (rng.random(30_000) < prediction_rate).astype(int)
    positive = np.round(rng.random(30_000), 2)
    probabilities = np.column_stack([1 - positive, positive])
    assert classification_metrics(truth, prediction, probabilities, labels=list(STRESS_CLASSES), stress=True) == _reference_metrics(
        truth, prediction, probabilities, labels=list(STRESS_CLASSES), stress=True
    )


def _label_series(kind: str) -> pd.Series:
    rng = np.random.default_rng(13)
    rows = 50_000
    index = np.sort(rng.choice(np.arange(rows * 3), size=rows, replace=False))  # gaps, like a masked slice
    if kind == "direction":
        values = rng.choice(["down", "flat", "up"], size=rows, p=[0.21, 0.5, 0.29]).astype(object)
        return pd.Series(values, index=index, dtype=object)
    if kind == "direction_ties":
        values = np.repeat(np.asarray(["up", "down", "flat"], dtype=object), rows // 3 + 1)[:rows]
        return pd.Series(rng.permutation(values), index=index, dtype=object)
    if kind == "stress_nullable":
        values = pd.array((rng.random(rows) < 0.06).astype(int), dtype="Int64")
        values[rng.random(rows) < 0.01] = pd.NA
        return pd.Series(values, index=index)
    if kind == "rare_class":
        values = np.where(rng.random(rows) < 0.0001, "up", np.where(rng.random(rows) < 0.5, "down", "flat")).astype(object)
        return pd.Series(values, index=index, dtype=object)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["direction", "direction_ties", "stress_nullable", "rare_class"])
@pytest.mark.parametrize("maximum", [997, 20_000, 49_999])
def test_bounded_sample_selects_the_same_rows_as_the_label_based_version(kind, maximum):
    y = _label_series(kind)
    x = pd.DataFrame({"a": np.arange(len(y), dtype=float)}, index=y.index)
    expected_x, expected_y = _reference_bounded_sample(x, y, maximum)
    actual_x, actual_y = _bounded_sample(x, y, maximum)
    assert actual_y.index.equals(expected_y.index)
    pd.testing.assert_series_equal(actual_y, expected_y)
    pd.testing.assert_frame_equal(actual_x, expected_x)


def _reference_sequences(frame, endpoints, lookback, columns=DEEPLOB_INPUT_COLUMNS):
    """The whole-frame build_lob_sequences that the windowed version replaced, kept verbatim."""
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").astype("int64").to_numpy()
    valid = frame.get("book_valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool).to_numpy()
    selected_values, selected_endpoints = [], []
    for endpoint in np.asarray(endpoints, dtype=int):
        start = endpoint - lookback + 1
        if start < 0 or endpoint >= len(frame):
            continue
        segment = slice(start, endpoint + 1)
        if not valid[segment].all() or not np.all(np.diff(times[segment]) == 1_000_000_000):
            continue
        window = values[segment]
        if window.shape != (lookback, len(columns)) or not np.isfinite(window).all():
            continue
        if tuple(columns) == DEEPLOB_INPUT_COLUMNS:
            window = window.copy()
            mid = (window[:, 0] + window[:, 10]) / 2.0
            if np.any(mid <= 0):
                continue
            window[:, :20] = (window[:, :20] / mid[:, None] - 1.0) * 10000.0
        selected_values.append(window)
        selected_endpoints.append(endpoint)
    if not selected_values:
        return np.empty((0, lookback, len(columns)), dtype=np.float32), np.asarray([], dtype=int)
    return np.stack(selected_values).astype(np.float32, copy=False), np.asarray(selected_endpoints, dtype=int)


def test_windowed_sequences_match_the_whole_frame_version():
    rng = np.random.default_rng(23)
    rows = 3_000
    times = pd.date_range("2026-06-01", periods=rows, freq="1s").to_numpy()
    times[1_500:] += np.timedelta64(7, "s")  # a gap breaks contiguity
    frame = pd.DataFrame({"timestamp_utc": pd.DatetimeIndex(times).tz_localize("UTC"), "book_valid": rng.random(rows) > 0.002})
    mid = 100 + np.cumsum(rng.normal(0, 0.01, rows))
    for level in range(1, 11):
        frame[f"bid_price_{level}"] = mid - 0.01 * level
        frame[f"ask_price_{level}"] = mid + 0.01 * level
        frame[f"bid_qty_{level}"] = rng.gamma(2.0, 1.0, rows).astype(np.float32)
        frame[f"ask_qty_{level}"] = rng.gamma(2.0, 1.0, rows)
    frame.loc[rng.random(rows) < 0.001, "ask_qty_3"] = np.nan
    endpoints = np.concatenate([[-5, 0, 10, 10, rows - 1, rows + 3], rng.integers(0, rows, 400)])
    for lookback in (2, 50):
        expected_x, expected_endpoints = _reference_sequences(frame, endpoints, lookback)
        actual_x, actual_endpoints = build_lob_sequences(frame, endpoints, lookback)
        np.testing.assert_array_equal(actual_endpoints, expected_endpoints)
        np.testing.assert_array_equal(actual_x, expected_x)
    assert len(build_lob_sequences(frame, [-1, rows], 5)[0]) == 0


@pytest.mark.parametrize("task", ["direction", "stress"])
def test_single_scoring_pass_matches_predict(task):
    rng = np.random.default_rng(17)
    rows = 4_000
    x = pd.DataFrame(rng.normal(size=(rows, 5)), columns=[f"f{index}" for index in range(5)])
    x.iloc[rng.random(rows) < 0.05, 2] = np.nan
    signal = x["f0"].fillna(0) + 0.5 * x["f1"] + rng.normal(scale=1.0, size=rows)
    if task == "direction":
        y = pd.Series(np.where(signal > 0.6, "up", np.where(signal < -0.6, "down", "flat")), dtype=object)
        class_order = DIRECTION_CLASSES
    else:
        y = pd.Series((signal > 1.6).astype(int))
        class_order = STRESS_CLASSES
    train, valid = x.iloc[:3_000], x.iloc[3_000:]
    for name, model in make_classifiers(task).items():
        model.fit(train, y.iloc[:3_000].to_numpy())
        labels, probabilities = _predict_with_probabilities(model, valid, class_order)
        np.testing.assert_array_equal(labels, np.asarray(model.predict(valid)), err_msg=name)
        assert classification_metrics(y.iloc[3_000:].to_numpy(), labels, probabilities, labels=list(class_order), stress=task == "stress") == _reference_metrics(
            y.iloc[3_000:].to_numpy(), model.predict(valid), probabilities, labels=list(class_order), stress=task == "stress"
        ), name
