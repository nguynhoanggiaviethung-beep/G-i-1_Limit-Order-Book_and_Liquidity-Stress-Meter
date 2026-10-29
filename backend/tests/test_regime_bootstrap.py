from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from scripts.run_lob_regime_bootstrap import (
    REGIMES,
    Accumulator,
    average_precision_from_histogram,
    bootstrap_p_value,
    bootstrap_weights,
    holm,
    macro_f1_from_confusion,
    mcc_from_confusion,
    probability_bins,
    regime_codes,
)


def _confusion(truth, predicted, k):
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def test_confusion_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    truth, predicted = rng.integers(0, 3, 5000), rng.integers(0, 3, 5000)
    predicted[:2000] = truth[:2000]
    matrix = _confusion(truth, predicted, 3)
    assert mcc_from_confusion(matrix[None])[0] == pytest.approx(matthews_corrcoef(truth, predicted), rel=1e-12)
    assert macro_f1_from_confusion(matrix[None])[0] == pytest.approx(f1_score(truth, predicted, average="macro"), rel=1e-12)
    # A constant prediction has undefined MCC, which sklearn reports as 0.
    assert mcc_from_confusion(_confusion(truth, np.zeros_like(truth), 3)[None])[0] == 0.0


def test_binned_average_precision_is_exact_when_scores_sit_on_bins():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 2, 4000)
    scores = np.clip(np.round(0.3 * truth + rng.uniform(0, 0.7, 4000), 3) - 0.0005, 0, 1)  # one score per bin
    histogram = np.zeros((1000, 2))
    np.add.at(histogram, (probability_bins(scores), truth), 1)
    assert average_precision_from_histogram(histogram[None])[0] == pytest.approx(average_precision_score(truth, scores), rel=1e-12)


def test_accumulator_point_estimate_matches_pooled_metrics():
    rng = np.random.default_rng(3)
    n = 3000
    blocks = rng.integers(0, 10, n)
    truth = rng.integers(0, 2, n)
    probability = np.clip(0.4 * truth + rng.uniform(0, 0.6, n), 0, 1)
    probabilities = np.column_stack([1 - probability, probability])
    regimes = np.zeros((n, len(REGIMES)), dtype=bool)
    regimes[:, 0] = True
    regimes[: n // 2, 1] = True
    accumulator = Accumulator("stress", 10)
    accumulator.add(blocks, regimes, truth, probabilities)
    estimate = accumulator.metrics(np.ones((1, 10)), 0)
    assert estimate["n"][0] == n
    assert estimate["mcc"][0] == pytest.approx(matthews_corrcoef(truth, (probability >= 0.5).astype(int)), rel=1e-12)
    assert estimate["brier"][0] == pytest.approx(np.mean((probability - truth) ** 2), rel=1e-12)
    assert estimate["pr_auc"][0] == pytest.approx(average_precision_score(truth, probability), abs=2e-3)
    half = accumulator.metrics(np.ones((1, 10)), 1)
    assert half["n"][0] == n // 2
    # Dropping a block in the weights drops exactly its rows.
    weights = np.ones((1, 10))
    weights[0, 4] = 0
    assert accumulator.metrics(weights, 0)["n"][0] == n - int((blocks == 4).sum())


def test_bootstrap_weights_keep_each_week_size():
    block_weeks = np.repeat(np.arange(3), 5)
    weights = bootstrap_weights(block_weeks, 50, seed=0)
    for week in range(3):
        assert (weights[:, block_weeks == week].sum(axis=1) == 5).all()


def test_bootstrap_p_value_and_holm():
    assert bootstrap_p_value(np.full(999, 0.1)) == pytest.approx(1 / 1000)
    assert bootstrap_p_value(np.r_[np.full(500, -1.0), np.full(500, 1.0)]) == pytest.approx(1.0)
    adjusted = holm(pd.Series([0.01, 0.04, 0.03]))
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_regimes_use_training_cutoffs_and_utc_sessions():
    times = pd.Series(pd.to_datetime(["2026-06-01 03:00", "2026-06-01 09:00", "2026-06-01 20:00"], utc=True))
    codes = regime_codes(times, np.array([0.1, 0.5, np.nan]), (0.2, 0.4))
    names = list(REGIMES)
    assert codes[0, names.index("vol_low")] and codes[1, names.index("vol_high")]
    assert not codes[2, 1:4].any()  # no volatility regime without rv_300s
    assert codes[0, names.index("session_asia")] and codes[1, names.index("session_europe")] and codes[2, names.index("session_americas")]
