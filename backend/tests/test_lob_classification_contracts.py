import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.evaluation.lob_classification import apply_embargo, expanding_week_folds, paired_classification_tests
from app.features.classification_features import build_classification_features
from app.labels.direction import make_direction_labels
from app.labels.stress import (
    StressThresholds,
    fit_spread_only_threshold,
    fit_stress_thresholds,
    make_stress_labels,
    make_spread_only_labels,
    monthly_positive_rates,
)
from app.models.deeplob_classifier import DEEPLOB_INPUT_COLUMNS, DeepLOBClassifier, build_lob_sequences
from scripts.build_lob_classification_dataset import build_partition_features
from scripts.run_lob_classification import _current_stress_state, _select_model


def test_stress_thresholds_prefer_tick_spread_and_notional_depth():
    count = 100
    frame = pd.DataFrame({
        "spread_ticks": np.r_[np.ones(count - 2), 3.0, 4.0],
        "relative_spread_bps": np.linspace(0.1, 1.0, count),
        "price_impact_bps": np.r_[np.ones(count - 1), 10.0],
        "depth_notional_10": np.r_[np.full(count - 1, 10000.0), 10.0],
        "depth_qty_10": np.linspace(1.0, 100.0, count),
        "rv_60s": np.r_[np.ones(count - 1), 5.0],
    })
    thresholds = fit_stress_thresholds(frame)
    assert thresholds.spread_metric == "spread_ticks"
    assert thresholds.depth_metric == "depth_notional_10"
    assert 0 < thresholds.positive_rate_train < 0.10


def test_stress_positive_rates_are_reported_by_utc_month():
    timestamps = pd.Series(pd.date_range("2025-01-31T23:59:58Z", periods=5, freq="s"))
    labels = pd.Series([0, 1, 1, pd.NA, 0], dtype="Int64")
    rates = monthly_positive_rates(labels, timestamps)
    assert rates["2025-01"] == {"n": 2, "positive_rate": 0.5}
    assert rates["2025-02"] == {"n": 2, "positive_rate": 0.5}


def test_single_spread_sensitivity_uses_future_values_and_train_threshold():
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01T00:00:00Z", periods=6, freq="s"),
        "book_valid": True,
        "mid_price": [100.0] * 6,
        "spread_ticks": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    })
    metric, threshold = fit_spread_only_threshold(frame.iloc[:4], quantile=0.5)
    labels = make_spread_only_labels(frame, horizon_seconds=1, metric=metric, threshold=threshold)
    assert metric == "spread_ticks"
    assert labels.iloc[2] == 1
    assert labels.iloc[3] == 1
    assert pd.isna(labels.iloc[-1])


def test_composite_stress_label_stays_missing_when_a_future_component_is_missing():
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="s"),
        "book_valid": True,
        "mid_price": [100.0] * 5,
        "spread_ticks": [1.0, 1.0, np.nan, 1.0, 1.0],
        "price_impact_bps": [1.0] * 5,
        "depth_notional_10": [100.0] * 5,
        "rv_60s": [0.1] * 5,
    })
    thresholds = StressThresholds(2.0, 2.0, 50.0, 1.0, 0.05, definition="composite")
    labels = make_stress_labels(frame, horizon_seconds=1, thresholds=thresholds)
    assert labels.iloc[0] == 0
    assert pd.isna(labels.iloc[1])


@pytest.mark.parametrize("missing_column", ["spread_ticks", "price_impact_bps", "depth_notional_10", "rv_60s"])
def test_composite_stress_requires_every_future_component(missing_column):
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="s"),
        "book_valid": True,
        "mid_price": [100.0] * 3,
        "spread_ticks": [1.0] * 3,
        "price_impact_bps": [1.0] * 3,
        "depth_notional_10": [100.0] * 3,
        "rv_60s": [0.1] * 3,
    })
    frame.loc[1, missing_column] = np.nan
    thresholds = StressThresholds(2.0, 2.0, 50.0, 1.0, 0.05, definition="composite")
    labels = make_stress_labels(frame, horizon_seconds=1, thresholds=thresholds)
    assert pd.isna(labels.iloc[0])


def test_composite_stress_uses_strict_or_thresholds_for_each_component():
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01T00:00:00Z", periods=7, freq="s"),
        "book_valid": True,
        "mid_price": [100.0] * 7,
        "spread_ticks": [1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0],
        "price_impact_bps": [1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0],
        "depth_notional_10": [100.0, 100.0, 100.0, 20.0, 100.0, 50.0, 100.0],
        "rv_60s": [0.1, 0.1, 0.1, 0.1, 3.0, 1.0, 0.1],
    })
    thresholds = StressThresholds(2.0, 2.0, 50.0, 1.0, 0.05, definition="composite")
    labels = make_stress_labels(frame, horizon_seconds=1, thresholds=thresholds)
    assert labels.iloc[:4].tolist() == [1, 1, 1, 1]
    assert labels.iloc[4] == 0


def test_current_composite_stress_state_does_not_treat_missing_as_non_stress():
    frame = pd.DataFrame({
        "book_valid": [True, True],
        "spread_ticks": [1.0, np.nan],
        "price_impact_bps": [1.0, 1.0],
        "depth_notional_10": [100.0, 100.0],
        "rv_60s": [0.1, 0.1],
    })
    thresholds = StressThresholds(2.0, 2.0, 50.0, 1.0, 0.05, definition="composite")
    states = _current_stress_state(frame, thresholds)
    assert states.iloc[0] == 0
    assert pd.isna(states.iloc[1])


def test_model_selection_uses_mcc_only_as_a_tie_breaker():
    scores = {
        "logistic": [{"macro_f1": 0.6, "pr_auc": 0.4, "mcc": 0.3}],
        "random_forest": [{"macro_f1": 0.6, "pr_auc": 0.4, "mcc": 0.5}],
        "xgboost": [{"macro_f1": 0.59, "pr_auc": 0.45, "mcc": 0.9}],
    }
    assert _select_model("direction", scores) == "random_forest"
    assert _select_model("stress", scores) == "xgboost"


def test_direction_labels_require_exact_contiguous_future_prices():
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z", "2025-01-01T00:00:03Z",
            "2025-01-01T00:00:04Z", "2025-01-01T00:00:05Z",
        ]),
        "book_valid": True,
        "mid_price": [100.0, 101.0, 103.0, 102.0, 104.0],
    })
    labels = make_direction_labels(frame, horizon_seconds=1, delta=0.5)
    assert labels.iloc[0] == "up"
    assert pd.isna(labels.iloc[1])
    assert labels.iloc[2] == "down"
    assert labels.iloc[3] == "up"
    assert pd.isna(labels.iloc[4])


def test_direction_boundaries_are_flat_and_nonfinite_delta_is_rejected():
    frame = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="s"),
        "book_valid": True,
        "mid_price": [100.0, 101.0, 100.0, 98.9, 98.9],
    })
    labels = make_direction_labels(frame, horizon_seconds=1, delta=1.0)
    assert labels.iloc[0] == "flat"
    assert labels.iloc[1] == "flat"
    with pytest.raises(ValueError, match="delta"):
        make_direction_labels(frame, horizon_seconds=1, delta=float("nan"))


def test_deeplob_sequences_drop_windows_crossing_missing_seconds():
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z", "2025-01-01T00:00:02Z",
            "2025-01-01T00:00:04Z", "2025-01-01T00:00:05Z",
        ]),
        "book_valid": True,
        "a": np.arange(5, dtype=float),
        "b": np.arange(5, dtype=float) + 10,
    })
    sequences, endpoints = build_lob_sequences(frame, [2, 4], lookback=3, columns=("a", "b"))
    assert sequences.shape == (1, 3, 2)
    assert endpoints.tolist() == [2]


def test_deeplob_price_channels_are_normalized_to_each_snapshot_mid():
    rows = []
    for timestamp, mid in zip(pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="s"), (100.0, 110.0)):
        row = {"timestamp_utc": timestamp, "book_valid": True}
        for level in range(1, 11):
            row[f"bid_price_{level}"] = mid * (1 - 0.01 * level)
            row[f"ask_price_{level}"] = mid * (1 + 0.01 * level)
            row[f"bid_qty_{level}"] = 1.0
            row[f"ask_qty_{level}"] = 1.0
        rows.append(row)
    sequences, endpoints = build_lob_sequences(pd.DataFrame(rows), [1], lookback=2)
    assert sequences.shape == (1, 2, 40)
    assert tuple(DEEPLOB_INPUT_COLUMNS[:2]) == ("bid_price_1", "bid_price_2")
    assert np.isclose(sequences[0, 0, 0], -100.0)
    assert endpoints.tolist() == [1]


def test_paired_mcnemar_and_minute_blocked_brier_dm_are_returned():
    rng = np.random.default_rng(19)
    truth = rng.integers(0, 3, size=3600)
    prediction = rng.integers(0, 3, size=3600)
    baseline = rng.integers(0, 3, size=3600)
    probability = np.eye(3)[prediction] * 0.8 + 0.2 / 3
    baseline_probability = np.eye(3)[baseline] * 0.8 + 0.2 / 3
    timestamps = pd.date_range("2026-11-23T00:00:00Z", periods=len(truth), freq="s")
    result = paired_classification_tests(
        truth, prediction, baseline, probability, baseline_probability,
        horizon_seconds=5, baseline_name="logistic", timestamps=timestamps,
    )
    assert result["baseline"] == "logistic"
    assert result["mcnemar"]["p_value_exact_two_sided"] >= 0
    assert result["diebold_mariano_brier"]["n_blocks"] == 60
    assert result["diebold_mariano_brier"]["hac_lag_unit"] == "minutes"
    assert result["diebold_mariano_brier"]["hac_lags"] >= 10


def test_minute_blocked_dm_withholds_asymptotic_p_value_below_30_blocks():
    truth = np.asarray([0, 1] * 600)
    prediction = truth.copy()
    baseline = np.roll(truth, 1)
    probabilities = np.column_stack([1 - prediction, prediction]).astype(float)
    baseline_probabilities = np.column_stack([1 - baseline, baseline]).astype(float)
    timestamps = pd.date_range("2026-11-23T00:00:00Z", periods=len(truth), freq="s")
    result = paired_classification_tests(
        truth, prediction, baseline, probabilities, baseline_probabilities,
        horizon_seconds=5, timestamps=timestamps,
    )
    assert result["diebold_mariano_brier"]["n_blocks"] == 20
    assert result["diebold_mariano_brier"]["statistic"] is None
    assert result["diebold_mariano_brier"]["p_value_two_sided_normal"] is None


def test_weekly_folds_are_frozen_to_the_development_period_and_embargoed():
    folds = expanding_week_folds("2026-11-02T00:00:00Z", "2026-11-22T23:59:59Z")
    assert folds == ("2026-11-02", "2026-11-09", "2026-11-16")
    timestamps = pd.Series(pd.to_datetime([
        "2026-11-01T23:54:59Z", "2026-11-01T23:55:00Z", "2026-11-02T00:00:00Z",
    ]))
    kept = apply_embargo(np.array([True, True, False]), timestamps, pd.Timestamp("2026-11-02T00:00:00Z"), 300)
    assert kept.tolist() == [True, False, False]


def test_deeplob_torch_fit_predict_proba_smoke():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(60, 4, 40)).astype(np.float32)
    y = np.asarray([0, 1] * 30)
    model = DeepLOBClassifier(
        lookback=4, hidden_size=8, batch_size=16, max_epochs=1, patience=1,
    ).fit(x, y)
    probabilities = model.predict_proba(x[:8])
    assert probabilities.shape == (8, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_vector_price_impact_is_missing_if_one_book_side_cannot_fill_notional():
    row = {"timestamp_utc": "2025-01-01T00:00:00Z", "book_valid": True}
    for level in range(1, 21):
        row[f"bid_price_{level}"] = 100 - level
        row[f"bid_qty_{level}"] = 0.001
        row[f"ask_price_{level}"] = 101 + level
        row[f"ask_qty_{level}"] = 100
    result = build_classification_features(pd.DataFrame([row]), impact_notional_usdt=1000)
    assert np.isnan(result.loc[0, "impact_sell_bps"])
    assert np.isnan(result.loc[0, "price_impact_bps"])


def test_daily_feature_build_keeps_the_previous_day_warmup_tail():
    timestamps = pd.date_range("2025-01-01T23:54:59Z", periods=302, freq="s")
    rows = []
    for timestamp in timestamps:
        row = {
            "timestamp_utc": timestamp,
            "symbol": "BTCUSDT",
            "book_valid": True,
            "ofi_1s": 0.0,
            "aggressive_buy_volume_1s": 1.0,
            "aggressive_sell_volume_1s": 1.0,
        }
        for level in range(1, 21):
            row[f"bid_price_{level}"] = 100 - (level - 1)
            row[f"bid_qty_{level}"] = 100.0
            row[f"ask_price_{level}"] = 102 + (level - 1)
            row[f"ask_qty_{level}"] = 100.0
        rows.append(row)
    raw = pd.DataFrame(rows)
    settings = Settings(
        lob_feature_lookback_seconds=300,
        lob_tick_size_by_symbol={"BTCUSDT": 2.0},
        lob_stress_impact_notional_by_symbol={"BTCUSDT": 50000.0},
    )
    features, tail = build_partition_features(
        raw_frame=raw.iloc[-1:], previous_tail=raw.iloc[:-1], date="2025-01-02",
        symbol="BTCUSDT", settings=settings,
    )
    assert len(tail) == 301
    assert len(features) == 1
    assert np.isfinite(features.iloc[0]["rv_300s"])
    assert features.iloc[0]["ofi_10s"] == 0
    assert features.iloc[0]["trade_imbalance_10s"] == 0
