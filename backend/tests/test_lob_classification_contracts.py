import numpy as np
import pandas as pd

from app.core.config import Settings
from app.evaluation.lob_classification import paired_classification_tests
from app.features.classification_features import build_classification_features
from app.labels.direction import make_direction_labels
from app.labels.stress import (
    fit_spread_only_threshold,
    fit_stress_thresholds,
    make_spread_only_labels,
    monthly_positive_rates,
)
from app.models.deeplob_classifier import DEEPLOB_INPUT_COLUMNS, build_lob_sequences
from scripts.build_lob_classification_dataset import build_partition_features


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


def test_paired_mcnemar_and_horizon_adjusted_brier_dm_are_returned():
    truth = np.asarray([0, 1, 2, 0, 1, 2])
    prediction = np.asarray([0, 1, 1, 0, 2, 2])
    baseline = np.asarray([1, 1, 2, 0, 1, 0])
    probability = np.eye(3)[np.asarray([0, 1, 1, 0, 2, 2])]
    baseline_probability = np.eye(3)[baseline]
    result = paired_classification_tests(
        truth, prediction, baseline, probability, baseline_probability,
        horizon_seconds=5, baseline_name="logistic",
    )
    assert result["baseline"] == "logistic"
    assert result["mcnemar"]["p_value_exact_two_sided"] >= 0
    assert result["diebold_mariano_brier"]["hac_lags"] == 4


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
