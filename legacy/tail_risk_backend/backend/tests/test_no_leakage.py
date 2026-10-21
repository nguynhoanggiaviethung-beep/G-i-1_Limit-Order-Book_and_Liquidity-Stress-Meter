import numpy as np
import pandas as pd
import pytest

from app.datasets.leakage_checks import assert_no_target_leakage, feature_columns
from app.features.lob_targets import build_lob_targets


def _samples(times=None, valid=None):
    if times is None:
        times = pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="s")
    valid = valid or [True] * len(times)
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "sample_time_utc": times,
        "book_valid": valid,
        "mid_price": [100, 101, 102, 103, 104][:len(times)],
        "obi_qty_5": [0.1] * len(times),
    })


def test_lob_rv_target_uses_future_mid_returns_and_not_future_features():
    samples = _samples()
    dataset = build_lob_targets(samples, horizon_samples=2, sampling_interval_seconds=1)
    expected = np.log(101 / 100) ** 2 + np.log(102 / 101) ** 2
    assert dataset.loc[0, "target_valid"]
    assert np.isclose(dataset.loc[0, "target_rv"], expected)
    assert dataset.loc[0, "target_end_utc"] > dataset.loc[0, "sample_time_utc"]
    assert "target_rv" not in feature_columns(dataset)
    assert_no_target_leakage(dataset)


def test_target_is_suppressed_when_future_sample_grid_has_a_gap():
    times = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z", "2025-01-01T00:00:03Z",
        "2025-01-01T00:00:04Z", "2025-01-01T00:00:05Z",
    ], utc=True)
    dataset = build_lob_targets(_samples(times), horizon_samples=2, sampling_interval_seconds=1)
    assert not dataset.loc[0, "target_valid"]
    assert dataset.loc[2, "target_valid"]


def test_target_is_suppressed_when_book_was_invalid_in_horizon():
    dataset = build_lob_targets(_samples(valid=[True, False, True, True, True]), 2, 1)
    assert not dataset.loc[0, "target_valid"]
    assert dataset.loc[2, "target_valid"]


def test_leakage_check_rejects_target_available_at_or_before_origin():
    dataset = build_lob_targets(_samples(), 1, 1)
    dataset.loc[0, "target_end_utc"] = dataset.loc[0, "sample_time_utc"]
    with pytest.raises(ValueError, match="strictly after"):
        assert_no_target_leakage(dataset)
