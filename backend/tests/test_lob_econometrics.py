from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from scripts.run_lob_econometrics import blocked_dm, future_return_bps, newey_west_ols


def test_newey_west_matches_statsmodels_hac():
    rng = np.random.default_rng(3)
    n = 4_000
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    noise = np.convolve(rng.normal(size=n + 9), np.ones(10) / 10, mode="valid")  # serially correlated errors
    y = x @ np.array([0.1, 0.5, -0.2, 0.0]) + noise
    ours = newey_west_ols(x, y, lags=12)
    reference = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 12, "use_correction": True})
    np.testing.assert_allclose(ours["beta"], reference.params, rtol=1e-10)
    np.testing.assert_allclose(ours["se"], reference.bse, rtol=1e-10)
    assert ours["r2"] == pytest.approx(reference.rsquared, rel=1e-10)


def test_future_return_needs_a_contiguous_valid_path():
    times = pd.date_range("2026-06-01", periods=8, freq="1s").to_numpy()
    times[5:] += np.timedelta64(3, "s")
    frame = pd.DataFrame({
        "timestamp_utc": pd.DatetimeIndex(times).tz_localize("UTC"),
        "book_valid": [True, True, True, False, True, True, True, True],
        "mid_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
    })
    returns = future_return_bps(frame, 2)
    assert returns[0] == pytest.approx((102.0 / 100.0 - 1) * 10000)
    # Origins 1-2 cross the invalid second, origin 4 crosses the gap, and 6-7 have no future.
    assert np.isnan(returns[[1, 2, 3, 4, 6, 7]]).all()
    assert returns[5] == pytest.approx((107.0 / 105.0 - 1) * 10000)


def test_blocked_dm_detects_a_consistent_loss_difference():
    rng = np.random.default_rng(9)
    minutes = np.repeat(np.arange(200), 60)
    better = blocked_dm(0.2 + rng.normal(0, 1, len(minutes)), minutes)
    assert better["statistic"] > 3 and better["p_value"] < 0.01
    assert blocked_dm(np.ones(100), np.arange(100) // 60)["statistic"] is None  # too few minutes
