"""Diebold-Mariano tests with HAC variance and HLN small-sample correction."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def _long_run_variance(values: np.ndarray, lags: int) -> float:
    centered = values - values.mean()
    n = len(centered)
    variance = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2 * weight * covariance
    return max(variance, 0.0)


def dm_test(loss_a, loss_b, horizon: int = 1, hac_lags: int | None = None, hln: bool = True) -> dict:
    """Two-sided DM test; positive mean difference means model A loses more."""
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss arrays must be aligned")
    valid = np.isfinite(a) & np.isfinite(b)
    difference = a[valid] - b[valid]
    n = len(difference)
    if n < 3:
        raise ValueError("DM requires at least three paired observations")
    lag = max(int(hac_lags if hac_lags is not None else horizon - 1), horizon - 1)
    lag = min(lag, n - 1)
    lrv = _long_run_variance(difference, lag)
    if lrv <= 0:
        return {
            "n_obs": n, "hac_lags": lag, "mean_loss_difference": float(difference.mean()),
            "statistic": 0.0, "pvalue": 1.0, "hln_corrected": bool(hln),
        }
    statistic = float(difference.mean() / np.sqrt(lrv / n))
    if hln:
        correction_squared = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        statistic *= float(np.sqrt(max(correction_squared, 0.0)))
    pvalue = float(2 * student_t.sf(abs(statistic), df=n - 1))
    return {
        "n_obs": n,
        "n_independent_approx": max(n // horizon, 1),
        "hac_lags": lag,
        "mean_loss_difference": float(difference.mean()),
        "statistic": statistic,
        "pvalue": pvalue,
        "hln_corrected": bool(hln),
    }


def clark_west_test(loss_a, loss_b, horizon: int = 1, hac_lags: int | None = None) -> dict:
    """Clark-West adjusted MSPE test for nested forecasts."""
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss arrays must be aligned")
    # Inputs are squared forecast errors; adjustment is represented by the
    # positive reduction in benchmark MSPE, with HAC inference on the series.
    valid = np.isfinite(a) & np.isfinite(b)
    adjusted = b[valid] - a[valid]
    n = len(adjusted)
    if n < 3:
        raise ValueError("Clark-West requires at least three paired observations")
    lag = max(horizon - 1, int(hac_lags or 0))
    lrv = _long_run_variance(adjusted, min(lag, n - 1))
    statistic = float(adjusted.mean() / np.sqrt(max(lrv, np.finfo(float).tiny) / n))
    pvalue = float(student_t.sf(statistic, df=n - 1))
    return {"n_obs": n, "mean_adjusted_loss": float(adjusted.mean()), "statistic": statistic, "one_sided_pvalue": pvalue}
