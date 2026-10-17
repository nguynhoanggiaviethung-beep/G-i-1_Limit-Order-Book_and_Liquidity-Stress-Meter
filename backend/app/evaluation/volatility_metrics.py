"""Losses and calibration diagnostics for realized-volatility forecasts."""

from __future__ import annotations

import numpy as np


def qlike_loss(actual, forecast, floor: float | None = None) -> np.ndarray:
    """Patton QLIKE per observation, including the actual-only constant."""
    y = np.asarray(actual, dtype=float)
    h = np.asarray(forecast, dtype=float)
    if y.shape != h.shape:
        raise ValueError("actual and forecast must have the same shape")
    positive = y[np.isfinite(y) & (y > 0)]
    epsilon = floor if floor is not None else (
        np.quantile(positive, 0.01) if positive.size else np.finfo(float).tiny
    )
    h = np.maximum(h, np.maximum(np.asarray(epsilon, dtype=float), np.finfo(float).tiny))
    return y / h + np.log(h) - np.log(np.maximum(y, np.finfo(float).tiny)) - 1


def _mincer_zarnowitz(actual: np.ndarray, forecast: np.ndarray, hac_lags: int) -> dict:
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"intercept": None, "slope": None, "joint_pvalue": None}
    fit = sm.OLS(actual, sm.add_constant(forecast)).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(max(int(hac_lags), 0), max(len(actual) - 1, 0))}
    )
    test = fit.f_test((np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 1.0])))
    return {
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "joint_pvalue": float(np.asarray(test.pvalue).squeeze()),
    }


def volatility_metrics(
    actual,
    forecast,
    benchmark=None,
    train_floor: float | None = None,
    hac_lags: int = 0,
) -> dict[str, float | int | dict]:
    """Report RV, volatility, and log-RV losses and calibration statistics."""
    y = np.asarray(actual, dtype=float)
    raw_h = np.asarray(forecast, dtype=float)
    if y.shape != raw_h.shape:
        raise ValueError("actual and forecast must have the same shape")
    valid = np.isfinite(y) & np.isfinite(raw_h) & (y >= 0)
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        if b.shape != y.shape:
            raise ValueError("benchmark must align with actual and forecast")
        valid &= np.isfinite(b)
    y, raw_h = y[valid], raw_h[valid]
    if benchmark is not None:
        b = b[valid]
    if not y.size:
        raise ValueError("No finite actual/forecast pairs")
    positive = y[y > 0]
    if train_floor is None:
        floors = np.full(len(y), np.quantile(positive, 0.01) if positive.size else np.finfo(float).tiny)
    else:
        floor_values = np.asarray(train_floor, dtype=float)
        if floor_values.ndim == 0:
            floors = np.full(len(y), float(floor_values))
        else:
            if floor_values.shape != valid.shape:
                raise ValueError("train_floor must be scalar or align with input arrays")
            floors = floor_values[valid]
        floors = np.maximum(floors, np.finfo(float).tiny)
    h = np.maximum(raw_h, floors)
    result: dict = {
        "n_obs": int(len(y)),
        "n_forecast_floored": int(np.sum(raw_h < floors)),
        "floor_mean": float(np.mean(floors)),
        "mae": float(np.mean(np.abs(y - h))),
        "mse": float(np.mean((y - h) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - h) ** 2))),
        "qlike": float(np.mean(qlike_loss(y, h, floors))),
        "volatility_mae": float(np.mean(np.abs(np.sqrt(y) - np.sqrt(h)))),
        "volatility_rmse": float(np.sqrt(np.mean((np.sqrt(y) - np.sqrt(h)) ** 2))),
    }
    positive_pairs = (y > 0) & (h > 0)
    result["log_rv_mae"] = float(np.mean(np.abs(np.log(y[positive_pairs]) - np.log(h[positive_pairs])))) if positive_pairs.any() else float("nan")
    result["log_rv_rmse"] = float(np.sqrt(np.mean((np.log(y[positive_pairs]) - np.log(h[positive_pairs])) ** 2))) if positive_pairs.any() else float("nan")
    if benchmark is not None:
        denominator = float(np.sum((y - b) ** 2))
        result["r2_oos_vs_benchmark"] = (
            float(1 - np.sum((y - h) ** 2) / denominator) if denominator > 0 else float("nan")
        )
    result["mincer_zarnowitz"] = _mincer_zarnowitz(y, h, hac_lags)
    return result
