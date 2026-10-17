"""Naive variance forecasts used as frozen benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_forecast(realized_variance: pd.Series, horizon: int = 1) -> float:
    values = pd.to_numeric(realized_variance, errors="coerce").dropna()
    if values.empty or horizon < 1:
        raise ValueError("Need observed RV and a positive horizon")
    return float(max(values.iloc[-1], 0.0))


def rolling_mean_forecast(realized_variance: pd.Series, window: int = 30, horizon: int = 1) -> float:
    values = pd.to_numeric(realized_variance, errors="coerce").dropna().tail(window)
    if values.empty or horizon < 1:
        raise ValueError("Need observations and a positive horizon")
    return float(max(values.mean(), 0.0))


def ewma_forecast(daily_returns: pd.Series, decay: float = 0.94, horizon: int = 1) -> float:
    """Return-variance EWMA with configurable daily decay."""
    values = pd.to_numeric(daily_returns, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0 or not 0 < decay < 1 or horizon < 1:
        raise ValueError("Need returns, decay in (0, 1), and a positive horizon")
    initial = min(30, len(values))
    variance = float(np.mean(values[:initial] ** 2))
    for value in values[initial:]:
        variance = decay * variance + (1 - decay) * float(value**2)
    return max(variance, np.finfo(float).tiny)
