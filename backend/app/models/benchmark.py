"""Naive rolling-mean benchmark for realized variance."""

import pandas as pd


def rolling_mean_forecast(realized_variance: pd.Series, window: int = 22, horizon: int = 1) -> float:
    values = pd.to_numeric(realized_variance, errors="coerce").dropna().tail(window)
    if values.empty or horizon < 1:
        raise ValueError("Need observations and a positive horizon")
    return float(values.mean())
