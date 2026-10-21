"""Ex-ante bull, bear, and stress labels using trailing information only."""

from __future__ import annotations

import pandas as pd


def label_regimes(daily: pd.DataFrame, rules: dict | None = None) -> pd.Series:
    rules = rules or {}
    data = daily.sort_values("utc_date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    rv = pd.to_numeric(data["rv"], errors="coerce")
    returns = pd.to_numeric(data.get("close_ret", data.get("daily_return")), errors="coerce")
    sma200 = close.rolling(200, min_periods=200).mean()
    drawdown90 = close / close.rolling(90, min_periods=90).max() - 1
    drawdown365 = close / close.rolling(365, min_periods=365).max() - 1
    rv_window = int(rules.get("stress_rv_window_days", 365))
    rv_cutoff = rv.rolling(rv_window, min_periods=rv_window).quantile(
        float(rules.get("stress_rv_quantile", 0.95))
    )
    stress = (
        (rv > rv_cutoff)
        | (returns <= float(rules.get("stress_daily_return_floor", -10.0)))
    ).fillna(False)
    bear = (
        (close < sma200)
        | (drawdown365 <= float(rules.get("bear_drawdown_365d_floor", -0.30)))
    ).fillna(False)
    bull = (
        (close > sma200)
        & (drawdown90 > float(rules.get("bull_drawdown_90d_floor", -0.10)))
    ).fillna(False)
    labels = pd.Series("neutral", index=data.index, dtype="object")
    labels.loc[bull] = "bull"
    labels.loc[bear] = "bear"
    labels.loc[stress] = "stress"
    return labels.reindex(daily.index).rename("regime")
