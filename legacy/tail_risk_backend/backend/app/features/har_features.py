"""Point-in-time HAR and robustness feature windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def har_features(daily: pd.DataFrame, windows: tuple[int, int] = (7, 30)) -> pd.DataFrame:
    """Build predictors known after each UTC day closes."""
    weekly, monthly = windows
    ordered = daily.sort_values("utc_date").copy()
    rv = pd.to_numeric(ordered["rv"], errors="coerce")
    output = pd.DataFrame(index=ordered.index)
    output["rv_lag_1"] = rv
    output[f"rv_mean_{weekly}"] = rv.rolling(weekly, min_periods=weekly).mean()
    output[f"rv_mean_{monthly}"] = rv.rolling(monthly, min_periods=monthly).mean()
    output["rv_mean_5"] = rv.rolling(5, min_periods=5).mean()
    output["rv_mean_22"] = rv.rolling(22, min_periods=22).mean()
    output["rv_max_7"] = rv.rolling(7, min_periods=7).max()
    output["ln_rv_lag_1"] = np.log(rv.where(rv > 0))
    output[f"ln_rv_mean_{weekly}"] = np.log(output[f"rv_mean_{weekly}"].where(output[f"rv_mean_{weekly}"] > 0))
    output[f"ln_rv_mean_{monthly}"] = np.log(output[f"rv_mean_{monthly}"].where(output[f"rv_mean_{monthly}"] > 0))
    for column in (
        "bpv", "jump", "rs_pos", "rs_neg", "rq", "close_ret", "hl_range",
        "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if column in ordered:
            output[column] = ordered[column]
            component = pd.to_numeric(ordered[column], errors="coerce")
            for window in set((weekly, monthly, 5, 22)):
                if column in {"bpv", "jump", "rs_pos", "rs_neg"}:
                    output[f"{column}_mean_{window}"] = component.rolling(window, min_periods=window).mean()
    if "volume" in output:
        output["ln_volume"] = np.log1p(output["volume"].where(output["volume"] >= 0))
    if "quote_volume" in output:
        output["ln_turnover"] = np.log1p(output["quote_volume"].where(output["quote_volume"] >= 0))
    if "number_of_trades" in output:
        output["ln_trade_count"] = np.log1p(output["number_of_trades"].where(output["number_of_trades"] >= 0))
    dates = pd.to_datetime(ordered["utc_date"], utc=True)
    output["day_of_week"] = dates.dt.dayofweek.to_numpy()
    output["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int).to_numpy()
    if {"rq", "rv_lag_1"}.issubset(output.columns):
        output["harq_adjustment"] = np.sqrt(output["rq"].clip(lower=0)) * output["rv_lag_1"]
    return output
