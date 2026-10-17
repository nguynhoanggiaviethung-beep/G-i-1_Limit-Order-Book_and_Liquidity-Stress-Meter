"""Realized variance and volatility aggregated from intraday returns."""

import numpy as np
import pandas as pd

from app.features.returns import add_returns


def daily_realized_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate squared 5-minute log returns by UTC date; RV is in percent squared."""
    data = add_returns(frame)
    data["utc_date"] = pd.to_datetime(data["open_time"], utc=True).dt.floor("D")
    data["squared_return"] = data["log_return"].pow(2)
    grouped = data.groupby(["symbol", "utc_date"], as_index=False).agg(
        realized_variance=("squared_return", "sum"),
        observations=("log_return", "count"),
        daily_return=("log_return", "sum"),
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
    )
    grouped["realized_volatility"] = np.sqrt(grouped["realized_variance"])
    return grouped
