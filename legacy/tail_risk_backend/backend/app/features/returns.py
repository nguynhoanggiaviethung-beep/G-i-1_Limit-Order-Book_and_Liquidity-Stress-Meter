"""Point-in-time return features from normalized OHLCV data."""

import numpy as np
import pandas as pd


def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["symbol", "open_time"]).copy()
    result["open_time"] = pd.to_datetime(result["open_time"], utc=True)
    grouped = result.groupby("symbol", sort=False)
    previous_close = grouped["close"].shift(1)
    gaps = grouped["open_time"].diff()
    expected = (
        result.groupby("symbol", sort=False)["interval"]
        .transform(lambda values: values.iloc[0] if len(values) else "5m")
        .map(pd.Timedelta)
        if "interval" in result
        else pd.Series(pd.Timedelta("5m"), index=result.index)
    )
    contiguous = gaps.eq(expected)
    result["log_return"] = (100 * np.log(result["close"] / previous_close)).where(contiguous)
    result["simple_return"] = (100 * (result["close"] / previous_close - 1)).where(contiguous)
    return result
