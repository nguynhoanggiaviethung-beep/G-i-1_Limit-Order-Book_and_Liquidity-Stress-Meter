"""Point-in-time return features from normalized OHLCV data."""

import numpy as np
import pandas as pd


def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("open_time").copy()
    result["log_return"] = 100 * result.groupby("symbol", sort=False)["close"].transform(lambda x: np.log(x / x.shift(1)))
    result["simple_return"] = result.groupby("symbol", sort=False)["close"].pct_change() * 100
    return result
