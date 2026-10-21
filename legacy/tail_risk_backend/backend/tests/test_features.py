import pandas as pd

from app.features.realized_volatility import daily_realized_volatility
from app.features.returns import add_returns


def test_returns_and_daily_realized_variance_are_time_ordered():
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 3,
        "open_time": pd.to_datetime(["2020-01-01T00:10Z", "2020-01-01T00:00Z", "2020-01-01T00:05Z"], utc=True),
        "close": [102.0, 100.0, 101.0],
        "high": [102.0, 100.0, 101.0], "low": [102.0, 100.0, 101.0],
        "volume": [1.0, 1.0, 1.0], "quote_volume": [1.0, 1.0, 1.0],
    })
    returns = add_returns(frame)
    assert returns.iloc[0]["open_time"] == pd.Timestamp("2020-01-01T00:00Z")
    rv = daily_realized_volatility(frame)
    assert rv.loc[0, "observations"] == 2
    # A partial date is diagnostic only and must never become a training target.
    assert pd.isna(rv.loc[0, "realized_variance"])
    assert rv.loc[0, "rv_partial"] > 0
    assert not rv.loc[0, "is_complete_day"]
