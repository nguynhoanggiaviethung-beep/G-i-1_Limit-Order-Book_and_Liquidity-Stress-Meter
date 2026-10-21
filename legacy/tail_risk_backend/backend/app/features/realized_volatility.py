"""Daily realized-variation measures with explicit completeness flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.returns import add_returns


def daily_realized_volatility(frame: pd.DataFrame, bars_per_day: int = 288) -> pd.DataFrame:
    """Compute RV, BPV, jump, semivariances, quarticity, and OHLCV by UTC day.

    Incomplete days keep a partial-RV diagnostic but have NaN in the RV target
    fields, preventing them from entering model features or target windows.
    Missing intervals are never filled and returns spanning gaps are NaN.
    """
    if frame.empty:
        return pd.DataFrame()
    data = add_returns(frame)
    data["utc_date"] = pd.to_datetime(data["open_time"], utc=True).dt.floor("D")
    rows: list[dict] = []
    for (symbol, day), group in data.groupby(["symbol", "utc_date"], sort=True):
        returns = group["log_return"].to_numpy(dtype=float)
        valid = np.isfinite(returns)
        observed = int(valid.sum())
        complete = len(group) == bars_per_day and observed == bars_per_day
        partial_rv = float(np.square(returns[valid]).sum())
        adjacent = valid[1:] & valid[:-1]
        bpv_partial = float(
            (np.pi / 2) * (bars_per_day / max(bars_per_day - 1, 1))
            * np.sum(np.abs(returns[1:][adjacent]) * np.abs(returns[:-1][adjacent]))
        )
        positive = returns[valid & (returns > 0)]
        negative = returns[valid & (returns < 0)]
        quarticity_sum = float(np.power(returns[valid], 4).sum())
        rv = partial_rv if complete else np.nan
        bpv = bpv_partial if complete else np.nan
        record = {
            "symbol": symbol,
            "utc_date": day,
            "n_bars": int(len(group)),
            "n_obs": observed,
            "observations": observed,
            "is_complete_day": bool(complete),
            "rv_partial": partial_rv,
            "rv": rv,
            "realized_variance": rv,
            "realized_volatility": float(np.sqrt(rv)) if complete else np.nan,
            "bpv": bpv,
            "jump": max(rv - bpv, 0.0) if complete else np.nan,
            "rs_pos": float(np.square(positive).sum()) if complete else np.nan,
            "rs_neg": float(np.square(negative).sum()) if complete else np.nan,
            "rq": float((bars_per_day / 3) * quarticity_sum) if complete else np.nan,
            "close_ret": float(returns[valid].sum()) if complete else np.nan,
            "daily_return": float(returns[valid].sum()) if complete else np.nan,
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group["close"].iloc[-1]),
            "volume": float(group["volume"].sum()),
            "quote_volume": float(group["quote_volume"].sum()),
            "number_of_trades": float(group["number_of_trades"].sum()) if "number_of_trades" in group else np.nan,
            "taker_buy_base_volume": float(group["taker_buy_base_volume"].sum()) if "taker_buy_base_volume" in group else np.nan,
            "taker_buy_quote_volume": float(group["taker_buy_quote_volume"].sum()) if "taker_buy_quote_volume" in group else np.nan,
        }
        record["hl_range"] = (
            float(np.log(record["high"] / record["low"])) if record["low"] > 0 else np.nan
        )
        rows.append(record)

    result = pd.DataFrame(rows)
    complete_frames = []
    for symbol, group in result.groupby("symbol", sort=False):
        group = group.set_index("utc_date").sort_index()
        calendar = pd.date_range(group.index.min(), group.index.max(), freq="D", tz="UTC", name="utc_date")
        group = group.reindex(calendar)
        group["symbol"] = symbol
        group["utc_date"] = calendar
        group["n_bars"] = group["n_bars"].fillna(0).astype("int64")
        group["n_obs"] = group["n_obs"].fillna(0).astype("int64")
        group["observations"] = group["observations"].fillna(0).astype("int64")
        group["is_complete_day"] = group["is_complete_day"].fillna(False).astype(bool)
        group["available_at_utc"] = group["utc_date"] + pd.Timedelta(days=1)
        complete_frames.append(group.reset_index(drop=True))
    result = pd.concat(complete_frames, ignore_index=True)
    columns = [
        "symbol", "utc_date", "available_at_utc", "n_bars", "n_obs", "observations",
        "is_complete_day", "rv_partial", "rv", "realized_variance", "realized_volatility",
        "bpv", "jump", "rs_pos", "rs_neg", "rq", "close_ret", "daily_return",
        "high", "low", "close", "hl_range", "volume", "quote_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    return result[columns].sort_values(["symbol", "utc_date"]).reset_index(drop=True)
