"""Build one-second, point-in-time features from normalized full-depth grids."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _window_sum(values: pd.Series, segments: pd.Series, window: int) -> pd.Series:
    return values.groupby(segments, sort=False).transform(
        lambda part: part.rolling(window, min_periods=window).sum()
    )


def _impact(prices: np.ndarray, quantities: np.ndarray, mid: np.ndarray, notional: float, buy: bool) -> np.ndarray:
    level_notional = prices * quantities
    cumulative_notional = np.cumsum(level_notional, axis=1)
    cumulative_quantity = np.cumsum(quantities, axis=1)
    filled = cumulative_notional >= notional
    complete = filled.any(axis=1)
    level = np.argmax(filled, axis=1)
    row = np.arange(len(prices))
    before_notional = np.where(level > 0, cumulative_notional[row, np.maximum(level - 1, 0)], 0.0)
    before_quantity = np.where(level > 0, cumulative_quantity[row, np.maximum(level - 1, 0)], 0.0)
    level_price = prices[row, level]
    quantity = before_quantity + np.divide(
        notional - before_notional,
        level_price,
        out=np.zeros_like(mid, dtype=float),
        where=level_price > 0,
    )
    vwap = np.divide(notional, quantity, out=np.full_like(mid, np.nan, dtype=float), where=quantity > 0)
    impact = (vwap - mid) / mid * 10000.0 if buy else (mid - vwap) / mid * 10000.0
    impact[~complete | ~np.isfinite(mid) | (mid <= 0)] = np.nan
    return impact


def build_classification_features(
    grid: pd.DataFrame,
    impact_notional_usdt: float = 100000.0,
    tick_size: float | None = None,
) -> pd.DataFrame:
    """Return the input grid with derived LOB, OFI/trade, and rolling RV features."""
    if grid.empty:
        return grid.copy()
    frame = grid.copy()
    time_column = "timestamp_utc" if "timestamp_utc" in frame else "sample_time_utc"
    if time_column not in frame:
        raise ValueError("Grid must have timestamp_utc or sample_time_utc")
    frame["timestamp_utc"] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc", kind="mergesort")
    duplicate_count = int(frame.duplicated("timestamp_utc", keep="last").sum())
    if duplicate_count:
        logger.warning(
            "Resolved %d duplicate order-book feature timestamps by keeping the last input row",
            duplicate_count,
        )
    frame = frame.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
    if "book_valid" not in frame:
        frame["book_valid"] = True
    if frame["book_valid"].dtype == object:
        frame["book_valid"] = frame["book_valid"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    else:
        frame["book_valid"] = frame["book_valid"].fillna(False).astype(bool)
    bid_prices = [f"bid_price_{level}" for level in range(1, 21)]
    bid_qty = [f"bid_qty_{level}" for level in range(1, 21)]
    ask_prices = [f"ask_price_{level}" for level in range(1, 21)]
    ask_qty = [f"ask_qty_{level}" for level in range(1, 21)]
    required = [*bid_prices, *bid_qty, *ask_prices, *ask_qty]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Normalized replay grid is missing full-depth columns: {', '.join(missing[:4])}")

    bp = frame[bid_prices].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bq = frame[bid_qty].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    ap = frame[ask_prices].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aq = frame[ask_qty].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    mid = (bp[:, 0] + ap[:, 0]) / 2.0
    spread = ap[:, 0] - bp[:, 0]
    top_total = bq[:, 0] + aq[:, 0]
    microprice = np.divide(ap[:, 0] * bq[:, 0] + bp[:, 0] * aq[:, 0], top_total, out=mid.copy(), where=top_total > 0)
    frame["best_bid"] = bp[:, 0]
    frame["best_bid_qty"] = bq[:, 0]
    frame["best_ask"] = ap[:, 0]
    frame["best_ask_qty"] = aq[:, 0]
    frame["mid_price"] = mid
    frame["absolute_spread"] = spread
    frame["relative_spread_bps"] = np.divide(spread, mid, out=np.full_like(mid, np.nan), where=mid > 0) * 10000.0
    configured_tick = pd.to_numeric(frame.get("tick_size"), errors="coerce") if "tick_size" in frame else None
    if configured_tick is not None:
        tick_values = configured_tick.to_numpy(dtype=float)
        if tick_size is not None:
            tick_values = np.where(np.isfinite(tick_values) & (tick_values > 0), tick_values, tick_size)
    elif tick_size is not None:
        tick_values = np.full(len(frame), float(tick_size), dtype=float)
    else:
        tick_values = np.full(len(frame), np.nan, dtype=float)
    frame["tick_size"] = tick_values
    frame["spread_ticks"] = np.divide(spread, tick_values, out=np.full_like(spread, np.nan), where=tick_values > 0)
    frame["microprice"] = microprice
    frame["microprice_minus_mid_bps"] = np.divide(microprice - mid, mid, out=np.full_like(mid, np.nan), where=mid > 0) * 10000.0
    for depth in (1, 5, 10, 20):
        bid_depth = bq[:, :depth].sum(axis=1)
        ask_depth = aq[:, :depth].sum(axis=1)
        total = bid_depth + ask_depth
        frame[f"bid_depth_qty_{depth}"] = bid_depth
        frame[f"ask_depth_qty_{depth}"] = ask_depth
        frame[f"depth_qty_{depth}"] = total
        frame[f"bid_depth_notional_{depth}"] = (bp[:, :depth] * bq[:, :depth]).sum(axis=1)
        frame[f"ask_depth_notional_{depth}"] = (ap[:, :depth] * aq[:, :depth]).sum(axis=1)
        frame[f"depth_notional_{depth}"] = frame[f"bid_depth_notional_{depth}"] + frame[f"ask_depth_notional_{depth}"]
        frame[f"obi_qty_{depth}"] = np.divide(bid_depth - ask_depth, total, out=np.full_like(total, np.nan), where=total > 0)

    # Depth slope: resting notional per basis point of distance from mid out to level 20.
    # A steep (large) slope means a deep, resilient side; a flat one means thin liquidity.
    with np.errstate(divide="ignore", invalid="ignore"):
        bid_distance_bps = (mid - bp[:, 19]) / mid * 10000.0
        ask_distance_bps = (ap[:, 19] - mid) / mid * 10000.0
    bid_slope = np.divide(frame["bid_depth_notional_20"].to_numpy(), bid_distance_bps, out=np.full_like(mid, np.nan), where=bid_distance_bps > 0)
    ask_slope = np.divide(frame["ask_depth_notional_20"].to_numpy(), ask_distance_bps, out=np.full_like(mid, np.nan), where=ask_distance_bps > 0)
    frame["bid_depth_slope_20"] = bid_slope
    frame["ask_depth_slope_20"] = ask_slope
    slope_total = bid_slope + ask_slope
    frame["depth_slope_imbalance_20"] = np.divide(bid_slope - ask_slope, slope_total, out=np.full_like(mid, np.nan), where=slope_total > 0)

    frame["impact_buy_bps"] = _impact(ap, aq, mid, impact_notional_usdt, buy=True)
    frame["impact_sell_bps"] = _impact(bp, bq, mid, impact_notional_usdt, buy=False)
    both_sides_filled = frame["impact_buy_bps"].notna() & frame["impact_sell_bps"].notna()
    frame["price_impact_bps"] = np.where(
        both_sides_filled,
        frame[["impact_buy_bps", "impact_sell_bps"]].max(axis=1),
        np.nan,
    )

    times = frame["timestamp_utc"].astype("int64").to_numpy()
    one_second = 1_000_000_000
    continuous = np.r_[False, np.diff(times) == one_second]
    valid = frame["book_valid"].to_numpy(dtype=bool) & np.isfinite(mid)
    starts = ~(continuous & valid & np.r_[False, valid[:-1]])
    segments = pd.Series(np.cumsum(starts), index=frame.index)
    returns = pd.Series(np.log(mid), index=frame.index).diff().where(valid)
    returns = returns.mask(segments.groupby(segments).cumcount().eq(0))
    frame["mid_return_1s"] = returns
    squared = returns.pow(2)
    for window in (10, 60, 300):
        frame[f"rv_{window}s"] = _window_sum(squared, segments, window)

    ofi_name = next((name for name in ("ofi_1s", "ofi_event_sum_1s", "ofi") if name in frame), None)
    if ofi_name is not None:
        frame["ofi_1s"] = pd.to_numeric(frame[ofi_name], errors="coerce")
        for window in (5, 10):
            frame[f"ofi_{window}s"] = _window_sum(frame["ofi_1s"], segments, window)
    else:
        for name in ("ofi_1s", "ofi_5s", "ofi_10s"):
            if name not in frame:
                frame[name] = np.nan

    buy_name = next((name for name in ("aggressive_buy_volume_1s", "buy_volume_1s") if name in frame), None)
    sell_name = next((name for name in ("aggressive_sell_volume_1s", "sell_volume_1s") if name in frame), None)
    if buy_name and sell_name:
        buy = pd.to_numeric(frame[buy_name], errors="coerce").fillna(0)
        sell = pd.to_numeric(frame[sell_name], errors="coerce").fillna(0)
        for window in (5, 10):
            buys = _window_sum(buy, segments, window)
            sells = _window_sum(sell, segments, window)
            frame[f"trade_imbalance_{window}s"] = np.divide(buys - sells, buys + sells, out=np.full(len(frame), np.nan), where=(buys + sells).to_numpy() > 0)
    else:
        for name in ("trade_imbalance_5s", "trade_imbalance_10s"):
            if name not in frame:
                frame[name] = np.nan
    for source, output in (
        ("addition_intensity_1s", "addition_intensity"),
        ("cancellation_intensity_1s", "cancellation_intensity"),
    ):
        if source in frame:
            values = pd.to_numeric(frame[source], errors="coerce")
            frame[source] = values
            frame[f"{output}_5s"] = _window_sum(values, segments, 5)
        else:
            frame[source] = np.nan
            frame[f"{output}_5s"] = np.nan
    return frame

