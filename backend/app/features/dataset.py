"""Leakage-safe direct datasets for one-, seven-, and thirty-day horizons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.features.har_features import har_features
from app.features.point_in_time import PITFrame


@dataclass(frozen=True)
class DatasetSpec:
    horizon: int
    origin: str = "00:00:00Z"
    feature_sets: tuple[str, ...] = ("har", "rv_components", "ohlcv", "calendar")
    windows: tuple[int, int] = (7, 30)

    def __post_init__(self) -> None:
        if self.horizon not in (1, 7, 30):
            raise ValueError("Research horizons are fixed at 1, 7, and 30 days")
        if self.origin != "00:00:00Z":
            raise ValueError("The primary forecast origin is fixed at 00:00 UTC")


@dataclass
class Dataset:
    x: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    feature_available_at: pd.Series

    def assert_no_leakage(self) -> None:
        PITFrame(self.x, self.feature_available_at).assert_available(self.meta["origin_time"])
        if (self.meta["target_end"] <= self.meta["origin_time"]).any():
            raise ValueError("Target window must start after its forecast origin")

    def to_frame(self) -> pd.DataFrame:
        frame = self.meta.copy()
        frame["target_rv_mean"] = self.y
        return frame.join(self.x, how="left")


def build_dataset(
    daily: pd.DataFrame,
    spec: DatasetSpec,
    as_of: str | pd.Timestamp,
) -> Dataset:
    """Create direct h-day targets; incomplete days and unavailable labels are excluded."""
    if daily.empty:
        raise ValueError("Daily realized-variation frame is empty")
    ordered = daily.sort_values("utc_date").copy().reset_index(drop=True)
    ordered["utc_date"] = pd.to_datetime(ordered["utc_date"], utc=True).dt.floor("D")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    features = har_features(ordered, spec.windows)
    features["bpv_share"] = features["bpv"] / features["rv_lag_1"].where(features["rv_lag_1"] > 0)
    features["jump_share"] = features["jump"] / features["rv_lag_1"].where(features["rv_lag_1"] > 0)
    features["taker_buy_ratio"] = (
        pd.to_numeric(ordered.get("taker_buy_base_volume", pd.Series(np.nan, index=ordered.index)), errors="coerce")
        / pd.to_numeric(ordered.get("volume", pd.Series(np.nan, index=ordered.index)), errors="coerce").replace(0, np.nan)
    )
    features["origin_day_of_week"] = (pd.to_datetime(ordered["utc_date"], utc=True) + pd.Timedelta(days=1)).dt.dayofweek

    dates = pd.DatetimeIndex(ordered["utc_date"])
    valid_days = ordered["is_complete_day"].fillna(False).astype(bool).to_numpy()
    rv = pd.to_numeric(ordered["rv"], errors="coerce").to_numpy(dtype=float)
    daily_return = pd.to_numeric(ordered.get("close_ret", ordered.get("daily_return")), errors="coerce").to_numpy(dtype=float)
    n = len(ordered)
    targets = np.full(n, np.nan, dtype=float)
    cumulative_returns = np.full(n, np.nan, dtype=float)
    origin_times = dates + pd.Timedelta(days=1)
    target_starts = origin_times
    target_ends = origin_times + pd.Timedelta(days=spec.horizon)
    target_closes = target_ends - pd.Timedelta(milliseconds=1)
    n_dropped_gaps = np.zeros(n, dtype="int64")

    for anchor in range(n):
        start = anchor + 1
        end = start + spec.horizon
        if end > n:
            continue
        target_slice = slice(start, end)
        complete = valid_days[target_slice] & np.isfinite(rv[target_slice])
        n_dropped_gaps[anchor] = int((~complete).sum())
        if complete.all() and target_closes[anchor] <= cutoff:
            targets[anchor] = float(np.mean(rv[target_slice]))
            if np.isfinite(daily_return[target_slice]).all():
                cumulative_returns[anchor] = float(np.sum(daily_return[target_slice]))

    available_at = pd.to_datetime(ordered.get(
        "available_at_utc", ordered["utc_date"] + pd.Timedelta(days=1)
    ), utc=True)
    origin_rows = pd.DataFrame({
        "origin_time": origin_times,
        "target_start": target_starts,
        "target_end": target_ends,
        "target_close_time": target_closes,
        "label_available": np.isfinite(targets),
        "n_dropped_gaps": n_dropped_gaps,
        "y_cum_return": cumulative_returns,
    }, index=features.index)
    x = features.copy()
    x.index = pd.DatetimeIndex(origin_times, name="origin_time")
    origin_rows.index = x.index
    target_series = pd.Series(targets, index=x.index, name="target_rv_mean")
    available_series = pd.Series(available_at.to_numpy(), index=x.index, name="available_at_utc")
    dataset = Dataset(x=x, y=target_series, meta=origin_rows, feature_available_at=available_series)
    dataset.assert_no_leakage()
    return dataset
