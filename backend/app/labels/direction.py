"""Three-class future mid-price direction labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION_LABEL_SCHEMA_VERSION = "midprice-direction.v1"


def _valid_future_window(frame: pd.DataFrame, origin: int, horizon_seconds: int) -> int | None:
    target = origin + horizon_seconds
    if target >= len(frame):
        return None
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").iloc[origin : target + 1]
    if times.isna().any() or len(times) != horizon_seconds + 1:
        return None
    expected = 1_000_000_000
    if not np.all(np.diff(times.astype("int64").to_numpy()) == expected):
        return None
    if "book_valid" in frame and not frame["book_valid"].iloc[origin : target + 1].fillna(False).astype(bool).all():
        return None
    return target


def fit_direction_delta(training_frame: pd.DataFrame, horizon_seconds: int, quantile: float = 0.50) -> float:
    """Fit a symbol/horizon-specific flat-band threshold using training rows only."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    prices = pd.to_numeric(training_frame["mid_price"], errors="coerce").to_numpy(dtype=float)
    valid = valid_future_origins(training_frame, horizon_seconds)[:-horizon_seconds]
    changes = np.abs(prices[horizon_seconds:] - prices[:-horizon_seconds])
    changes = changes[valid]
    changes = changes[np.isfinite(changes)]
    if not len(changes):
        raise ValueError("No complete training windows are available to fit delta")
    return float(np.quantile(changes, quantile))


def make_direction_labels(frame: pd.DataFrame, horizon_seconds: int, delta: float) -> pd.Series:
    """Label each origin using the exact future timestamp and contiguous path."""
    if horizon_seconds < 1 or not np.isfinite(delta) or delta < 0:
        raise ValueError("horizon_seconds must be positive and delta non-negative")
    prices = pd.to_numeric(frame["mid_price"], errors="coerce").to_numpy(dtype=float)
    valid = valid_future_origins(frame, horizon_seconds)[:-horizon_seconds]
    labels = np.full(len(frame), None, dtype=object)
    changes = prices[horizon_seconds:] - prices[:-horizon_seconds]
    origins = np.flatnonzero(valid)
    origin_changes = changes[valid]
    labels[origins] = np.where(origin_changes > delta, "up", np.where(origin_changes < -delta, "down", "flat"))
    return pd.Series(labels, index=frame.index, name="direction_label", dtype="object")


def valid_future_origins(frame: pd.DataFrame, horizon_seconds: int) -> np.ndarray:
    """Vectorized contiguous-window mask; one means t..t+h are all valid."""
    size = len(frame)
    output = np.zeros(size, dtype=bool)
    if horizon_seconds < 1 or size <= horizon_seconds:
        return output
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").astype("int64").to_numpy()
    valid = np.ones(size, dtype=bool)
    if "book_valid" in frame:
        valid &= frame["book_valid"].fillna(False).astype(bool).to_numpy()
    valid &= np.isfinite(pd.to_numeric(frame["mid_price"], errors="coerce").to_numpy(dtype=float))
    steps = np.zeros(size, dtype=np.int8)
    steps[1:] = ((np.diff(times) == 1_000_000_000) & valid[1:] & valid[:-1]).astype(np.int8)
    prefix = np.r_[0, np.cumsum(steps, dtype=np.int64)]
    origins = np.arange(size - horizon_seconds)
    ends = origins + horizon_seconds
    output[origins] = (prefix[ends + 1] - prefix[origins + 1]) == horizon_seconds
    return output

