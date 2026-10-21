"""Forward RV and return targets built exclusively from sampled LOB mid-prices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_lob_targets(
    samples: pd.DataFrame,
    horizon_samples: int,
    sampling_interval_seconds: int = 1,
) -> pd.DataFrame:
    """Attach future targets only when every sampled book state is valid and on-grid."""
    if horizon_samples < 1 or sampling_interval_seconds < 1:
        raise ValueError("horizon_samples and sampling_interval_seconds must be positive")
    required = {"sample_time_utc", "mid_price", "book_valid"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Missing LOB sample fields: {', '.join(sorted(missing))}")
    frame = samples.sort_values("sample_time_utc").drop_duplicates("sample_time_utc", keep="last").reset_index(drop=True).copy()
    times = pd.to_datetime(frame["sample_time_utc"], utc=True)
    mids = pd.to_numeric(frame["mid_price"], errors="coerce").to_numpy(dtype=float)
    valid = frame["book_valid"].fillna(False).to_numpy(dtype=bool) & np.isfinite(mids) & (mids > 0)
    returns = np.full(len(frame), np.nan, dtype=float)
    returns[1:] = np.log(mids[1:] / mids[:-1])
    frame["sample_time_utc"] = times
    count = len(frame)
    target_return = np.full(count, np.nan, dtype=float)
    target_rv = np.full(count, np.nan, dtype=float)
    target_abs_loss = np.full(count, np.nan, dtype=float)
    target_end = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    target_valid = np.zeros(count, dtype=bool)
    if count > horizon_samples:
        step_ns = int(sampling_interval_seconds) * 1_000_000_000
        time_ns = times.array.asi8
        gap_at = np.zeros(count, dtype=bool)
        gap_at[1:] = np.diff(time_ns) != step_ns
        bad_state_prefix = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
        bad_transition = np.zeros(count, dtype=bool)
        bad_transition[1:] = gap_at[1:] | ~np.isfinite(returns[1:])
        bad_transition_prefix = np.r_[0, np.cumsum(bad_transition, dtype=np.int64)]
        squared_returns_prefix = np.r_[0.0, np.cumsum(np.nan_to_num(returns, nan=0.0) ** 2)]
        origins = np.arange(count - horizon_samples)
        ends = origins + horizon_samples
        complete = bad_state_prefix[ends + 1] - bad_state_prefix[origins] == 0
        complete &= bad_transition_prefix[ends + 1] - bad_transition_prefix[origins + 1] == 0
        complete &= np.isfinite(mids[origins]) & np.isfinite(mids[ends])
        selected = origins[complete]
        selected_ends = selected + horizon_samples
        selected_returns = np.log(mids[selected_ends] / mids[selected])
        target_return[selected] = selected_returns
        target_rv[selected] = squared_returns_prefix[selected_ends + 1] - squared_returns_prefix[selected + 1]
        target_abs_loss[selected] = np.maximum(-selected_returns, 0.0)
        target_end[selected] = time_ns[selected_ends].astype("datetime64[ns]")
        target_valid[selected] = True
    frame["target_return"] = target_return
    frame["target_rv"] = target_rv
    frame["target_abs_loss"] = target_abs_loss
    frame["target_end_utc"] = pd.to_datetime(target_end, utc=True)
    frame["target_valid"] = target_valid
    return frame
