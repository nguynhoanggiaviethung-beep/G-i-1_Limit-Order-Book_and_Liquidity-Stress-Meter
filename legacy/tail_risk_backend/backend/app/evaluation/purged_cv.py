"""Expanding time-series folds that purge overlapping future target windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_time_series_splits(
    origins,
    target_end,
    validation_mask=None,
    folds: int = 5,
    embargo_days: int = 29,
):
    """Yield integer train/validation positions in chronological validation blocks."""
    origin = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    end = pd.DatetimeIndex(pd.to_datetime(target_end, utc=True))
    if len(origin) != len(end):
        raise ValueError("origins and target_end must have equal lengths")
    validation = np.ones(len(origin), dtype=bool) if validation_mask is None else np.asarray(validation_mask, dtype=bool)
    positions = np.flatnonzero(validation)
    if len(positions) < folds * 2:
        raise ValueError("Not enough validation observations for the requested folds")
    blocks = [block for block in np.array_split(positions, folds) if len(block)]
    embargo = pd.Timedelta(days=embargo_days)
    for block in blocks:
        first_validation = origin[block[0]]
        latest_train_origin = first_validation - embargo
        train_mask = (origin < first_validation) & (origin <= latest_train_origin) & (end <= first_validation)
        train_positions = np.flatnonzero(train_mask)
        if len(train_positions):
            yield train_positions, block


def assert_train_targets_before_origin(target_end, origin) -> None:
    target_end = pd.to_datetime(target_end, utc=True)
    origin = pd.to_datetime(origin, utc=True)
    if (target_end > origin).any():
        raise ValueError("Training contains a target window ending after the forecast origin")
