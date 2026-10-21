"""Automated target timing and feature-source leakage checks."""

from __future__ import annotations

import pandas as pd


NON_FEATURE_COLUMNS = {
    "symbol", "sample_time_utc", "event_time_utc", "received_at_utc", "last_update_id",
    "book_valid", "quality_flags", "target_return", "target_rv", "target_abs_loss",
    "target_end_utc", "target_valid", "fold", "split",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [name for name in frame.columns if name not in NON_FEATURE_COLUMNS]
    return [name for name in columns if pd.api.types.is_numeric_dtype(frame[name])]


def assert_no_target_leakage(frame: pd.DataFrame) -> dict:
    required = {"sample_time_utc", "target_end_utc", "target_valid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing target timing fields: {', '.join(sorted(missing))}")
    origin = pd.to_datetime(frame["sample_time_utc"], utc=True, errors="coerce")
    end = pd.to_datetime(frame["target_end_utc"], utc=True, errors="coerce")
    valid = frame["target_valid"].fillna(False).astype(bool)
    invalid = valid & (end.isna() | origin.isna() | (end <= origin))
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} targets that are not strictly after their feature origin")
    forbidden = [
        name for name in feature_columns(frame)
        if any(token in name.lower() for token in ("target", "future", "label"))
    ]
    if forbidden:
        raise ValueError(f"Target-like values entered model features: {', '.join(forbidden)}")
    return {"rows": len(frame), "valid_targets": int(valid.sum()), "feature_columns": feature_columns(frame)}
