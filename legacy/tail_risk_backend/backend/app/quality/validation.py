"""Validation for sampled local order-book states and point-in-time features."""

from __future__ import annotations

import pandas as pd


def validate_lob_samples(
    frame: pd.DataFrame,
    sampling_interval_seconds: int = 1,
    stale_after_seconds: int = 5,
) -> dict:
    required = {"symbol", "sample_time_utc", "last_update_id", "book_valid", "mid_price", "best_bid", "best_ask"}
    missing = sorted(required - set(frame.columns))
    errors: list[str] = []
    if missing:
        return {"rows": len(frame), "valid_rows": 0, "errors": [f"Missing columns: {', '.join(missing)}"]}
    data = frame.copy()
    data["sample_time_utc"] = pd.to_datetime(data["sample_time_utc"], utc=True, errors="coerce")
    duplicates = int(data.duplicated(["symbol", "sample_time_utc"]).sum())
    if duplicates:
        errors.append(f"Found {duplicates} duplicate sample keys")
    invalid_times = int(data["sample_time_utc"].isna().sum())
    if invalid_times:
        errors.append(f"Found {invalid_times} invalid sample timestamps")
    prices = data[["mid_price", "best_bid", "best_ask"]].apply(pd.to_numeric, errors="coerce")
    malformed = (
        ~data["book_valid"].fillna(False).astype(bool)
        | prices.isna().any(axis=1)
        | (prices["best_bid"] <= 0)
        | (prices["best_ask"] <= prices["best_bid"])
        | (prices["mid_price"] <= 0)
    )
    invalid_book_rows = int(malformed.sum())
    if invalid_book_rows:
        errors.append(f"Found {invalid_book_rows} invalid or crossed book sample(s)")
    stale_rows = 0
    if "event_age_ms" in data:
        stale_rows = int((pd.to_numeric(data["event_age_ms"], errors="coerce") > stale_after_seconds * 1000).sum())
        if stale_rows:
            errors.append(f"Found {stale_rows} samples with stale underlying events")
    gaps: list[dict] = []
    ordered = data.sort_values(["symbol", "sample_time_utc"]).reset_index(drop=True)
    for symbol, group in ordered.groupby("symbol", sort=False):
        deltas = group["sample_time_utc"].diff()
        for index in range(1, len(group)):
            delta = deltas.iloc[index]
            if delta > pd.Timedelta(seconds=sampling_interval_seconds * 1.5):
                gaps.append({"symbol": symbol, "after": group.iloc[index - 1]["sample_time_utc"].isoformat(),
                             "before": group.iloc[index]["sample_time_utc"].isoformat(),
                             "gap_seconds": delta.total_seconds()})
    return {
        "rows": len(data),
        "valid_rows": int((~malformed & data["sample_time_utc"].notna()).sum()),
        "duplicates": duplicates,
        "invalid_timestamps": invalid_times,
        "invalid_book_rows": invalid_book_rows,
        "stale_rows": stale_rows,
        "sampling_gaps": gaps,
        "sampling_interval_seconds": sampling_interval_seconds,
        "errors": errors,
    }
