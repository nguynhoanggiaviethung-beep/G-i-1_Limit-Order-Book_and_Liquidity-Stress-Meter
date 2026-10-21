"""Construct leakage-safe external features from timestamped source events."""

from __future__ import annotations

import pandas as pd


FEATURE_SPECS = {
    "funding_rate": ("external_funding_rate_last", pd.Timedelta(1, unit="D")),
    "open_interest_value": ("external_open_interest_value_last", pd.Timedelta(2, unit="D")),
    "basis_rate": ("external_basis_rate_last", pd.Timedelta(2, unit="D")),
    "vixcls": ("external_vix_last", pd.Timedelta(7, unit="D")),
    "dtwexbgs": ("external_usd_broad_index_last", pd.Timedelta(7, unit="D")),
}


def attach_point_in_time_external(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    symbol: str,
    min_coverage: float = 0.95,
    coverage_start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach latest non-stale events available at each existing daily origin.

    Sparse histories are deliberately excluded from the returned feature frame;
    a 30-day source cannot silently reduce the training sample of a multi-year
    walk-forward study.
    """
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    required = {"feature_name", "available_at_utc", "observation_time_utc", "value"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"External events are missing columns: {sorted(missing)}")
    if "available_at_utc" not in daily:
        raise ValueError("daily data must contain available_at_utc")

    output = daily.copy().reset_index(drop=True)
    origins = pd.to_datetime(output["available_at_utc"], utc=True)
    complete = output.get("is_complete_day", pd.Series(True, index=output.index)).fillna(False).astype(bool)
    if coverage_start is not None:
        start = pd.Timestamp(coverage_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        complete &= origins.ge(start)
    event_frame = events.copy()
    event_frame["available_at_utc"] = pd.to_datetime(event_frame["available_at_utc"], utc=True)
    event_frame["observation_time_utc"] = pd.to_datetime(event_frame["observation_time_utc"], utc=True)
    event_frame["value"] = pd.to_numeric(event_frame["value"], errors="coerce")
    if "symbol" in event_frame:
        event_frame = event_frame.loc[event_frame["symbol"].isna() | event_frame["symbol"].eq(symbol.upper())]

    report = {}
    for source_name, (feature_name, max_age) in FEATURE_SPECS.items():
        source = event_frame.loc[
            event_frame["feature_name"].str.lower().eq(source_name)
        ].dropna(subset=["value", "available_at_utc"])
        if source.empty:
            report[source_name] = {"included": False, "coverage": 0.0, "reason": "no source rows"}
            continue
        source = source.sort_values("available_at_utc").drop_duplicates("available_at_utc", keep="last")
        left = pd.DataFrame({"_origin": origins, "_row": output.index}).sort_values("_origin")
        right = source[["available_at_utc", "observation_time_utc", "value"]].rename(
            columns={
                "available_at_utc": "_available",
                "observation_time_utc": "_observed",
                "value": "_value",
            }
        )
        matched = pd.merge_asof(
            left,
            right.sort_values("_available"),
            left_on="_origin",
            right_on="_available",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row")
        age = matched["_origin"] - matched["_available"]
        values = matched["_value"].where(age.le(max_age)).reset_index(drop=True)
        coverage = float(values.loc[complete].notna().mean()) if complete.any() else 0.0
        included = coverage >= min_coverage
        report[source_name] = {
            "included": included,
            "coverage": coverage,
            "feature_column": feature_name,
            "reason": None if included else f"coverage below {min_coverage:.0%}",
        }
        if included:
            output[feature_name] = values.to_numpy()
    return output, report
