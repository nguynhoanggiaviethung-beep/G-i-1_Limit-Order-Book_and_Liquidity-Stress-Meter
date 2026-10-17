"""Schema, candle integrity, duplicate, gap, and closed-candle checks."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


PROCESSED_COLUMNS = [
    "symbol", "interval", "open_time", "close_time", "open", "high", "low", "close",
    "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def normalize_klines(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)
    frame = raw.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time_ms"], unit="ms", utc=True)
    numeric = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame[PROCESSED_COLUMNS].sort_values("open_time").reset_index(drop=True)


def validate_klines(
    frame: pd.DataFrame,
    interval: str = "5m",
    now: datetime | None = None,
    raise_on_error: bool = True,
) -> dict[str, int | list[str]]:
    errors: list[str] = []
    missing_columns = sorted(set(PROCESSED_COLUMNS) - set(frame.columns))
    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")
        report: dict[str, int | list[str]] = {"rows": len(frame), "errors": errors, "duplicates": 0, "gaps": 0}
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return report

    duplicates = int(frame.duplicated(["symbol", "interval", "open_time"]).sum())
    if duplicates:
        errors.append(f"Found {duplicates} duplicate candle keys")
    invalid = ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
               (frame["low"] > frame[["open", "close", "high"]].min(axis=1)) |
               (frame["close"] <= 0) | (frame["volume"] < 0) | (frame["quote_volume"] < 0))
    invalid_count = int(invalid.sum())
    if invalid_count:
        errors.append(f"Found {invalid_count} invalid OHLCV candles")
    interval_delta = pd.Timedelta(interval)
    ordered = frame.sort_values("open_time").reset_index(drop=True)
    differences = ordered["open_time"].diff().dropna()
    gap_edges = []
    missing_intervals = 0
    for index, difference in differences.items():
        if difference > interval_delta:
            previous = ordered.loc[index - 1, "open_time"]
            following = ordered.loc[index, "open_time"]
            absent = max(int(difference / interval_delta) - 1, 1)
            missing_intervals += absent
            gap_edges.append({"after": previous.isoformat(), "before": following.isoformat(), "missing_candles": absent})
    if missing_intervals:
        errors.append(f"Found {missing_intervals} missing {interval} candle(s) across {len(gap_edges)} gap(s)")
    current = now or datetime.now(timezone.utc)
    if not frame.empty:
        open_times = pd.to_datetime(frame["open_time"], utc=True)
        unclosed = int((open_times + interval_delta > pd.Timestamp(current)).sum())
    else:
        unclosed = 0
    report = {"rows": len(frame), "duplicates": duplicates, "invalid_ohlcv": invalid_count,
              "gaps": len(gap_edges), "missing_intervals": missing_intervals, "gap_edges": gap_edges,
              "unclosed_candles": unclosed, "errors": errors}
    if unclosed:
        errors.append(f"Found {unclosed} candle(s) that have not closed")
    if errors and raise_on_error:
        raise ValueError("; ".join(errors))
    return report
