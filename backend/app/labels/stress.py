"""Train-only liquidity-stress thresholds and forward labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.labels.direction import valid_future_origins


@dataclass(frozen=True)
class StressThresholds:
    spread_upper: float
    impact_upper: float
    depth_lower: float
    rv_60_upper: float
    positive_rate_train: float
    upper_quantile: float = 0.975
    depth_quantile: float = 0.025
    spread_metric: str = "spread_ticks"
    depth_metric: str = "depth_notional_10"

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def fit_stress_thresholds(
    training_frame: pd.DataFrame,
    upper_quantile: float = 0.975,
    depth_quantile: float = 0.025,
) -> StressThresholds:
    """Fit the composite definition only on the supplied training partition."""
    if not 0 < depth_quantile < upper_quantile < 1:
        raise ValueError("Stress quantiles must satisfy 0 < depth_quantile < upper_quantile < 1")
    spread_name = next((name for name in ("spread_ticks", "absolute_spread", "relative_spread_bps") if name in training_frame), None)
    depth_name = next((name for name in ("depth_notional_10", "depth_qty_10") if name in training_frame), None)
    if spread_name is None or depth_name is None:
        raise ValueError("Stress thresholds need spread_ticks (or absolute_spread) and top-10 depth")
    spread_series = pd.to_numeric(training_frame[spread_name], errors="coerce")
    depth_series = pd.to_numeric(training_frame[depth_name], errors="coerce")
    spread = spread_series.dropna().to_numpy()
    impact_source = training_frame.get("price_impact_bps", training_frame.get("impact_bps"))
    impact = pd.to_numeric(impact_source, errors="coerce").dropna().to_numpy()
    depth = depth_series.dropna().to_numpy()
    rv = pd.to_numeric(training_frame["rv_60s"], errors="coerce").dropna().to_numpy()
    if min(map(len, (spread, impact, depth, rv))) == 0:
        raise ValueError("Stress thresholds need spread, price impact, depth_qty_10, and rv_60s")
    thresholds = (float(np.quantile(spread, upper_quantile)), float(np.quantile(impact, upper_quantile)),
                  float(np.quantile(depth, depth_quantile)), float(np.quantile(rv, upper_quantile)))
    impact_series = pd.to_numeric(impact_source, errors="coerce")
    rv_series = pd.to_numeric(training_frame["rv_60s"], errors="coerce")
    valid = spread_series.notna() & impact_series.notna() & depth_series.notna() & rv_series.notna()
    positive = (
        (spread_series > thresholds[0])
        | (impact_series > thresholds[1])
        | (depth_series < thresholds[2])
        | (rv_series > thresholds[3])
    )
    prevalence = float(positive.loc[valid].mean()) if valid.any() else float("nan")
    return StressThresholds(
        *thresholds,
        prevalence,
        upper_quantile=upper_quantile,
        depth_quantile=depth_quantile,
        spread_metric=spread_name,
        depth_metric=depth_name,
    )


def make_stress_labels(frame: pd.DataFrame, horizon_seconds: int, thresholds: StressThresholds) -> pd.Series:
    """Mark future spread/impact/depth/RV stress; gaps remain unlabeled."""
    spread_name = thresholds.spread_metric if thresholds.spread_metric in frame else next(
        (name for name in ("spread_ticks", "absolute_spread", "relative_spread_bps") if name in frame),
        None,
    )
    depth_name = thresholds.depth_metric if thresholds.depth_metric in frame else next(
        (name for name in ("depth_notional_10", "depth_qty_10") if name in frame),
        None,
    )
    if spread_name is None or depth_name is None:
        raise ValueError("Stress labels need spread and top-10 depth features")
    spread = pd.to_numeric(frame[spread_name], errors="coerce").to_numpy(dtype=float)
    impact_name = "price_impact_bps" if "price_impact_bps" in frame else "impact_bps"
    impact = pd.to_numeric(frame[impact_name], errors="coerce").to_numpy(dtype=float)
    depth = pd.to_numeric(frame[depth_name], errors="coerce").to_numpy(dtype=float)
    rv = pd.to_numeric(frame["rv_60s"], errors="coerce").to_numpy(dtype=float)
    valid = valid_future_origins(frame, horizon_seconds)[:-horizon_seconds]
    output = np.full(len(frame), None, dtype=object)
    future = [array[horizon_seconds:] for array in (spread, impact, depth, rv)]
    origins = np.flatnonzero(valid)
    values = [array[valid] for array in future]
    complete = np.logical_and.reduce([np.isfinite(value) for value in values])
    selected = origins[complete]
    s, i, d, v = (value[complete] for value in values)
    output[selected] = (
        (s > thresholds.spread_upper)
        | (i > thresholds.impact_upper)
        | (d < thresholds.depth_lower)
        | (v > thresholds.rv_60_upper)
    ).astype(int)
    return pd.Series(output, index=frame.index, name="stress_label", dtype="Int64")


def monthly_positive_rates(labels: pd.Series, timestamps: pd.Series) -> dict[str, dict[str, float | int]]:
    """Report observed positive prevalence by UTC calendar month."""
    time_values = pd.to_datetime(timestamps, utc=True, errors="coerce")
    valid = labels.notna() & time_values.notna()
    if not valid.any():
        return {}
    frame = pd.DataFrame({"month": time_values.loc[valid].dt.strftime("%Y-%m"), "label": labels.loc[valid].astype(int)})
    grouped = frame.groupby("month", sort=True)["label"]
    return {
        str(month): {"n": int(values.size), "positive_rate": float(values.mean())}
        for month, values in grouped
    }


def fit_spread_only_threshold(training_frame: pd.DataFrame, quantile: float = 0.95) -> tuple[str, float]:
    """Fit a train-only single-spread threshold for the label sensitivity run."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    metric = next((name for name in ("spread_ticks", "absolute_spread", "relative_spread_bps") if name in training_frame), None)
    if metric is None:
        raise ValueError("Spread-only threshold requires spread_ticks or an absolute spread field")
    values = pd.to_numeric(training_frame[metric], errors="coerce")
    if "book_valid" in training_frame:
        values = values.where(training_frame["book_valid"].fillna(False).astype(bool))
    values = values.dropna().to_numpy(dtype=float)
    if not len(values):
        raise ValueError("No valid training spread observations are available")
    return metric, float(np.quantile(values, quantile))


def make_spread_only_labels(frame: pd.DataFrame, horizon_seconds: int, metric: str, threshold: float) -> pd.Series:
    """Create strict greater-than future spread labels for sensitivity reporting."""
    if metric not in frame:
        raise ValueError(f"Spread metric {metric!r} is absent from the feature frame")
    spread = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    valid = valid_future_origins(frame, horizon_seconds)[:-horizon_seconds]
    output = np.full(len(frame), None, dtype=object)
    future = spread[horizon_seconds:]
    origins = np.flatnonzero(valid)
    usable = np.isfinite(future[valid])
    selected = origins[usable]
    output[selected] = (future[valid][usable] > threshold).astype(int)
    return pd.Series(output, index=frame.index, name="spread_only_stress_label", dtype="Int64")

