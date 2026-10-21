"""Train-only liquidity-stress thresholds and forward labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.labels.direction import valid_future_origins


@dataclass(frozen=True)
class StressThresholds:
    spread_q95: float
    impact_q95: float
    depth_q05: float
    rv_60_q95: float
    positive_rate_train: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_stress_thresholds(
    training_frame: pd.DataFrame,
    upper_quantile: float = 0.95,
    depth_quantile: float = 0.05,
) -> StressThresholds:
    """Fit the composite definition only on the supplied training partition."""
    if not 0 < depth_quantile < upper_quantile < 1:
        raise ValueError("Stress quantiles must satisfy 0 < depth_quantile < upper_quantile < 1")
    spread = pd.to_numeric(training_frame["relative_spread_bps"], errors="coerce").dropna().to_numpy()
    impact_source = training_frame.get("price_impact_bps", training_frame.get("impact_bps"))
    impact = pd.to_numeric(impact_source, errors="coerce").dropna().to_numpy()
    depth = pd.to_numeric(training_frame["depth_qty_10"], errors="coerce").dropna().to_numpy()
    rv = pd.to_numeric(training_frame["rv_60s"], errors="coerce").dropna().to_numpy()
    if min(map(len, (spread, impact, depth, rv))) == 0:
        raise ValueError("Stress thresholds need spread, price impact, depth_qty_10, and rv_60s")
    thresholds = (float(np.quantile(spread, upper_quantile)), float(np.quantile(impact, upper_quantile)),
                  float(np.quantile(depth, depth_quantile)), float(np.quantile(rv, upper_quantile)))
    positive = (
        (pd.to_numeric(training_frame["relative_spread_bps"], errors="coerce") >= thresholds[0])
        | (pd.to_numeric(impact_source, errors="coerce") >= thresholds[1])
        | (pd.to_numeric(training_frame["depth_qty_10"], errors="coerce") <= thresholds[2])
        | (pd.to_numeric(training_frame["rv_60s"], errors="coerce") >= thresholds[3])
    )
    return StressThresholds(*thresholds, float(positive.mean()))


def make_stress_labels(frame: pd.DataFrame, horizon_seconds: int, thresholds: StressThresholds) -> pd.Series:
    """Mark future spread/impact/depth/RV stress; gaps remain unlabeled."""
    spread = pd.to_numeric(frame["relative_spread_bps"], errors="coerce").to_numpy(dtype=float)
    impact_name = "price_impact_bps" if "price_impact_bps" in frame else "impact_bps"
    impact = pd.to_numeric(frame[impact_name], errors="coerce").to_numpy(dtype=float)
    depth = pd.to_numeric(frame["depth_qty_10"], errors="coerce").to_numpy(dtype=float)
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
        (s >= thresholds.spread_q95)
        | (i >= thresholds.impact_q95)
        | (d <= thresholds.depth_q05)
        | (v >= thresholds.rv_60_q95)
    ).astype(int)
    return pd.Series(output, index=frame.index, name="stress_label", dtype="Int64")

