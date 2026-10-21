"""Peaks-over-threshold EVT tail forecasts on standardized losses."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def evt_pot_var_es(
    sigma: float,
    standardized_residuals,
    confidence: float,
    threshold_quantile: float = 0.90,
    min_exceedances: int = 30,
) -> dict:
    losses = -np.asarray(standardized_residuals, dtype=float)
    losses = losses[np.isfinite(losses)]
    if sigma < 0 or not 0 < confidence < 1 or not 0 < threshold_quantile < 1:
        raise ValueError("Invalid sigma, confidence, or EVT threshold")
    threshold = float(np.quantile(losses, threshold_quantile))
    excess = losses[losses > threshold] - threshold
    if len(excess) < min_exceedances:
        raise ValueError(f"EVT needs at least {min_exceedances} exceedances; got {len(excess)}")
    shape, _, scale = genpareto.fit(excess, floc=0)
    tail_probability = 1 - confidence
    exceedance_rate = len(excess) / len(losses)
    ratio = tail_probability / exceedance_rate
    if shape < 1e-8:
        standardized_var = threshold - scale * np.log(ratio)
    else:
        standardized_var = threshold + (scale / shape) * (ratio ** (-shape) - 1)
    if shape >= 1:
        standardized_es = float("inf")
    else:
        standardized_es = (
            standardized_var + scale - shape * threshold
        ) / (1 - shape)
    return {
        "var_loss_percent": max(float(sigma * standardized_var), 0.0),
        "expected_shortfall_loss_percent": max(float(sigma * standardized_es), 0.0),
        "threshold": threshold,
        "shape": float(shape),
        "scale": float(scale),
        "n_exceedances": int(len(excess)),
        "reliability": "low_exceedances" if len(excess) < 50 else "estimated",
    }
