"""Filtered historical simulation for multi-day loss distributions."""

from __future__ import annotations

import numpy as np


def fhs_var_es(
    sigma_path,
    standardized_residuals,
    confidence: float,
    n_paths: int = 10000,
    seed: int = 20260930,
) -> dict:
    """Bootstrap standardized shocks and aggregate simulated losses over h days."""
    sigma = np.asarray(sigma_path, dtype=float).reshape(-1)
    residuals = np.asarray(standardized_residuals, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if not sigma.size or np.any(sigma < 0) or residuals.size < 30:
        raise ValueError("FHS needs a non-negative volatility path and >=30 finite residuals")
    if not 0 < confidence < 1 or n_paths < 100:
        raise ValueError("confidence must be in (0,1) and n_paths >=100")
    rng = np.random.default_rng(seed)
    shocks = rng.choice(residuals, size=(n_paths, len(sigma)), replace=True)
    losses = -(shocks * sigma).sum(axis=1)
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return {
        "var_loss_percent": max(var, 0.0),
        "expected_shortfall_loss_percent": max(es, 0.0),
        "n_paths": int(n_paths),
        "horizon_days": int(len(sigma)),
        "risk_unit": f"loss-positive, % for {len(sigma)} day(s)",
        "method": "filtered_historical_simulation",
    }
