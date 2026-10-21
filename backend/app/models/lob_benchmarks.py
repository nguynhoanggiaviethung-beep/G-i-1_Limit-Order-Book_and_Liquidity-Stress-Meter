"""Small interpretable baselines for LOB-derived targets."""

from __future__ import annotations

import numpy as np


def persistence_rv(past_rv: float | None, horizon_samples: int, reference_window: int = 60) -> float:
    if horizon_samples < 1 or reference_window < 1:
        raise ValueError("horizon and reference window must be positive")
    if past_rv is None or not np.isfinite(float(past_rv)) or float(past_rv) < 0:
        return float("nan")
    return max(float(past_rv) * horizon_samples / reference_window, np.finfo(float).tiny)


def empirical_var_es(returns, confidence: float) -> tuple[float, float]:
    """Return positive loss VaR and ES from past-only LOB mid-price returns."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 30:
        return float("nan"), float("nan")
    losses = -values
    var = max(float(np.quantile(losses, confidence)), 0.0)
    tail = losses[losses >= var]
    es = max(float(tail.mean()) if tail.size else var, var)
    # FZ0 requires strictly ES > VaR; this epsilon only stabilizes tied empirical tails.
    es = max(es, var + np.finfo(float).eps)
    return var, es
