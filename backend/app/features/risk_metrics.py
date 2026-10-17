"""Historical simulation VaR and Expected Shortfall in loss-positive units."""

import pandas as pd


def historical_var_es(returns: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("returns has no valid observations")
    loss = -clean
    var = float(loss.quantile(confidence))
    tail = loss[loss >= var]
    return {"confidence": confidence, "var": var, "expected_shortfall": float(tail.mean()), "observations": int(len(clean))}
