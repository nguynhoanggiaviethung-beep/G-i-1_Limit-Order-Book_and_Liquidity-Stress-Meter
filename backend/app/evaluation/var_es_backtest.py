"""Basic VaR exception and quantile-loss summaries."""

import numpy as np
import pandas as pd


def var_backtest(returns: pd.Series, var_loss: float, confidence: float) -> dict[str, float | int]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    losses = -pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    hits = losses > var_loss
    return {"observations": int(len(losses)), "exceptions": int(hits.sum()),
            "exception_rate": float(hits.mean()) if len(hits) else float("nan"),
            "expected_exception_rate": float(1 - confidence),
            "quantile_loss": float(np.mean((confidence - (losses <= var_loss)) * (losses - var_loss))) if len(losses) else float("nan")}
