"""Joint VaR/ES scoring and exceedance diagnostics."""

from __future__ import annotations

import numpy as np


def fz0_loss(losses, var, es, confidence: float) -> np.ndarray:
    """Loss-positive FZ0 score; lower values indicate better forecasts."""
    actual = np.asarray(losses, dtype=float)
    threshold = np.asarray(var, dtype=float)
    expected = np.asarray(es, dtype=float)
    if actual.shape != threshold.shape or actual.shape != expected.shape:
        raise ValueError("loss, VaR, and ES arrays must have equal shapes")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    if np.any(threshold <= 0) or np.any(expected <= threshold):
        raise ValueError("FZ0 requires positive VaR and ES > VaR")
    exceedance = (actual > threshold).astype(float)
    return exceedance * (actual - threshold) / ((1 - confidence) * expected) + threshold / expected + np.log(expected) - 1


def acerbi_szekely_z2(losses, var, es, confidence: float) -> dict:
    actual, threshold, expected = map(lambda x: np.asarray(x, dtype=float), (losses, var, es))
    if actual.shape != threshold.shape or actual.shape != expected.shape or np.any(expected <= 0):
        raise ValueError("Aligned losses, VaR, and positive ES are required")
    hits = actual > threshold
    z2 = float(np.sum(actual * hits / expected) / (len(actual) * (1 - confidence)) - 1)
    return {
        "z2": z2,
        "n_obs": int(len(actual)),
        "exceptions": int(hits.sum()),
        "expected_exceedances": float(len(actual) * (1 - confidence)),
        "interpretation": "positive values indicate ES underestimation" if z2 > 0 else "non-positive",
    }


def e_backtest(losses, var, es, confidence: float, betting_fraction: float = 0.5) -> dict:
    """Fixed-fraction e-process; crossing 1/alpha rejects conditional calibration."""
    actual, threshold, expected = map(lambda x: np.asarray(x, dtype=float), (losses, var, es))
    if actual.shape != threshold.shape or actual.shape != expected.shape:
        raise ValueError("Aligned losses, VaR, and ES are required")
    if np.any(expected <= threshold) or not 0 < confidence < 1 or not 0 < betting_fraction <= 1:
        raise ValueError("Require ES > VaR, confidence in (0,1), and betting fraction in (0,1]")
    e_values = np.maximum(actual - threshold, 0) / ((1 - confidence) * (expected - threshold))
    factors = 1 + betting_fraction * (e_values - 1)
    log_process = np.cumsum(np.log(np.maximum(factors, np.finfo(float).tiny)))
    maximum = float(np.exp(np.clip(log_process.max(initial=0), -700, 700)))
    return {
        "max_e_value": maximum,
        "log_e_process_final": float(log_process[-1]) if len(log_process) else 0.0,
        "reject_at_5pct": bool(maximum >= 20),
        "n_obs": int(len(actual)),
        "betting_fraction": betting_fraction,
    }


def es_backtest(losses, var, es, confidence: float) -> dict:
    actual, threshold, expected = map(lambda x: np.asarray(x, dtype=float), (losses, var, es))
    valid = np.isfinite(actual) & np.isfinite(threshold) & np.isfinite(expected)
    actual, threshold, expected = actual[valid], threshold[valid], expected[valid]
    scores = fz0_loss(actual, threshold, expected, confidence)
    return {
        "n_obs": int(len(actual)),
        "fz0_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "acerbi_szekely": acerbi_szekely_z2(actual, threshold, expected, confidence),
        "e_backtest": e_backtest(actual, threshold, expected, confidence),
        "mean_es": float(np.mean(expected)) if len(expected) else float("nan"),
    }
