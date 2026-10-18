"""VaR and ES backtests with horizon-aware dependence handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def quantile_loss(losses, var, confidence: float) -> float:
    actual = np.asarray(losses, dtype=float)
    threshold = np.asarray(var, dtype=float)
    error = actual - threshold
    return float(np.nanmean((confidence - (error < 0).astype(float)) * error))


def kupiec_uc(hits, p: float) -> dict:
    values = np.asarray(hits, dtype=int)
    n = len(values)
    exceptions = int(values.sum())
    if n == 0 or not 0 < p < 1:
        raise ValueError("Kupiec test needs hits and expected rate p in (0,1)")
    observed = exceptions / n
    lr = 2 * (
        xlogy(exceptions, max(observed, np.finfo(float).tiny))
        + xlogy(n - exceptions, max(1 - observed, np.finfo(float).tiny))
        - xlogy(exceptions, p)
        - xlogy(n - exceptions, 1 - p)
    )
    return {
        "n_obs": n,
        "exceptions": exceptions,
        "expected_exceedances": n * p,
        "exception_rate": observed,
        "lr_uc": float(max(lr, 0.0)),
        "pvalue": float(chi2.sf(max(lr, 0.0), 1)),
    }


def christoffersen_ind(hits) -> dict:
    values = np.asarray(hits, dtype=int)
    if values.size < 3:
        raise ValueError("Christoffersen independence test needs at least 3 hits")
    previous, current = values[:-1], values[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    ll_null = xlogy(n00 + n10, max(1 - p, np.finfo(float).tiny)) + xlogy(n01 + n11, max(p, np.finfo(float).tiny))
    ll_alt = (
        xlogy(n00, max(1 - p01, np.finfo(float).tiny))
        + xlogy(n01, max(p01, np.finfo(float).tiny))
        + xlogy(n10, max(1 - p11, np.finfo(float).tiny))
        + xlogy(n11, max(p11, np.finfo(float).tiny))
    )
    lr = max(float(-2 * (ll_null - ll_alt)), 0.0)
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "lr_ind": lr, "pvalue": float(chi2.sf(lr, 1)),
    }


def christoffersen_cc(hits, p: float) -> dict:
    uc = kupiec_uc(hits, p)
    ind = christoffersen_ind(hits)
    lr = uc["lr_uc"] + ind["lr_ind"]
    return {
        **uc,
        **{key: value for key, value in ind.items() if key not in {"pvalue"}},
        "lr_cc": float(lr),
        "pvalue_cc": float(chi2.sf(lr, 2)),
    }


def dq_test(hits, var, confidence: float, lags: int = 4) -> dict:
    values = np.asarray(hits, dtype=float)
    quantiles = np.asarray(var, dtype=float)
    if values.shape != quantiles.shape:
        raise ValueError("hits and VaR vectors must align")
    n = len(values)
    if n <= lags + 5:
        return {"n_obs": n, "lags": lags, "statistic": None, "pvalue": None, "status": "insufficient_sample"}
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError("Install statsmodels to run the DQ test") from exc
    response = values[lags:] - (1 - confidence)
    columns = [np.ones(n - lags), quantiles[lags:]]
    columns.extend(values[lags - i - 1:n - i - 1] for i in range(lags))
    design = np.column_stack(columns)
    fit = sm.OLS(response, design).fit()
    restriction = np.eye(design.shape[1])
    test = fit.f_test((restriction, np.zeros(design.shape[1])))
    statistic = float(np.asarray(test.fvalue).squeeze() * design.shape[1])
    return {
        "n_obs": int(n),
        "lags": int(lags),
        "statistic": statistic,
        "pvalue": float(chi2.sf(statistic, design.shape[1])),
        "reference_distribution": "chi2",
    }


def non_overlapping(frame, horizon: int, offset: int = 0):
    if horizon < 1 or not 0 <= offset < horizon:
        raise ValueError("horizon must be positive and offset in [0,horizon)")
    return frame.iloc[offset::horizon]


def var_backtest(losses, var, confidence: float, horizon: int = 1, offset: int = 0) -> dict:
    """Coverage tests use disjoint h-day samples for horizons above one."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    values = pd.DataFrame({"loss": np.asarray(losses, dtype=float), "var": np.asarray(var, dtype=float)}).dropna()
    if horizon > 1:
        values = non_overlapping(values, horizon, offset)
    hits = (values["loss"].to_numpy() > values["var"].to_numpy()).astype(int)
    if len(hits) == 0:
        raise ValueError("No aligned loss and VaR observations")
    result = {
        "horizon_days": horizon,
        "offset": offset,
        "n_obs": int(len(hits)),
        "n_independent": int(len(hits)),
        "exceptions": int(hits.sum()),
        "expected_exceedances": float(len(hits) * (1 - confidence)),
        "exception_rate": float(hits.mean()),
        "quantile_loss": quantile_loss(values["loss"], values["var"], confidence),
        "kupiec": kupiec_uc(hits, 1 - confidence),
        "independence": christoffersen_ind(hits) if len(hits) >= 3 else None,
        "conditional_coverage": christoffersen_cc(hits, 1 - confidence) if len(hits) >= 3 else None,
    }
    if horizon == 1:
        result["dq"] = dq_test(hits, values["var"].to_numpy(), confidence)
    if horizon > 1 and result["expected_exceedances"] < 10:
        result["warning"] = "Low expected exception count; descriptive evidence only"
    return result
