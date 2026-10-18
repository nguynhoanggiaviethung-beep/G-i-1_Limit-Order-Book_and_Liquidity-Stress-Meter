"""Parametric VaR and ES for loss-positive return distributions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t


def param_var_es(
    sigma: float,
    distribution: str,
    confidence: float,
    params: dict | None = None,
) -> tuple[float, float]:
    """Return positive-loss VaR and ES from a standardized innovation law."""
    if sigma < 0 or not 0 < confidence < 1:
        raise ValueError("sigma must be non-negative and confidence in (0, 1)")
    params = params or {}
    tail = 1 - confidence
    if distribution == "normal":
        q = float(norm.ppf(tail))
        es = float(norm.pdf(q) / tail)
        return max(-sigma * q, 0.0), max(sigma * es, 0.0)
    if distribution in {"t", "student_t"}:
        degrees = float(params.get("nu", params.get("df", 8)))
        if degrees <= 2:
            raise ValueError("Student-t innovations require nu > 2 for unit variance")
        scale = np.sqrt((degrees - 2) / degrees)
        q = float(student_t.ppf(tail, df=degrees))
        es = float(student_t.pdf(q, df=degrees) * (degrees + q * q) / ((degrees - 1) * tail) * scale)
        return max(float(-sigma * scale * q), 0.0), max(float(sigma * es), 0.0)
    if distribution == "skewt":
        try:
            from arch.univariate import SkewStudent
        except ImportError as exc:
            raise RuntimeError("Install arch for standardized skew-t risk forecasts") from exc
        eta = float(params.get("eta", params.get("nu", 8)))
        lam = float(params.get("lambda", 0.0))
        if eta <= 2 or not -1 < lam < 1:
            raise ValueError("Skew-t requires eta > 2 and lambda in (-1, 1)")
        law = SkewStudent()
        params_array = np.asarray([eta, lam], dtype=float)
        q = float(law.ppf(tail, params_array))
        lower_partial = float(law.partial_moment(1, z=q, parameters=params_array))
        return max(-sigma * q, 0.0), max(-sigma * lower_partial / tail, 0.0)
    raise ValueError(f"Unsupported parametric distribution: {distribution}")


def fit_skewt_params(standardized_residuals) -> dict:
    """Fit Hansen's unit-variance skew-t shape to centered, unit-variance residuals.

    ``param_var_es`` evaluates the standardized Hansen law, so the empirical
    residual pool must be centered and scaled before fitting its shape. The
    fitted law then has the same zero-mean, unit-variance convention used by
    the VaR/ES quantile and partial-moment formulas below.
    """
    from arch.univariate import SkewStudent

    values = np.asarray(standardized_residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise ValueError("Skew-t fitting requires at least 100 finite residuals")
    values = values - float(np.mean(values))
    sample_scale = float(np.std(values, ddof=0))
    if not np.isfinite(sample_scale) or sample_scale <= np.finfo(float).eps:
        raise ValueError("Skew-t fitting requires residuals with positive finite variance")
    values = values / sample_scale
    distribution = SkewStudent()
    start = np.asarray(distribution.starting_values(values), dtype=float)

    def objective(parameters):
        if parameters[0] <= 2.001 or abs(parameters[1]) >= 0.999:
            return 1e20
        return -float(distribution.loglikelihood(
            parameters,
            values,
            np.ones_like(values),
            individual=False,
        ))

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(2.01, 200.0), (-0.99, 0.99)],
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"Skew-t MLE did not converge: {result.message}")
    return {"eta": float(result.x[0]), "lambda": float(result.x[1])}


def rv_to_sigma(rv_forecast: float, kappa: float = 1.0) -> float:
    """Compatibility alias; see risk.scaling.rv_to_sigma."""
    from app.risk.scaling import rv_to_sigma as convert

    return convert(rv_forecast, kappa)
