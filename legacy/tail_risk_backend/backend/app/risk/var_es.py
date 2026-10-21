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
    location = float(params.get("location", params.get("loc", 0.0)))
    innovation_scale = float(params.get("scale", 1.0))
    if not np.isfinite(location) or not np.isfinite(innovation_scale) or innovation_scale <= 0:
        raise ValueError("innovation location must be finite and scale must be positive")
    tail = 1 - confidence
    if distribution == "normal":
        q = float(norm.ppf(tail))
        lower_mean = -float(norm.pdf(q) / tail)
        var = -sigma * (location + innovation_scale * q)
        es = -sigma * (location + innovation_scale * lower_mean)
        return max(var, 0.0), max(es, 0.0)
    if distribution in {"t", "student_t"}:
        degrees = float(params.get("nu", params.get("df", 8)))
        if degrees <= 2:
            raise ValueError("Student-t innovations require nu > 2 for unit variance")
        scale = np.sqrt((degrees - 2) / degrees)
        q = float(student_t.ppf(tail, df=degrees))
        lower_mean = -float(student_t.pdf(q, df=degrees) * (degrees + q * q) / ((degrees - 1) * tail) * scale)
        var = -sigma * (location + innovation_scale * scale * q)
        es = -sigma * (location + innovation_scale * lower_mean)
        return max(float(var), 0.0), max(float(es), 0.0)
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
        var = -sigma * (location + innovation_scale * q)
        es = -sigma * (location + innovation_scale * lower_partial / tail)
        return max(var, 0.0), max(es, 0.0)
    raise ValueError(f"Unsupported parametric distribution: {distribution}")


def fit_skewt_params(standardized_residuals) -> dict:
    """Fit standardized skew-t shape and retain empirical residual location/scale.

    The shape likelihood is evaluated after centering/scaling, matching
    ``arch.univariate.SkewStudent``. Returning the removed location and scale
    lets ``param_var_es`` transform the standardized quantile and partial
    moment back to the empirical innovation distribution.
    """
    from arch.univariate import SkewStudent

    values = np.asarray(standardized_residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        raise ValueError("Skew-t fitting requires at least 100 finite residuals")
    location = float(np.mean(values))
    values = values - location
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

    result = None
    failures: list[str] = []
    # Short windows can make the likelihood gradient at arch's moment start
    # numerically abnormal. Retry a small, deterministic set of valid starts.
    tried_starts: list[np.ndarray] = []
    for candidate in (start, np.asarray([8.0, 0.0]), np.asarray([5.0, 0.0]), np.asarray([20.0, 0.0])):
        if any(np.allclose(candidate, tried) for tried in tried_starts):
            continue
        tried_starts.append(candidate)
        attempt = minimize(
            objective,
            candidate,
            method="L-BFGS-B",
            bounds=[(2.01, 200.0), (-0.99, 0.99)],
        )
        if attempt.success and np.isfinite(attempt.fun):
            result = attempt
            break
        failures.append(str(attempt.message))
    if result is None:
        raise ValueError(f"Skew-t MLE did not converge: {'; '.join(failures)}")
    return {
        "eta": float(result.x[0]),
        "lambda": float(result.x[1]),
        "location": location,
        "scale": sample_scale,
    }


def rv_to_sigma(rv_forecast: float, kappa: float = 1.0) -> float:
    """Compatibility alias; see risk.scaling.rv_to_sigma."""
    from app.risk.scaling import rv_to_sigma as convert

    return convert(rv_forecast, kappa)
