"""GARCH-family estimation helpers backed by the arch package."""

from __future__ import annotations


GARCH_SPECS = {
    "garch_t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "gjr_t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "egarch_t": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "garch_skewt": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "skewt"},
}


def fit_garch_family(returns, model_name: str = "garch_t"):
    """Fit a configured daily GARCH-family model to percent returns."""
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("Install the arch package to run GARCH models") from exc
    if model_name not in GARCH_SPECS:
        raise ValueError(f"Unsupported GARCH model: {model_name}")
    spec = GARCH_SPECS[model_name]
    model = arch_model(
        returns,
        mean="Constant",
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        dist=spec["dist"],
        rescale=False,
    )
    return model.fit(disp="off", show_warning=False, cov_type="robust")


def fit_garch(returns):
    """Backward-compatible GARCH(1,1) Student-t adapter."""
    return fit_garch_family(returns, "garch_t")


def forecast_variance_path(result, horizon: int) -> list[float]:
    """Expected daily return variance for each forecast step."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    try:
        forecast = result.forecast(horizon=horizon, reindex=False)
    except (ValueError, NotImplementedError):
        forecast = result.forecast(
            horizon=horizon,
            reindex=False,
            method="simulation",
            simulations=2000,
        )
    row = forecast.variance.iloc[-1]
    return [float(max(value, 0.0)) for value in row.iloc[:horizon]]
