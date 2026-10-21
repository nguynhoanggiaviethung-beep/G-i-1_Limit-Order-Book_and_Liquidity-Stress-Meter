"""Interpretable tabular models whose features come only from the LOB."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lob_rv_regressors(seed: int = 20261021) -> dict:
    """Return Ridge and histogram-gradient-boosting log-RV regressors."""
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                learning_rate=0.06,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }


def make_har_rv_regressor():
    """Return a linear HAR-RV model for lagged realized-volatility windows."""
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())


def make_lob_quantile_regressor(confidence: float, alpha: float = 1e-4):
    """Fit the lower return quantile for a loss VaR confidence level."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        QuantileRegressor(quantile=1 - confidence, alpha=alpha, solver="highs"),
    )
