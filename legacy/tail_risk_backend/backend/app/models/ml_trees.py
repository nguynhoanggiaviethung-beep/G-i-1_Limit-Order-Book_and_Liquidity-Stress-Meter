"""Tree-model builders, with optional LightGBM and XGBoost dependencies."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def make_random_forest(
    random_state: int = 20260930,
    n_estimators: int = 600,
    max_depth: int | None = None,
    min_samples_leaf: int = 15,
    max_features: float = 0.7,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def lightgbm_qlike_objective(y_true, raw_prediction):
    actual = np.asarray(y_true, dtype=float)
    prediction = np.asarray(raw_prediction, dtype=float)
    variance = np.exp(np.clip(prediction, -30, 30))
    ratio = actual / np.maximum(variance, np.finfo(float).tiny)
    gradient = 1.0 - ratio
    hessian = np.maximum(ratio, 1e-8)
    return gradient, hessian


def make_lightgbm(qlike: bool = False, random_state: int = 20260930, alpha: float | None = None, **overrides):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Install backend/requirements-optional.txt to use LightGBM") from exc
    kwargs = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "verbosity": -1,
        "n_jobs": -1,
    }
    if alpha is not None:
        kwargs.update(objective="quantile", alpha=alpha)
    elif qlike:
        kwargs.update(objective=lightgbm_qlike_objective)
    kwargs.update(overrides)
    return LGBMRegressor(**kwargs)


def make_xgboost(random_state: int = 20260930, **overrides):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("Install backend/requirements-optional.txt to use XGBoost") from exc
    kwargs = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    kwargs.update(overrides)
    return XGBRegressor(**kwargs)
