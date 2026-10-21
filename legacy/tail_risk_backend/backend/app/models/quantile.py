"""Direct conditional quantile regressors and crossing correction."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from app.models.ml_trees import make_lightgbm


QUANTILE_MODELS = ("linear_qr", "lightgbm_quantile", "qrf")


def rearrange_quantiles(values: dict[float, float]) -> dict[float, float]:
    """Monotone rearrangement prevents predicted quantile crossing."""
    levels = sorted(values)
    ordered = np.maximum.accumulate([float(values[level]) for level in levels])
    return {level: float(value) for level, value in zip(levels, ordered)}


def make_linear_quantile(quantile: float, alpha: float = 0.0) -> QuantileRegressor:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0,1)")
    return QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")


class QuantileRandomForest:
    """Empirical response quantiles from terminal-leaf neighborhoods."""

    def __init__(
        self,
        quantiles=(0.95, 0.975, 0.99),
        random_state: int = 20260930,
        n_estimators: int = 250,
        min_samples_leaf: int = 15,
    ):
        self.quantiles = tuple(float(q) for q in quantiles)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=0.7,
            random_state=random_state,
            n_jobs=-1,
        )
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.model.fit(self.x_train, self.y_train)
        self.leaf_ids = [tree.apply(self.x_train) for tree in self.model.estimators_]
        return self

    def predict_quantiles(self, x) -> dict[float, np.ndarray]:
        matrix = np.asarray(x, dtype=float)
        output = {level: [] for level in self.quantiles}
        for row in matrix:
            leaf_targets = []
            for tree, train_leaf in zip(self.model.estimators_, self.leaf_ids):
                leaf = tree.apply(row.reshape(1, -1))[0]
                leaf_targets.extend(self.y_train[train_leaf == leaf])
            neighborhood = np.asarray(leaf_targets if leaf_targets else self.y_train, dtype=float)
            for level in self.quantiles:
                output[level].append(float(np.quantile(neighborhood, level)))
        return {level: np.asarray(values) for level, values in output.items()}


def standardize_features(x_train, x_valid):
    """Fit scaling only on training rows to prevent validation leakage."""
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_valid), scaler


def fit_conditional_quantiles(
    model_name: str,
    x_train,
    loss_train,
    quantiles=(0.95, 0.975, 0.99),
    random_state: int = 20260930,
    config: dict | None = None,
) -> dict:
    """Fit direct loss-positive quantile regressors using training-only scaling."""
    if model_name not in QUANTILE_MODELS:
        raise ValueError(f"Unsupported quantile model: {model_name}")
    config = config or {}
    levels = tuple(float(level) for level in quantiles)
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(loss_train, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[valid], y[valid]
    if len(y) < int(config.get("min_train_rows", 100)):
        raise ValueError(f"Too few complete training rows for {model_name}: {len(y)}")

    if model_name == "qrf":
        estimator = QuantileRandomForest(
            quantiles=levels,
            random_state=random_state,
            n_estimators=int(config.get("n_estimators", 250)),
            min_samples_leaf=int(config.get("min_samples_leaf", 15)),
        ).fit(x, y)
        return {"estimator": estimator, "scaler": None, "columns": None}

    scaler = None
    fit_x = x
    if model_name == "linear_qr":
        scaler = StandardScaler().fit(x)
        fit_x = scaler.transform(x)
    estimators = {}
    for level in levels:
        if model_name == "linear_qr":
            estimator = make_linear_quantile(
                level,
                alpha=float(config.get("alpha", 0.001)),
            )
        else:
            estimator = make_lightgbm(
                alpha=level,
                random_state=random_state,
                n_estimators=int(config.get("n_estimators", 250)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                num_leaves=int(config.get("num_leaves", 15)),
                min_child_samples=int(config.get("min_child_samples", 30)),
            )
        estimator.fit(fit_x, y)
        estimators[level] = estimator
    return {"estimator": estimators, "scaler": scaler, "columns": None}


def predict_conditional_quantiles(model_name: str, state: dict, x_valid, quantiles) -> dict[float, float]:
    """Predict and monotonically rearrange loss quantiles at one origin."""
    levels = tuple(float(level) for level in quantiles)
    x = np.asarray(x_valid, dtype=float).reshape(1, -1)
    if model_name == "qrf":
        predictions = state["estimator"].predict_quantiles(x)
        raw = {level: float(np.asarray(predictions[level]).reshape(-1)[0]) for level in levels}
    else:
        scaler = state["scaler"]
        x_model = scaler.transform(x) if scaler is not None else x
        raw = {
            level: float(np.asarray(state["estimator"][level].predict(x_model)).reshape(-1)[0])
            for level in levels
        }
    return rearrange_quantiles(raw)


def empirical_expected_shortfall(loss_train, var_forecast: float) -> float:
    """Estimate ES from the point-in-time training tail beyond the predicted VaR."""
    values = np.asarray(loss_train, dtype=float)
    values = values[np.isfinite(values)]
    tail = values[values >= var_forecast]
    estimate = float(np.mean(tail)) if tail.size else float(var_forecast)
    return max(estimate, float(np.nextafter(var_forecast, np.inf)))
