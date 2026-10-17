"""Direct conditional quantile regressors and crossing correction."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


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

    def __init__(self, quantiles=(0.95, 0.975, 0.99), random_state: int = 20260930):
        self.quantiles = tuple(float(q) for q in quantiles)
        self.model = RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=15,
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
