"""Direct HAR-RV models for each forecast horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


HAR_FEATURES = ["rv_lag_1", "rv_mean_7", "rv_mean_30"]
HAR_ROBUSTNESS_FEATURES = ["rv_lag_1", "rv_mean_5", "rv_mean_22"]
HAR_FEATURE_SETS = {
    "har": HAR_FEATURES,
    "har_log": HAR_FEATURES,
    "har_j": HAR_FEATURES + ["jump"],
    "har_cj": [
        "bpv", "bpv_mean_7", "bpv_mean_30",
        "jump", "jump_mean_7", "jump_mean_30",
    ],
    "shar": [
        "rs_pos", "rs_pos_mean_7", "rs_pos_mean_30",
        "rs_neg", "rs_neg_mean_7", "rs_neg_mean_30",
    ],
    "harq": HAR_FEATURES + ["harq_adjustment"],
    "har_5_22": HAR_ROBUSTNESS_FEATURES,
}


class HARDirectModel:
    """OLS HAR or log-HAR; log forecasts use a residual Jensen correction."""

    def __init__(self, log_scale: bool = False) -> None:
        self.log_scale = log_scale
        self.model = LinearRegression()
        self.residual_variance = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "HARDirectModel":
        target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        values = x.to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
        if valid.sum() < max(30, values.shape[1] * 5):
            raise ValueError("Too few complete observations for HAR fit")
        fit_target = np.log(np.maximum(target[valid], np.finfo(float).tiny)) if self.log_scale else target[valid]
        self.model.fit(values[valid], fit_target)
        residual = fit_target - self.model.predict(values[valid])
        self.residual_variance = float(np.var(residual, ddof=1)) if len(residual) > 1 else 0.0
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        result = self.model.predict(x.to_numpy(dtype=float))
        if self.log_scale:
            result = np.exp(result + 0.5 * self.residual_variance)
        return np.maximum(np.asarray(result, dtype=float), np.finfo(float).tiny)


def make_har_model(log_scale: bool = False) -> HARDirectModel:
    return HARDirectModel(log_scale=log_scale)
