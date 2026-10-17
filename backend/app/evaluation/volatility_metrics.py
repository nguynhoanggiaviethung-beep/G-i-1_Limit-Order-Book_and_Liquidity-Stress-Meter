"""Out-of-sample volatility forecast metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def volatility_metrics(actual, forecast) -> dict[str, float]:
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("actual and forecast must have the same non-empty shape")
    if np.any(actual < 0) or np.any(forecast < 0):
        raise ValueError("variance/volatility values must be non-negative")
    eps = np.finfo(float).tiny
    qlike = np.mean(np.log(np.maximum(forecast, eps)) + actual / np.maximum(forecast, eps))
    return {"mae": float(mean_absolute_error(actual, forecast)),
            "rmse": float(np.sqrt(mean_squared_error(actual, forecast))), "qlike": float(qlike)}
