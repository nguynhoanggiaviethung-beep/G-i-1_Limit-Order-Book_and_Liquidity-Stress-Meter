import numpy as np
import pandas as pd

from scripts.run_risk_eval import _prepare_residual_histories, _residual_pool


def test_residual_pool_drops_near_zero_burn_in_forecasts():
    origin = pd.Timestamp("2024-01-10", tz="UTC")
    frame = pd.DataFrame(
        {
            "model": ["harq", "harq", "harq"],
            "status": ["ok", "ok", "ok"],
            "target_end": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
            ),
            "origin_time": pd.to_datetime(
                ["2023-12-31", "2024-01-01", "2024-01-02"], utc=True
            ),
            "y_cum_return": [0.01, -0.02, 0.03],
            "rv_forecast": [np.finfo(float).tiny, 0.04, 0.09],
            "kappa_return_variance_to_rv": [1.0, 1.0, 1.0],
            "train_forecast_floor": [0.02, 0.02, 0.02],
        }
    )

    residuals = _residual_pool(frame, "harq", origin, horizon=1, window=20)
    prepared = _residual_pool(
        frame,
        "harq",
        origin,
        horizon=1,
        window=20,
        prepared_histories=_prepare_residual_histories(frame),
    )

    np.testing.assert_allclose(residuals, [-0.1, 0.1])
    np.testing.assert_allclose(prepared, residuals)


def test_residual_pool_accepts_legacy_forecasts_without_training_floor():
    origin = pd.Timestamp("2024-01-10", tz="UTC")
    frame = pd.DataFrame(
        {
            "model": ["harq"],
            "status": ["ok"],
            "target_end": pd.to_datetime(["2024-01-01"], utc=True),
            "origin_time": pd.to_datetime(["2023-12-31"], utc=True),
            "y_cum_return": [0.01],
            "rv_forecast": [0.04],
            "kappa_return_variance_to_rv": [1.0],
        }
    )

    residuals = _residual_pool(frame, "harq", origin, horizon=1, window=20)
    prepared = _residual_pool(
        frame,
        "harq",
        origin,
        horizon=1,
        window=20,
        prepared_histories=_prepare_residual_histories(frame),
    )

    np.testing.assert_allclose(residuals, [0.05])
    np.testing.assert_allclose(prepared, residuals)
