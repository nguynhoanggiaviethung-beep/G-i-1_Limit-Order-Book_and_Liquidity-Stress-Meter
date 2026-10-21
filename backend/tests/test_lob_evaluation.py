import numpy as np
import pandas as pd

from app.evaluation.lob_volatility import walk_forward_lob
from app.features.lob_targets import build_lob_targets


def test_walk_forward_fits_lob_rv_and_quantile_models_without_target_features():
    rng = np.random.default_rng(17)
    count = 180
    returns = rng.normal(0, 0.001, count)
    mid = 100 * np.exp(np.cumsum(returns))
    samples = pd.DataFrame({
        "symbol": "BTCUSDT",
        "sample_time_utc": pd.date_range("2025-01-01T00:00:00Z", periods=count, freq="s"),
        "book_valid": True,
        "mid_price": mid,
        "best_bid": mid - 0.01,
        "best_ask": mid + 0.01,
        "relative_spread_bps": np.full(count, 2.0),
        "obi_qty_5": rng.uniform(-0.8, 0.8, count),
        "past_rv_60": np.full(count, 1e-5),
    })
    dataset = build_lob_targets(samples, horizon_samples=2)
    forecasts, metrics = walk_forward_lob(
        dataset,
        horizon_samples=2,
        test_fraction=0.25,
        minimum_train_rows=30,
        refit_interval_samples=15,
        confidence_levels=(0.95,),
    )
    assert len(forecasts) > 0
    assert forecasts["ridge_rv_forecast"].notna().any()
    assert forecasts["hist_gradient_boosting_rv_forecast"].notna().any()
    assert forecasts["quantile_var_loss_950"].notna().any()
    assert "target_rv" not in metrics["feature_columns"]
    assert metrics["input_source"].startswith("Binance Spot LOB only")
