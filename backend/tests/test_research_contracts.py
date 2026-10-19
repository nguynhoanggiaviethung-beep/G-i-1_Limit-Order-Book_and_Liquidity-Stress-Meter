from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.evaluation.dm_test import dm_test
from app.evaluation.es_backtests import acerbi_szekely_z2, fz0_loss
from app.evaluation.mcs import model_confidence_set
from app.evaluation.purged_cv import purged_time_series_splits
from app.evaluation.var_es_backtest import christoffersen_cc, dq_test, kupiec_uc, non_overlapping
from app.features.dataset import DatasetSpec, build_dataset
from app.evaluation.volatility_metrics import volatility_metrics
from app.evaluation.tuning import tune_hyperparameters
from app.evaluation.walkforward import _fit_model, walk_forward
from app.api.schemas.forecast import ForecastRequest
from app.services import forecast_service


@pytest.mark.parametrize("model_name", ["lightgbm_l2", "xgboost"])
def test_variance_tree_models_train_on_log_target(monkeypatch, model_name):
    class CaptureEstimator:
        def fit(self, x, y):
            self.target = np.asarray(y, dtype=float)
            self.prediction = float(np.mean(self.target))

        def predict(self, x):
            return np.full(len(x), self.prediction)

    estimator = CaptureEstimator()
    monkeypatch.setattr(
        "app.evaluation.walkforward._make_estimator",
        lambda model, seed, hyperparameters=None: (estimator, None),
    )
    index = pd.date_range("2020-01-01", periods=150, freq="D", tz="UTC")
    x = pd.DataFrame({name: np.linspace(0.1, 1.0, len(index)) for name in (
        "ln_rv_lag_1", "ln_rv_mean_7", "ln_rv_mean_30", "bpv_share", "jump_share",
        "close_ret", "hl_range", "ln_volume", "ln_turnover", "ln_trade_count",
        "taker_buy_ratio", "day_of_week", "is_weekend",
    )}, index=index)
    y = pd.Series(np.linspace(0.01, 4.0, len(index)), index=index)

    state = _fit_model(model_name, x, y, 1, index[-1])

    np.testing.assert_allclose(estimator.target, np.log(y.to_numpy()))
    assert state.log_target
    assert np.isfinite(state.residual_variance)
    assert state.residual_variance >= 0
    assert state.variance_floor == pytest.approx(y.quantile(0.01))


def _daily_frame(days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    rv = np.linspace(0.001, 0.01, days)
    return pd.DataFrame({
        "utc_date": dates,
        "available_at_utc": dates + pd.Timedelta(days=1),
        "is_complete_day": True,
        "rv": rv,
        "bpv": rv * 0.8,
        "jump": rv * 0.2,
        "rs_pos": rv * 0.55,
        "rs_neg": rv * 0.45,
        "rq": rv**2,
        "close_ret": np.linspace(-0.02, 0.02, days),
        "hl_range": np.full(days, 0.03),
        "volume": np.full(days, 100.0),
        "quote_volume": np.full(days, 10000.0),
        "number_of_trades": np.full(days, 500.0),
        "taker_buy_base_volume": np.full(days, 50.0),
    })


def test_dataset_uses_only_past_features_and_drops_incomplete_target_days():
    daily = _daily_frame()
    first = build_dataset(daily, DatasetSpec(horizon=1), as_of="2021-01-01T00:00:00Z")
    changed_future = daily.copy()
    changed_future.loc[changed_future.index >= 80, "rv"] *= 1000
    second = build_dataset(changed_future, DatasetSpec(horizon=1), as_of="2021-01-01T00:00:00Z")
    origin = pd.Timestamp("2020-03-20T00:00:00Z")
    pd.testing.assert_series_equal(first.x.loc[origin], second.x.loc[origin])

    daily.loc[50, "is_complete_day"] = False
    daily.loc[50, "rv"] = np.nan
    with_gap = build_dataset(daily, DatasetSpec(horizon=1), as_of="2021-01-01T00:00:00Z")
    # Day 49's next-day target is the incomplete day 50.
    gap_target_origin = daily.loc[49, "utc_date"] + pd.Timedelta(days=1)
    assert np.isnan(with_gap.y.loc[gap_target_origin])
    assert with_gap.meta.loc[gap_target_origin, "n_dropped_gaps"] == 1


def test_purged_folds_obey_target_end_and_embargo():
    origins = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    target_end = origins + pd.Timedelta(days=7)
    validation_mask = np.arange(len(origins)) >= 60
    splits = list(purged_time_series_splits(
        origins, target_end, validation_mask=validation_mask, folds=4, embargo_days=10
    ))
    assert len(splits) == 4
    for train, valid in splits:
        validation_origin = origins[valid[0]]
        assert (target_end[train] <= validation_origin).all()
        assert (origins[train] <= validation_origin - pd.Timedelta(days=10)).all()


def test_non_overlapping_offsets_are_disjoint_and_coverage_tests_return_finite_values():
    frame = pd.DataFrame({"day": np.arange(31)})
    samples = [non_overlapping(frame, horizon=7, offset=i)["day"].to_numpy() for i in range(7)]
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            assert set(samples[left]).isdisjoint(samples[right])

    hits = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    kupiec = kupiec_uc(hits, 0.05)
    conditional = christoffersen_cc(hits, 0.05)
    assert np.isfinite(kupiec["pvalue"])
    assert np.isfinite(conditional["pvalue_cc"])


def test_fz0_and_dm_contracts():
    losses = np.array([0.01, 0.03, 0.02, 0.08, 0.04, 0.02, 0.01, 0.06])
    var = np.full(len(losses), 0.04)
    es = np.full(len(losses), 0.06)
    scores = fz0_loss(losses, var, es, confidence=0.95)
    assert scores.shape == losses.shape
    with pytest.raises(ValueError, match="ES > VaR"):
        fz0_loss(losses, var, var, confidence=0.95)
    result = dm_test(scores, scores + np.linspace(0, 0.2, len(scores)), horizon=1, hac_lags=2)
    assert result["n_obs"] == len(losses)
    assert result["pvalue"] < 1.0

    z2 = acerbi_szekely_z2(
        np.linspace(-2.0, 6.0, 100),
        np.full(100, 2.0),
        np.full(100, 3.0),
        confidence=0.95,
        simulations=500,
        seed=31,
    )
    assert 0 <= z2["pvalue"] <= 1
    assert z2["simulations"] == 500
    assert "Monte Carlo" in z2["pvalue_method"]


def test_dq_test_uses_chi_square_reference_and_benchmark_r2_is_zero():
    rng = np.random.default_rng(17)
    hits = (rng.random(100) < 0.05).astype(int)
    result = dq_test(hits, np.linspace(0.02, 0.06, 100), confidence=0.95, lags=4)
    assert result["reference_distribution"] == "chi2"
    assert 0 <= result["pvalue"] <= 1

    actual = np.linspace(0.001, 0.02, 30)
    forecast = np.linspace(0.0001, 0.02, 30)
    metrics = volatility_metrics(actual, forecast, benchmark=forecast, train_floor=0.001)
    assert metrics["r2_oos_vs_benchmark"] == 0.0


def test_mcs_keeps_duplicate_loss_models_and_accepts_list_inclusion():
    pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(19)
    loss = pd.DataFrame({
        "model_a": rng.normal(size=30),
        "model_a_duplicate": np.nan,
        "model_b": rng.normal(size=30),
    })
    loss["model_a_duplicate"] = loss["model_a"]
    result = model_confidence_set(loss, alpha=0.10, bootstrap=100, block=5, seed=11)
    assert set(result["model"]) == set(loss.columns)
    duplicates = result.loc[result["model"].eq("model_a_duplicate")]
    assert duplicates.iloc[0]["included"] == result.loc[result["model"].eq("model_a"), "included"].iloc[0]


def test_tuning_uses_search_space_with_purged_validation_folds():
    rng = np.random.default_rng(23)
    origins = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    x = pd.DataFrame(rng.normal(size=(500, 3)), index=origins, columns=["x1", "x2", "x3"])
    y = pd.Series(np.exp(0.1 * x["x1"] - 0.05 * x["x2"] + rng.normal(scale=0.05, size=500)), index=origins)
    target_end = pd.Series(origins + pd.Timedelta(days=1), index=origins)
    params, score = tune_hyperparameters(
        "elasticnet", x, y, target_end, folds=3, embargo_days=29, max_candidates=2, seed=7
    )
    assert set(params) == {"alpha", "l1_ratio"}
    assert score["folds_used"] >= 2
    assert np.isfinite(score["qlike_cv"])


def test_forecast_request_exposes_distinct_production_mode():
    request = ForecastRequest(symbol="BTCUSDT", mode="production")
    assert request.mode == "production"


def test_walkforward_forecast_is_invariant_to_data_after_origin(monkeypatch):
    research = {
        "as_of_utc": "2020-07-01T00:00:00Z",
        "windows": {
            "burn_in_start": "2020-01-01",
            "first_train_start": "2020-01-01",
            "oos_start": "2020-03-01",
        },
        "refit_every_days": {"garch": 7, "har": 7, "linear_ml": 7, "trees": 30},
        "tuning": {"retune_every_days": 90, "folds": 2, "embargo_days": 29},
        "seed": 7,
    }
    monkeypatch.setattr("app.evaluation.walkforward.load_research_config", lambda: research)
    daily = _daily_frame(days=180)
    origin = pd.Timestamp("2020-05-20T00:00:00Z")
    full_dataset = build_dataset(daily, DatasetSpec(horizon=1), as_of=research["as_of_utc"])
    full, _ = walk_forward(daily, full_dataset, "BTCUSDT", 1, models=["random_walk"])
    baseline = full.loc[full["origin_time"].eq(origin), "rv_forecast"].iloc[0]

    changed = daily.copy()
    changed.loc[changed["utc_date"] >= origin, ["rv", "close_ret"]] *= 1000
    changed_dataset = build_dataset(changed, DatasetSpec(horizon=1), as_of=research["as_of_utc"])
    altered, _ = walk_forward(changed, changed_dataset, "BTCUSDT", 1, models=["random_walk"])
    altered_value = altered.loc[altered["origin_time"].eq(origin), "rv_forecast"].iloc[0]
    assert altered_value == baseline


def test_walkforward_resume_matches_full_forecast_suffix(monkeypatch):
    research = {
        "as_of_utc": "2020-07-01T00:00:00Z",
        "windows": {
            "burn_in_start": "2020-01-01",
            "first_train_start": "2020-01-01",
            "oos_start": "2020-03-01",
        },
        "refit_every_days": {"garch": 7, "har": 7, "linear_ml": 7, "trees": 30},
        "tuning": {"retune_every_days": 90, "folds": 2, "embargo_days": 29},
        "seed": 7,
    }
    monkeypatch.setattr("app.evaluation.walkforward.load_research_config", lambda: research)
    daily = _daily_frame(days=180)
    dataset = build_dataset(daily, DatasetSpec(horizon=1), as_of=research["as_of_utc"])
    full, _ = walk_forward(daily, dataset, "BTCUSDT", 1, models=["random_walk"])
    split = pd.Timestamp("2020-05-01T00:00:00Z")
    resumed, _ = walk_forward(daily, dataset, "BTCUSDT", 1, models=["random_walk"], start_after=split)
    expected = full.loc[full["origin_time"].gt(split), ["origin_time", "rv_forecast"]].reset_index(drop=True)
    actual = resumed[["origin_time", "rv_forecast"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_kupiec_detects_a_known_excess_exception_rate():
    rng = np.random.default_rng(2026)
    well_calibrated = rng.binomial(1, 0.05, size=20_000)
    misspecified = rng.binomial(1, 0.10, size=20_000)
    assert abs(well_calibrated.mean() - 0.05) < 0.01
    assert kupiec_uc(well_calibrated, 0.05)["pvalue"] > 0.01
    assert kupiec_uc(misspecified, 0.05)["pvalue"] < 0.01


def test_production_forecast_rejects_a_stale_local_feed(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 18, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(forecast_service, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        forecast_service,
        "load_market_data",
        lambda symbol, interval: pd.DataFrame({
            "open_time": [pd.Timestamp("2026-10-17T23:00:00Z")],
        }),
    )
    with pytest.raises(ValueError, match="feed is stale"):
        forecast_service.create_production_forecast("BTCUSDT", "random_walk", 1, [0.95])
