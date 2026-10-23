from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from app.evaluation.lob_classification import classification_metrics
from app.data.grid_store import GridStore
from app.features.classification_features import build_classification_features
from app.labels.direction import DIRECTION_LABEL_SCHEMA_VERSION, fit_direction_delta, make_direction_labels
from app.labels.stress import STRESS_LABEL_SCHEMA_VERSION, StressThresholds, make_stress_labels
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION, make_classifiers
from app.services.replay_service import ReplayService
from scripts import run_deeplob_classification, run_lob_classification
from scripts.run_lob_classification import _apply_isotonic, _study_provenance


EXPECTED_FEATURES = (
    "spread_ticks", "relative_spread_bps", "microprice_minus_mid_bps",
    "obi_qty_1", "obi_qty_5", "obi_qty_10", "obi_qty_20",
    "bid_depth_qty_5", "ask_depth_qty_5", "bid_depth_qty_10", "ask_depth_qty_10",
    "bid_depth_qty_20", "ask_depth_qty_20", "depth_notional_10",
    "ofi_1s", "ofi_5s", "ofi_10s", "trade_imbalance_5s", "trade_imbalance_10s",
    "price_impact_bps", "addition_intensity_1s", "cancellation_intensity_1s",
    "addition_intensity_5s", "cancellation_intensity_5s", "rv_10s", "rv_60s", "rv_300s",
)


def _grid_row(timestamp: pd.Timestamp, index: int) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": timestamp,
        "symbol": "BTCUSDT",
        "book_valid": True,
        "tick_size": 2.0,
        "ofi_1s": float(index + 1),
        "aggressive_buy_volume_1s": float(index + 2),
        "aggressive_sell_volume_1s": 1.0,
        "addition_intensity_1s": float(index),
        "cancellation_intensity_1s": float(index) / 2,
    }
    for level in range(1, 21):
        row[f"bid_price_{level}"] = 100.0 - (level - 1)
        row[f"bid_qty_{level}"] = 2.0
        row[f"ask_price_{level}"] = 102.0 + (level - 1)
        row[f"ask_qty_{level}"] = 1.0
    return row


def test_feature_registry_has_27_ordered_columns_all_created_by_builder():
    assert LOB_CLASSIFICATION_FEATURES == EXPECTED_FEATURES
    assert len(LOB_CLASSIFICATION_FEATURES) == len(set(LOB_CLASSIFICATION_FEATURES)) == 27
    times = pd.date_range("2026-10-01T00:00:00Z", periods=3, freq="s")
    features = build_classification_features(
        pd.DataFrame([_grid_row(timestamp, i) for i, timestamp in enumerate(times)]),
        impact_notional_usdt=100.0,
    )
    assert all(name in features.columns for name in LOB_CLASSIFICATION_FEATURES)


def test_study_provenance_records_config_schema_data_window_and_seed():
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2026-10-01T00:00:00Z", "2026-10-01T00:00:01Z"]),
        "symbol": ["BTCUSDT", "BTCUSDT"],
    })
    settings = SimpleNamespace(
        lob_first_validation_week_utc="2026-10-05T00:00:00Z",
        lob_development_end_utc="2026-10-11T23:59:59Z",
        lob_untouched_test_start_utc="2026-10-12T00:00:00Z",
        lob_untouched_test_end_utc="2026-10-18T23:59:59Z",
    )
    provenance = _study_provenance(
        frame,
        settings,
        (1, 5, 10),
        run_lob_classification.REPOSITORY_ROOT / "config" / "settings.yaml",
        "historical_archive_pilot",
    )
    assert provenance["configuration_file"] == "config/settings.yaml"
    assert len(provenance["configuration_sha256"]) == 64
    assert provenance["feature_schema_version"] == "lob-classification-features.v1"
    assert provenance["direction_label_schema_version"] == "midprice-direction.v1"
    assert provenance["stress_label_schema_version"] == "liquidity-stress.v1"
    assert provenance["python_version"]
    assert {"numpy", "pandas", "scikit-learn"}.issubset(provenance["dependency_versions"])
    assert provenance["observed_data_start_utc"] == "2026-10-01T00:00:00+00:00"
    assert provenance["observed_data_end_utc"] == "2026-10-01T00:00:01+00:00"
    assert provenance["symbols"] == ["BTCUSDT"]
    assert provenance["horizons_seconds"] == [1, 5, 10]
    assert provenance["random_seeds"]["sklearn_estimators"] == 42


def test_obi_ofi_and_rolling_features_match_hand_calculation():
    times = pd.date_range("2026-10-01T00:00:00Z", periods=5, freq="s")
    features = build_classification_features(
        pd.DataFrame([_grid_row(timestamp, i) for i, timestamp in enumerate(times)]),
        impact_notional_usdt=100.0,
    )
    assert features.loc[0, "obi_qty_1"] == pytest.approx(1 / 3)
    assert features.loc[4, "ofi_5s"] == pytest.approx(15.0)
    assert features.loc[4, "trade_imbalance_5s"] == pytest.approx((20 - 5) / 25)
    assert features.loc[4, "addition_intensity_5s"] == pytest.approx(10.0)


def test_rolling_features_reset_at_gap_and_prior_features_ignore_future_rows():
    seconds = [0, 1, 2, 3, *range(5, 17)]
    times = pd.DatetimeIndex(pd.Timestamp("2026-10-01T00:00:00Z") + pd.to_timedelta(seconds, unit="s"))
    raw = pd.DataFrame([_grid_row(timestamp, i) for i, timestamp in enumerate(times)])
    features = build_classification_features(raw, impact_notional_usdt=100.0)
    assert pd.isna(features.loc[4, "ofi_5s"])
    assert features.loc[8, "ofi_5s"] == pytest.approx(sum(range(5, 10)))
    assert np.isfinite(features.loc[14, "rv_10s"])

    changed_future = raw.copy()
    for index in range(4, len(changed_future)):
        changed_future.loc[index, "ofi_1s"] = 1_000_000.0
        changed_future.loc[index, "aggressive_buy_volume_1s"] = 1_000_000.0
        for level in range(1, 21):
            changed_future.loc[index, f"bid_price_{level}"] += 100.0
            changed_future.loc[index, f"ask_price_{level}"] += 100.0
    changed_features = build_classification_features(changed_future, impact_notional_usdt=100.0)
    pd.testing.assert_series_equal(
        features.loc[3, list(LOB_CLASSIFICATION_FEATURES)],
        changed_features.loc[3, list(LOB_CLASSIFICATION_FEATURES)],
        check_names=False,
    )


def test_duplicate_timestamps_are_stably_resolved_and_logged(caplog):
    timestamp = pd.Timestamp("2026-10-01T00:00:00Z")
    first = _grid_row(timestamp, 0)
    second = _grid_row(timestamp, 1)
    second["bid_qty_1"] = 9.0
    with caplog.at_level(logging.WARNING, logger="app.features.classification_features"):
        features = build_classification_features(pd.DataFrame([first, second]), impact_notional_usdt=100.0)
    assert len(features) == 1
    assert features.loc[0, "best_bid_qty"] == 9.0
    assert "1 duplicate order-book feature timestamp" in caplog.text


def test_grid_store_logs_and_stably_resolves_duplicate_partition_rows(monkeypatch, tmp_path, caplog):
    first = tmp_path / "symbol=BTCUSDT" / "date=2026-10-01" / "book_1s.parquet"
    second = tmp_path / "symbol=BTCUSDT" / "date=2026-10-02" / "book_1s.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    timestamp = pd.Timestamp("2026-10-01T00:00:00Z")

    def read_parquet(path):
        value = 1 if path == first else 2
        return pd.DataFrame({"timestamp_utc": [timestamp], "book_valid": [True], "value": [value]})

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    store = GridStore(tmp_path)
    with caplog.at_level(logging.WARNING, logger="app.data.grid_store"):
        rows = store.read("BTCUSDT", start="2026-10-01", end="2026-10-02")
    assert len(rows) == 1
    assert rows.iloc[0]["value"] == 2
    assert "1 duplicate replay timestamp" in caplog.text


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_direction_and_stress_labels_align_exactly_to_horizon_timestamp(horizon):
    times = pd.date_range("2026-10-01T00:00:00Z", periods=horizon + 2, freq="s")
    frame = pd.DataFrame({
        "timestamp_utc": times,
        "book_valid": True,
        "mid_price": np.arange(horizon + 2, dtype=float) + 100.0,
        "spread_ticks": [3.0 if i == horizon else 1.0 for i in range(horizon + 2)],
        "price_impact_bps": 1.0,
        "depth_notional_10": 100.0,
        "rv_60s": 0.1,
    })
    direction = make_direction_labels(frame, horizon_seconds=horizon, delta=0.5)
    thresholds = StressThresholds(2.0, 2.0, 50.0, 1.0, 0.05, definition="composite")
    stress = make_stress_labels(frame, horizon_seconds=horizon, thresholds=thresholds)
    assert direction.iloc[0] == "up"
    assert direction.iloc[1] == "up"
    assert pd.isna(direction.iloc[-1])
    assert stress.iloc[0] == 1
    assert stress.iloc[1] == 0
    assert pd.isna(stress.iloc[-1])


def test_direction_delta_is_fitted_from_the_supplied_training_window_only():
    times = pd.date_range("2026-10-01T00:00:00Z", periods=8, freq="s")
    frame = pd.DataFrame({"timestamp_utc": times, "book_valid": True, "mid_price": [100, 101, 101, 102, 102, 103, 103, 104]})
    train = frame.iloc[:5].copy()
    delta = fit_direction_delta(train, horizon_seconds=1, quantile=0.5)
    changed_heldout = frame.copy()
    changed_heldout.loc[5:, "mid_price"] = [10_000, 1, 999]
    assert fit_direction_delta(train, horizon_seconds=1, quantile=0.5) == delta
    assert fit_direction_delta(changed_heldout.iloc[:5], horizon_seconds=1, quantile=0.5) == delta


def test_classifier_imputer_is_fitted_on_training_data_and_not_validation():
    model = make_classifiers("direction")["logistic"]
    x_train = pd.DataFrame({"missing_feature": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "other": [0, 1, 0, 1, 0, 1]})
    y_train = np.array(["down", "up", "flat", "up", "down", "flat"])
    model.fit(x_train, y_train)
    imputer = model.named_steps["simpleimputer"]
    assert imputer.statistics_[0] == pytest.approx(4.0)
    model.predict_proba(pd.DataFrame({"missing_feature": [1_000_000.0], "other": [0]}))
    assert imputer.statistics_[0] == pytest.approx(4.0)


def test_imbalance_metrics_report_class_sensitive_scores():
    truth = np.array([0] * 98 + [1] * 2)
    prediction = np.zeros_like(truth)
    probabilities = np.tile([0.95, 0.05], (len(truth), 1))
    metrics = classification_metrics(truth, prediction, probabilities, labels=[0, 1], stress=True)
    assert metrics["n"] == 100
    assert metrics["positive_rate"] == pytest.approx(0.02)
    assert metrics["recall_by_class"]["1"] == 0.0
    assert metrics["pr_auc"] == pytest.approx(0.02)
    assert metrics["brier"] is not None


def test_calibrated_multiclass_probabilities_are_renormalized():
    calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.0, 0.2, 0.8, 1.0], [0, 0, 1, 1])
    raw = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
    calibrated = _apply_isotonic(raw, [calibrator, calibrator, calibrator])
    assert np.isfinite(calibrated).all()
    assert np.all(calibrated >= 0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("runner", "filename", "metrics_root", "settings"),
    [
        (
            run_lob_classification,
            "classification_metrics.json",
                "outputs/tables/historical_archive_pilot",
            SimpleNamespace(
                symbols=("BTCUSDT",), lob_classification_horizons_seconds=(5,),
                lob_embargo_seconds=300, lob_feature_lookback_seconds=300,
                lob_deeplob_lookback_seconds=100,
            ),
        ),
        (
            run_deeplob_classification,
            "deeplob_metrics.json",
                "outputs/tables",
            SimpleNamespace(symbols=("BTCUSDT",), lob_classification_horizons_seconds=(5,)),
        ),
    ],
)
def test_final_test_runners_refuse_a_repeat_without_explicit_override(monkeypatch, tmp_path, runner, filename, metrics_root, settings):
    output = tmp_path / metrics_root / "BTCUSDT" / filename
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_settings", lambda _path: settings)
    monkeypatch.setattr(sys, "argv", ["runner.py", "--symbol", "BTCUSDT", "--horizon", "5"])
    with pytest.raises(SystemExit, match="already exists"):
        runner.main()


def test_api_returns_unavailable_for_missing_artifact_and_incompatible_schema(monkeypatch, tmp_path):
    service = ReplayService(artifact_root=tmp_path, max_missing_feature_fraction=0.1)
    missing_artifact = service.predict_features("BTCUSDT", None, 5, {})
    assert missing_artifact["status"] == "unavailable"
    assert missing_artifact["p_up"] is None

    monkeypatch.setattr(service, "_bundle", lambda *_: {
        "feature_columns": ["unrecognized_feature"],
        "stress_probability_threshold": 0.37,
    })
    incompatible_schema = service.predict_features("BTCUSDT", None, 5, {})
    assert incompatible_schema["status"] == "unavailable"
    assert incompatible_schema["p_stress"] is None
    assert incompatible_schema["stress_probability_threshold"] == 0.37
    assert incompatible_schema["quality"]["missing_features"] == ["unrecognized_feature"]


def test_api_rejects_invalid_book_even_when_features_are_present(monkeypatch, tmp_path):
    service = ReplayService(artifact_root=tmp_path)
    monkeypatch.setattr(service, "_bundle", lambda *_: {
        "feature_columns": list(LOB_CLASSIFICATION_FEATURES),
        "stress_probability_threshold": 0.37,
    })
    features = {name: 1.0 for name in LOB_CLASSIFICATION_FEATURES}
    response = service.predict_features("BTCUSDT", None, 5, features, {"book_valid": False})
    assert response["status"] == "unavailable"
    assert response["reason"] == "The supplied order book is invalid"
    assert response["p_up"] is None


def test_api_accepts_exactly_ten_percent_missing_features(monkeypatch, tmp_path):
    class ProbabilityModel:
        def __init__(self, classes, probabilities):
            self.classes_ = classes
            self.probabilities = probabilities

        def predict_proba(self, _frame):
            return np.asarray([self.probabilities])

    service = ReplayService(artifact_root=tmp_path, max_missing_feature_fraction=0.1)
    columns = [f"feature_{index}" for index in range(10)]
    monkeypatch.setattr(service, "_bundle", lambda *_: {
        "feature_columns": columns,
        "direction_model": ProbabilityModel(["down", "flat", "up"], [0.2, 0.5, 0.3]),
        "stress_model": ProbabilityModel([0, 1], [0.7, 0.3]),
        "direction_class_order": ["down", "flat", "up"],
        "stress_probability_threshold": 0.4,
    })
    values = {name: 1.0 for name in columns}
    values[columns[-1]] = None
    response = service.predict_features("BTCUSDT", None, 5, values)
    assert response["status"] == "ok"
    assert response["quality"]["missing_feature_fraction"] == pytest.approx(0.1)


def test_loaded_artifact_threshold_is_returned_unchanged(tmp_path):
    artifact = tmp_path / "BTCUSDT" / "horizon_5.joblib"
    artifact.parent.mkdir(parents=True)
    joblib.dump({
        "symbol": "BTCUSDT", "horizon_seconds": 5,
        "feature_columns": ["missing_feature"],
        "stress_probability_threshold": 0.37,
    }, artifact)
    service = ReplayService(artifact_root=tmp_path)
    response = service.predict_features("BTCUSDT", None, 5, {})
    assert response["status"] == "unavailable"
    assert response["stress_probability_threshold"] is None


def test_artifact_loader_requires_matching_target_and_schema_metadata(tmp_path):
    path = tmp_path / "BTCUSDT" / "horizon_5.joblib"
    path.parent.mkdir(parents=True)
    metadata = {
        "symbol": "BTCUSDT",
        "horizon_seconds": 5,
        "feature_columns": list(LOB_CLASSIFICATION_FEATURES),
        "feature_schema_version": LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION,
        "label_schema_versions": {"direction": DIRECTION_LABEL_SCHEMA_VERSION, "stress": STRESS_LABEL_SCHEMA_VERSION},
        "target_names": {"direction": "midprice_direction", "stress": "liquidity_stress"},
        "direction_model": {"fixture": "direction"},
        "stress_model": {"fixture": "stress"},
    }
    joblib.dump(metadata, path)
    service = ReplayService(artifact_root=tmp_path)
    assert service._bundle("BTCUSDT", 5) is not None

    incompatible = dict(metadata)
    incompatible["feature_schema_version"] = "lob-classification-features.old"
    joblib.dump(incompatible, path)
    service = ReplayService(artifact_root=tmp_path)
    assert service._bundle("BTCUSDT", 5) is None


def test_api_rejects_artifact_for_another_symbol(tmp_path):
    artifact = tmp_path / "BTCUSDT" / "horizon_5.joblib"
    artifact.parent.mkdir(parents=True)
    joblib.dump({"symbol": "ETHUSDT", "horizon_seconds": 5}, artifact)
    service = ReplayService(artifact_root=tmp_path)
    response = service.predict_features("BTCUSDT", None, 5, {})
    assert response["status"] == "unavailable"
    assert response["p_up"] is None
