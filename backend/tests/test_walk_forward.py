from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.config import REPOSITORY_ROOT, Settings, load_settings
from app.evaluation.lob_classification import expanding_week_folds
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES
from scripts import run_lob_classification
from scripts.run_lob_classification import _bounded_sample, _sampled_training, run_symbol, write_fold_tables


SETTINGS_6M = REPOSITORY_ROOT / "config" / "settings_6m.yaml"


def test_six_month_settings_define_twelve_weekly_folds_before_a_five_week_test():
    settings = load_settings(SETTINGS_6M)
    folds = expanding_week_folds(settings.lob_first_validation_week_utc, settings.lob_development_end_utc)
    assert len(folds) == 12
    assert folds[0] == "2026-06-01" and folds[-1] == "2026-08-17"
    test_span = pd.Timestamp(settings.lob_untouched_test_end_utc) - pd.Timestamp(settings.lob_untouched_test_start_utc)
    assert test_span == pd.Timedelta(days=35) - pd.Timedelta(seconds=1)


def test_untouched_test_window_must_be_whole_weeks():
    with pytest.raises(ValueError, match="full UTC weeks"):
        Settings(
            lob_first_validation_week_utc="2026-06-01T00:00:00Z",
            lob_development_end_utc="2026-08-23T23:59:59Z",
            lob_untouched_test_start_utc="2026-08-24T00:00:00Z",
            lob_untouched_test_end_utc="2026-09-02T23:59:59Z",
        )


def test_sampling_before_slicing_selects_the_same_rows_as_the_full_slice():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(5000, 3)), columns=["a", "b", "c"])
    labels = pd.Series(rng.choice(["down", "flat", "up"], size=5000, p=[0.2, 0.5, 0.3]), dtype=object)
    mask = pd.Series(rng.random(5000) < 0.7)
    expected_x, expected_y = _bounded_sample(frame.loc[mask, ["a", "b"]], labels.loc[mask], 800)
    x, y = _sampled_training(frame, labels, mask, ["a", "b"], 800)
    assert y.index.equals(expected_y.index)
    pd.testing.assert_frame_equal(x, expected_x)


def _synthetic_day(day: str, rows: int, seed: int, source: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{day}T00:00:00Z", periods=rows, freq="1s")
    mid = 100.0 + np.cumsum(rng.normal(0, 0.02, rows))
    spread_ticks = rng.integers(1, 6, rows).astype(float)
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in LOB_CLASSIFICATION_FEATURES})
    frame["timestamp_utc"] = times
    frame["symbol"] = "BTCUSDT"
    frame["book_valid"] = True
    frame["book_source"] = source
    frame["mid_price"] = mid
    frame["spread_ticks"] = spread_ticks
    frame["absolute_spread"] = spread_ticks * 0.01
    frame["relative_spread_bps"] = frame["absolute_spread"] / mid * 10000
    frame["price_impact_bps"] = rng.gamma(2.0, 1.0, rows)
    frame["depth_notional_10"] = rng.lognormal(10, 0.5, rows)
    frame["depth_qty_10"] = frame["depth_notional_10"] / mid
    frame["rv_60s"] = rng.gamma(2.0, 1e-8, rows)
    frame["rv_300s"] = rng.gamma(2.0, 1e-8, rows)
    return frame


def test_development_only_run_never_opens_test_partitions_and_writes_fold_tables(tmp_path):
    feature_root = tmp_path / "features" / "symbol=BTCUSDT"
    for offset, day in enumerate(pd.date_range("2026-06-01", "2026-06-28", freq="D").strftime("%Y-%m-%d")):
        path = feature_root / f"date={day}" / "features_1s.parquet"
        path.parent.mkdir(parents=True)
        if day >= "2026-06-22":
            path.write_bytes(b"untouched test partition must not be opened")
            continue
        _synthetic_day(day, 1200, offset, "warm_start" if day < "2026-06-08" else "snapshot").to_parquet(path, index=False)
    settings = SimpleNamespace(
        data_dir=tmp_path,
        lob_first_validation_week_utc="2026-06-08T00:00:00Z",
        lob_development_end_utc="2026-06-21T23:59:59Z",
        lob_untouched_test_start_utc="2026-06-22T00:00:00Z",
        lob_untouched_test_end_utc="2026-06-28T23:59:59Z",
        lob_direction_delta_quantile=0.5,
        lob_stress_definition="composite",
        lob_stress_quantile=0.975,
        lob_stress_depth_quantile=0.025,
    )

    result = run_symbol(
        "BTCUSDT", (5,), 300, 1500, settings, SETTINGS_6M, tmp_path / "artifacts", "unit", score_final_test=False,
    )

    assert result["final_test_scored"] is False
    assert result["validation_weeks_utc"] == ["2026-06-08", "2026-06-15"]
    assert not (tmp_path / "artifacts").exists()
    block = result["horizons"]["5"]
    assert "direction_test" not in block and "stress_test" not in block
    manifest = pd.DataFrame(block["fold_manifest"])
    assert len(manifest) == 2
    for row in manifest.itertuples():
        # Training ends one embargo before validation starts, and no later.
        assert pd.Timestamp(row.validation_start_utc) - pd.Timestamp(row.train_end_exclusive_utc) == pd.Timedelta(seconds=300)
        assert row.direction_train_rows_fitted <= 1500
    assert manifest["direction_train_rows"].is_monotonic_increasing
    assert manifest.loc[0, "warm_start_share_train"] == pytest.approx(1.0, abs=0.01)
    assert manifest.loc[0, "warm_start_share_validation"] == 0.0
    assert [row["validation_week_utc"] for row in block["stress_thresholds_by_fold"]] == ["2026-06-08", "2026-06-15"]
    models = {row["model"] for row in block["walk_forward_direction"]}
    assert {"logistic", "class_frequency", "direction_persistence", "logistic_obi_only"} <= models

    symbol_dir = tmp_path / "out" / "BTCUSDT"
    symbol_dir.mkdir(parents=True)
    write_fold_tables(result, symbol_dir)
    metrics = pd.read_csv(symbol_dir / "fold_metrics.csv")
    assert set(metrics["week_utc"]) == {"2026-06-08", "2026-06-15"}
    assert {"direction", "stress"} == set(metrics["task"])
    assert "logistic_depth_levels_L=20" in set(metrics["model"])
    assert len(pd.read_csv(symbol_dir / "fold_manifest.csv")) == 2


def test_development_only_does_not_trip_the_final_test_repeat_guard(monkeypatch, tmp_path):
    existing = tmp_path / "out" / "BTCUSDT" / "classification_metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_lob_classification, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(run_lob_classification, "load_settings", lambda _path: SimpleNamespace(
        symbols=("BTCUSDT",), lob_classification_horizons_seconds=(5,), lob_embargo_seconds=300,
        lob_feature_lookback_seconds=300, lob_deeplob_lookback_seconds=100,
    ))
    monkeypatch.setattr(run_lob_classification, "run_symbol", lambda *args, **kwargs: calls.append(kwargs) or {"symbol": "BTCUSDT", "horizons": {}})
    monkeypatch.setattr(run_lob_classification.sys, "argv", [
        "runner.py", "--symbol", "BTCUSDT", "--horizon", "5", "--development-only", "--metrics-dir", str(tmp_path / "out"),
    ])
    assert run_lob_classification.main() == 0
    assert calls == [{"score_final_test": False, "output_dir": None}]
    assert (tmp_path / "out" / "BTCUSDT" / "walk_forward_metrics.json").is_file()
    assert existing.read_text(encoding="utf-8") == "{}"


def test_opened_marker_blocks_a_second_final_test_run(monkeypatch, tmp_path):
    marker = tmp_path / "out" / "BTCUSDT" / run_lob_classification.FINAL_TEST_OPENED_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_lob_classification, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(run_lob_classification, "load_settings", lambda _path: SimpleNamespace(
        symbols=("BTCUSDT",), lob_classification_horizons_seconds=(5,), lob_embargo_seconds=300,
        lob_feature_lookback_seconds=300, lob_deeplob_lookback_seconds=100,
    ))
    monkeypatch.setattr(run_lob_classification, "run_symbol", lambda *args, **kwargs: pytest.fail("must not run"))
    monkeypatch.setattr(run_lob_classification.sys, "argv", [
        "runner.py", "--symbol", "BTCUSDT", "--horizon", "5", "--metrics-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit, match="already exists"):
        run_lob_classification.main()


def test_final_test_run_reads_only_the_test_window_and_saves_predictions(tmp_path):
    feature_root = tmp_path / "features" / "symbol=BTCUSDT"
    for offset, day in enumerate(pd.date_range("2026-06-01", "2026-06-29", freq="D").strftime("%Y-%m-%d")):
        path = feature_root / f"date={day}" / "features_1s.parquet"
        path.parent.mkdir(parents=True)
        if day > "2026-06-28":
            path.write_bytes(b"partition after the test window must not be opened")
            continue
        _synthetic_day(day, 1200, offset, "warm_start" if day < "2026-06-08" else "snapshot").to_parquet(path, index=False)
    settings = SimpleNamespace(
        data_dir=tmp_path,
        lob_first_validation_week_utc="2026-06-08T00:00:00Z",
        lob_development_end_utc="2026-06-21T23:59:59Z",
        lob_untouched_test_start_utc="2026-06-22T00:00:00Z",
        lob_untouched_test_end_utc="2026-06-28T23:59:59Z",
        lob_direction_delta_quantile=0.5,
        lob_stress_definition="composite",
        lob_stress_quantile=0.975,
        lob_stress_depth_quantile=0.025,
        lob_spread_sensitivity_quantile=0.95,
        lob_stress_min_positive_rate=0.0,
        lob_stress_max_positive_rate=1.0,
    )
    out = tmp_path / "out" / "BTCUSDT"

    result = run_symbol(
        "BTCUSDT", (1, 5), 300, 1500, settings, SETTINGS_6M, tmp_path / "artifacts", "unit",
        score_final_test=True, output_dir=out,
    )

    assert result["final_test_scored_once"] is True
    assert result["test_rows_loaded"] == 7 * 1200
    assert result["data_rows"] == 28 * 1200
    assert (out / run_lob_classification.FINAL_TEST_OPENED_MARKER).is_file()
    for horizon in ("1", "5"):
        block = result["horizons"][horizon]
        assert {"class_frequency", "direction_persistence", "logistic_obi_only", "logistic_obi_plus_ofi", "logistic", "xgboost"} <= set(block["direction_test_comparisons"])
        assert {"stress_persistence", "logistic", "xgboost"} <= set(block["stress_test_comparisons"])
        direction = pd.read_parquet(out / block["test_prediction_files"]["direction"])
        stress = pd.read_parquet(out / block["test_prediction_files"]["stress"])
        assert len(direction) == block["test_rows_direction"] and len(stress) == block["test_rows_stress"]
        times = pd.to_datetime(direction["timestamp_utc"], utc=True)
        assert times.min() >= pd.Timestamp("2026-06-22T00:00:00Z") and times.max() < pd.Timestamp("2026-06-29T00:00:00Z")
        # The selected model's saved raw probabilities reproduce its reported comparison metrics.
        selected = block["direction_model"]
        probabilities = direction[[f"{selected}_p_{label}" for label in ("down", "flat", "up")]].to_numpy()
        predicted = np.asarray(["down", "flat", "up"])[probabilities.argmax(axis=1)]
        assert (predicted == direction["y_true"].to_numpy()).mean() == pytest.approx(block["direction_test_comparisons"][selected]["accuracy"], abs=1e-9)
        assert set(stress["stress_persistence"].unique()) <= {-1, 0, 1}
