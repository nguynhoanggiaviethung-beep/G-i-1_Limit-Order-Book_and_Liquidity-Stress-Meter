from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from scripts.run_lob_test_regime_bootstrap import TEST_CONTRASTS, horizon_state
from scripts.run_lob_regime_bootstrap import summarize

LABELS = ("down", "flat", "up")


def _write_predictions(tmp_path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-08-24T00:00:00Z")
    times = start + pd.to_timedelta(np.sort(rng.choice(14 * 86400, n, replace=False)), unit="s")
    truth = rng.integers(0, 3, n)
    direction = pd.DataFrame({"timestamp_utc": times, "y_true": np.asarray(LABELS)[truth], "rv_300s": rng.uniform(0, 3, n)})
    direction["direction_persistence"] = np.asarray(LABELS)[np.where(rng.uniform(size=n) < 0.5, truth, rng.integers(0, 3, n))]
    for name in ("selected_calibrated", "logistic"):
        raw = rng.dirichlet(np.ones(3), n) + 0.5 * np.eye(3)[truth]
        raw /= raw.sum(axis=1, keepdims=True)
        for k, label in enumerate(LABELS):
            direction[f"{name}_p_{label}"] = raw[:, k].astype(np.float32)
    stress_truth = (rng.uniform(size=n) < 0.1).astype(np.int8)
    stress = pd.DataFrame({"timestamp_utc": times, "y_true": stress_truth, "rv_300s": direction["rv_300s"]})
    stress["stress_persistence"] = np.where(rng.uniform(size=n) < 0.05, -1, np.where(rng.uniform(size=n) < 0.7, stress_truth, 1 - stress_truth))
    for name in ("selected_calibrated", "xgboost"):
        stress[f"{name}_p_stress"] = np.clip(0.4 * stress_truth + rng.uniform(0, 0.6, n), 0, 1).astype(np.float32)
    direction.to_parquet(tmp_path / "d.parquet")
    stress.to_parquet(tmp_path / "s.parquet")
    result = {"test_prediction_files": {"direction": "d.parquet", "stress": "s.parquet"},
              "regime_cutoffs_rv_300s_train_terciles": [1.0, 2.0], "stress_decision_threshold": 0.3}
    return direction, stress, result, start, start + pd.Timedelta(days=14) - pd.Timedelta(seconds=1)


def test_point_estimates_reproduce_metrics_from_saved_predictions(tmp_path):
    direction, stress, result, start, end = _write_predictions(tmp_path)
    state = horizon_state(tmp_path, 5, result, start, end)
    assert len(state["block_weeks"]) == 2 * 7 * 24
    models, contrasts, _ = summarize("X", state, replicates=50, seed=1, contrasts=TEST_CONTRASTS)
    pooled = models.loc[models["regime"] == "all"].set_index(["task", "model", "metric"])["estimate"]

    truth = direction["y_true"].to_numpy()
    predicted = np.asarray(LABELS)[direction[[f"logistic_p_{label}" for label in LABELS]].to_numpy().argmax(axis=1)]
    assert pooled["direction", "logistic", "macro_f1"] == pytest.approx(f1_score(truth, predicted, average="macro"), rel=1e-9)
    assert pooled["direction", "direction_persistence", "mcc"] == pytest.approx(
        matthews_corrcoef(truth, direction["direction_persistence"]), rel=1e-9)

    # Stress uses persistence-defined rows; the selected model uses the frozen threshold.
    defined = stress.loc[stress["stress_persistence"] >= 0]
    p = defined["selected_calibrated_p_stress"].to_numpy(dtype=float)
    assert pooled["stress", "selected_calibrated", "mcc"] == pytest.approx(matthews_corrcoef(defined["y_true"], p >= 0.3), rel=1e-9)
    xgb = defined["xgboost_p_stress"].to_numpy(dtype=float)
    assert pooled["stress", "xgboost", "mcc"] == pytest.approx(matthews_corrcoef(defined["y_true"], xgb >= 0.5), rel=1e-9)
    assert pooled["stress", "xgboost", "pr_auc"] == pytest.approx(average_precision_score(defined["y_true"], xgb), abs=2e-3)
    assert models.loc[(models["regime"] == "all") & (models["task"] == "stress"), "n"].eq(len(defined)).all()

    questions = set(contrasts["question"])
    assert "Headline: selected model vs persistence" in questions
    assert "Stress: XGBoost vs persistence" in questions
    assert contrasts["bootstrap_p_holm"].notna().all()


def test_predictions_outside_the_test_window_are_rejected(tmp_path):
    _, _, result, start, end = _write_predictions(tmp_path)
    with pytest.raises(ValueError, match="outside the test window"):
        horizon_state(tmp_path, 5, result, start + pd.Timedelta(days=1), end)
