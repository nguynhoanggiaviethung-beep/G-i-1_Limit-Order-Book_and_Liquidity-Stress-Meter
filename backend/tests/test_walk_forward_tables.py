from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_walk_forward_tables import model_summary, paired_comparisons


def _metrics() -> pd.DataFrame:
    rows = []
    for index, week in enumerate(("2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22")):
        for model, f1 in (("logistic_obi_only", 0.40 + 0.01 * index), ("logistic_obi_plus_ofi", 0.42 + 0.01 * index), ("class_frequency", 0.2)):
            rows.append({"symbol": "BTCUSDT", "horizon_seconds": 5, "task": "direction", "week_utc": week, "model": model,
                         "macro_f1": f1, "mcc": f1 - 0.2, "balanced_accuracy": f1, "brier": 0.6 - f1 / 10})
        rows.append({"symbol": "BTCUSDT", "horizon_seconds": 5, "task": "direction", "week_utc": week, "model": "logistic_depth_levels_L=5",
                     "macro_f1": 0.3, "mcc": 0.1, "balanced_accuracy": 0.3, "brier": 0.6, "depth_levels_L": 5})
    return pd.DataFrame(rows)


def test_summary_skips_sensitivity_variants_and_averages_over_weeks():
    summary = model_summary(_metrics()).set_index("model")
    assert "logistic_depth_levels_L=5" not in summary.index
    assert summary.loc["logistic_obi_only", "weeks"] == 4
    assert summary.loc["logistic_obi_only", "macro_f1_mean"] == pytest.approx(0.415)


def test_paired_contrast_counts_weeks_and_respects_lower_is_better():
    paired = paired_comparisons(_metrics())
    rq2 = paired.loc[paired["question"].str.startswith("RQ2")].set_index("metric")
    assert rq2.loc["macro_f1", "mean_difference"] == pytest.approx(0.02)
    assert rq2.loc["macro_f1", "weeks_challenger_better"] == 4
    # Brier is lower for the challenger here, which counts as better.
    assert rq2.loc["brier", "weeks_challenger_better"] == 4
