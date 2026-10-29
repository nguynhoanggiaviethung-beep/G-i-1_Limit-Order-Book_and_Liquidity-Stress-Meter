from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_six_month_figures import _development_partitions
from scripts.build_six_month_report_tables import _markdown_table, deep_model_metrics, feature_summary


def _feature_tree(tmp_path, dates):
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for date in dates:
            path = tmp_path / "features" / f"symbol={symbol}" / f"date={date}" / "features_1s.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"")
    return SimpleNamespace(data_dir=tmp_path, lob_untouched_test_start_utc="2026-08-24T00:00:00Z")


def test_descriptive_figures_never_read_test_partitions(tmp_path):
    settings = _feature_tree(tmp_path, ["2026-08-22", "2026-08-23", "2026-08-24", "2026-09-01"])
    paths = _development_partitions(settings, "BTCUSDT", pd.Timestamp("2026-08-01", tz="UTC"), pd.Timestamp("2026-08-24", tz="UTC"))
    assert [p.parent.name for p in paths] == ["date=2026-08-22", "date=2026-08-23"]
    with pytest.raises(RuntimeError, match="development data"):
        _development_partitions(settings, "BTCUSDT", pd.Timestamp("2026-08-01", tz="UTC"), pd.Timestamp("2026-08-25", tz="UTC"))


def test_feature_summary_skips_test_dates_and_invalid_books(tmp_path, monkeypatch):
    from app.models import lob_classifiers

    monkeypatch.setattr("scripts.build_six_month_report_tables.LOB_CLASSIFICATION_FEATURES", ("obi_qty_1",))
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for date, value in (("2026-08-23", 1.0), ("2026-08-24", 100.0)):
            path = tmp_path / "features" / f"symbol={symbol}" / f"date={date}" / "features_1s.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame({
                "timestamp_utc": pd.date_range(f"{date}T00:00:00Z", periods=4, freq="s"),
                "book_valid": [True, True, True, False],
                "obi_qty_1": [value, value, np.nan, 999.0],
            }).to_parquet(path)
    settings = SimpleNamespace(data_dir=tmp_path, lob_untouched_test_start_utc="2026-08-24T00:00:00Z")
    table = feature_summary(settings).set_index("symbol")
    assert table.loc["BTCUSDT", "n_valid_seconds"] == 3
    assert table.loc["BTCUSDT", "mean"] == 1.0  # neither the test day nor the invalid row leaks in
    assert table.loc["BTCUSDT", "missing_share"] == pytest.approx(1 / 3)
    assert lob_classifiers.LOB_CLASSIFICATION_FEATURES  # the real list is untouched


def test_deep_model_table_counts_weeks_and_refuses_test_results(tmp_path):
    results = []
    for week, (deep, logistic) in enumerate([(0.2, 0.3), (0.4, 0.3), (0.1, 0.3)]):
        for task, metric in (("direction", "mcc"), ("stress", "pr_auc")):
            results.append({"symbol": "BTCUSDT", "task": task, "week_utc": f"2026-06-0{week + 1}", "metrics": {
                "deeplob": {metric: deep, "macro_f1": 0.5}, "logistic": {metric: logistic, "macro_f1": 0.6}}})
    path = tmp_path / "deeplob.json"
    path.write_text(json.dumps({"holdout_accessed": False, "holdout_scored": False, "train_cap_per_fit": 8000,
                                "validation_row_cap_per_task": 2500, "results": results}), encoding="utf-8")
    table = deep_model_metrics(path).set_index(["task", "model"])
    assert table.loc[("direction", "logistic"), "weeks_deeplob_better_on_mcc"] == 1
    assert table.loc[("stress", "logistic"), "weeks_deeplob_better_on_pr_auc"] == 1
    assert table.loc[("direction", "deeplob"), "mcc_mean"] == pytest.approx(0.7 / 3)

    path.write_text(json.dumps({"holdout_accessed": True, "results": results}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="development-only"):
        deep_model_metrics(path)


def test_markdown_table_formats_floats_and_blanks_missing_values():
    text = _markdown_table(pd.DataFrame({"model": ["a|b", "c"], "mcc": [0.123456, np.nan]}))
    assert text.splitlines() == ["| model | mcc |", "|---|---|", "| a\\|b | 0.1235 |", "| c |  |"]
