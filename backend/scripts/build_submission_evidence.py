"""Build submission audit tables from saved pilot data and results (read-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def _missing_intervals(symbol: str, date: str, expected: int, timestamps: pd.Series) -> list[dict[str, Any]]:
    """Report absent UTC seconds in the calendar-day QA window."""
    if expected <= 0 or expected > 86400:
        return []
    start = pd.Timestamp(f"{date}T00:00:00Z")
    observed = np.zeros(expected, dtype=bool)
    parsed = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
    if len(parsed):
        seconds = ((parsed - start).dt.total_seconds().to_numpy(dtype=np.int64))
        seconds = seconds[(seconds >= 0) & (seconds < expected)]
        observed[np.unique(seconds)] = True
    absent = ~observed
    changes = np.diff(np.r_[False, absent, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        {
            "symbol": symbol,
            "date_utc": date,
            "missing_start_utc": (start + pd.Timedelta(int(left), unit="s")).isoformat().replace("+00:00", "Z"),
            "missing_end_utc_exclusive": (start + pd.Timedelta(int(right), unit="s")).isoformat().replace("+00:00", "Z"),
            "duration_seconds": int(right - left),
            "reason": "no grid row observed; QA does not attribute cause per second",
        }
        for left, right in zip(starts, ends)
    ]


def build_data_quality(settings, qa_root: Path, output_root: Path) -> list[dict[str, Any]]:
    grid_root = settings.data_dir / "grid"
    manifest: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for partition in sorted(grid_root.glob("symbol=*/date=*/book_1s.parquet")):
        symbol = partition.parent.parent.name.removeprefix("symbol=")
        date = partition.parent.name.removeprefix("date=")
        qa_path = qa_root / f"symbol={symbol}" / f"date={date}" / "qa.json"
        qa = _read_json(qa_path) if qa_path.is_file() else {}
        columns = [name for name in ("timestamp_utc", "event_time_utc", "book_valid", "sequence_gap", "qa_flags")]
        frame = pd.read_parquet(partition, columns=columns)
        timestamps = pd.to_datetime(frame.get("timestamp_utc", pd.Series(dtype=object)), utc=True, errors="coerce")
        valid = _bool_series(frame["book_valid"]) if "book_valid" in frame else pd.Series(False, index=frame.index)
        expected = qa.get("expected_seconds")
        expected = int(expected) if expected is not None else None
        observed = int(timestamps.notna().sum())
        valid_rows = int((valid & timestamps.notna()).sum())
        invalid_rows = int(len(frame) - valid_rows)
        missing = max(0, expected - observed) if expected is not None else None
        event_time = pd.to_datetime(frame.get("event_time_utc", pd.Series(dtype=object)), utc=True, errors="coerce", format="mixed").dropna()
        raw_files = qa.get("source_files") or []
        raw_root = settings.raw_data_dir / "binance_spot"
        existing_raw = [raw_root / Path(str(item).replace("\\", "/")) for item in raw_files]
        raw_bytes = sum(path.stat().st_size for path in existing_raw if path.is_file())
        gap_count = qa.get("sequence_gap_count")
        if gap_count is None and "sequence_gap" in frame:
            gap_count = int(_bool_series(frame["sequence_gap"]).sum())
        stale_count: int | None = None
        if "qa_flags" in frame:
            stale_count = int(frame["qa_flags"].astype(str).str.lower().str.contains("stale").sum())
        coverage = (valid_rows / expected) if expected else None
        checksum = _sha256(partition)
        row = {
            "symbol": symbol,
            "source": qa.get("source", "unknown"),
            "date_utc": date,
            "timestamp_basis": qa.get("timestamp_basis"),
            "first_event_time_utc": event_time.min().isoformat() if len(event_time) else None,
            "last_event_time_utc": event_time.max().isoformat() if len(event_time) else None,
            "first_grid_time_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
            "last_grid_time_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
            "raw_file_count_listed": len(raw_files),
            "raw_file_count_found": sum(path.is_file() for path in existing_raw),
            "raw_bytes": raw_bytes,
            "grid_rows_expected": expected,
            "grid_rows_observed": observed,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "coverage_pct": 100.0 * coverage if coverage is not None else None,
            "missing_seconds": missing,
            "sequence_gap_count": gap_count,
            "resync_count": qa.get("snapshot_recoveries"),
            "stale_book_count": stale_count,
            "duplicate_event_count": None,
            "duplicate_event_count_note": "not recorded in daily source QA",
            "checksum_sha256": checksum,
            "checksum_scope": "normalized book_1s.parquet partition",
            "collection_started_at_utc": None,
            "collection_ended_at_utc": None,
        }
        manifest.append(row)
        if expected is not None:
            missing_rows.extend(_missing_intervals(symbol, date, expected, timestamps))
        for name, value in (("sequence_gap", gap_count or 0), ("snapshot_recovery", qa.get("snapshot_recoveries", 0) or 0), ("invalid_book_event", qa.get("invalid_book_event_count", 0) or 0)):
            if int(value) > 0:
                sequence_rows.append({"symbol": symbol, "date_utc": date, "event_type": name, "count": int(value), "event_time_utc": None, "time_precision": "daily aggregate from QA"})

    base = output_root / "outputs" / "data_quality"
    manifest_columns = list(manifest[0]) if manifest else None
    _write_csv(base / "data_manifest.csv", manifest, manifest_columns)
    _write_csv(base / "coverage_by_day.csv", manifest, manifest_columns)
    _write_csv(base / "missing_intervals.csv", missing_rows, ["symbol", "date_utc", "missing_start_utc", "missing_end_utc_exclusive", "duration_seconds", "reason"])
    _write_csv(base / "sequence_events.csv", sequence_rows, ["symbol", "date_utc", "event_type", "count", "event_time_utc", "time_precision"])
    summaries = []
    for symbol in sorted({row["symbol"] for row in manifest}):
        rows = [row for row in manifest if row["symbol"] == symbol]
        expected_values = [row["grid_rows_expected"] for row in rows if row["grid_rows_expected"] is not None]
        total_expected = sum(expected_values)
        total_valid = sum(row["valid_rows"] for row in rows)
        summaries.append({
            "symbol": symbol,
            "source": rows[0]["source"],
            "partition_count": len(rows),
            "observed_start_utc": min(row["first_grid_time_utc"] for row in rows if row["first_grid_time_utc"]),
            "observed_end_utc": max(row["last_grid_time_utc"] for row in rows if row["last_grid_time_utc"]),
            "expected_seconds": total_expected,
            "observed_rows": sum(row["grid_rows_observed"] for row in rows),
            "valid_rows": total_valid,
            "invalid_rows": sum(row["invalid_rows"] for row in rows),
            "coverage_pct": 100.0 * total_valid / total_expected if total_expected else None,
            "missing_seconds": sum(row["missing_seconds"] or 0 for row in rows),
            "sequence_gaps": sum(row["sequence_gap_count"] or 0 for row in rows),
            "resynchronizations": sum(row["resync_count"] or 0 for row in rows),
            "raw_files_found": sum(row["raw_file_count_found"] for row in rows),
            "raw_bytes": sum(row["raw_bytes"] for row in rows),
        })
    _write_json(base / "quality_summary.json", {
        "schema_version": "submission_data_quality.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "study_label": "historical_archive_pilot",
        "data_root": str(grid_root),
        "partitions": len(manifest),
        "symbols": summaries,
        "notes": [
            "Coverage is computed from observed valid grid rows and daily expected_seconds in QA sidecars.",
            "Missing intervals identify seconds with no grid row; they do not infer the cause.",
            "Duplicate event counts are null because source QA does not record them.",
            "Checksums identify normalized Parquet partitions, not compressed provider source files.",
        ],
    })
    return manifest


def build_feature_audits(settings, metrics_root: Path, output_root: Path) -> None:
    feature_root = settings.data_dir / "features"
    all_samples: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in settings.symbols}
    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        totals = {name: {"rows": 0, "valid": 0, "sum": 0.0, "sum_sq": 0.0} for name in LOB_CLASSIFICATION_FEATURES}
        for partition in sorted((feature_root / f"symbol={symbol}").glob("date=*/features_1s.parquet")):
            frame = pd.read_parquet(partition, columns=["timestamp_utc", "book_valid", *LOB_CLASSIFICATION_FEATURES])
            valid = _bool_series(frame["book_valid"]) if "book_valid" in frame else pd.Series(True, index=frame.index)
            frame = frame.loc[valid]
            if frame.empty:
                continue
            for name in LOB_CLASSIFICATION_FEATURES:
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                stats = totals[name]
                stats["rows"] += len(values)
                stats["valid"] += len(finite)
                stats["sum"] += float(finite.sum())
                stats["sum_sq"] += float(np.square(finite).sum())
            step = max(1, math.ceil(len(frame) / 5000))
            all_samples[symbol].append(frame.iloc[::step][["timestamp_utc", *LOB_CLASSIFICATION_FEATURES]].copy())
        sampled = pd.concat(all_samples[symbol], ignore_index=True) if all_samples[symbol] else pd.DataFrame(columns=["timestamp_utc", *LOB_CLASSIFICATION_FEATURES])
        sampled_numeric = sampled[list(LOB_CLASSIFICATION_FEATURES)].apply(pd.to_numeric, errors="coerce")
        sampled_numeric = sampled_numeric.replace([np.inf, -np.inf], np.nan)
        correlations = sampled_numeric.corr(min_periods=2)
        for left in LOB_CLASSIFICATION_FEATURES:
            s = totals[left]
            n = s["valid"]
            mean = s["sum"] / n if n else None
            variance = max(0.0, (s["sum_sq"] - s["sum"] ** 2 / n) / (n - 1)) if n > 1 else None
            values = sampled_numeric[left].dropna()
            quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict() if len(values) else {}
            summary_rows.append({
                "symbol": symbol, "feature": left, "rows": s["rows"], "nonmissing": n,
                "missing_rate": 1 - n / s["rows"] if s["rows"] else None,
                "mean": mean, "std": math.sqrt(variance) if variance is not None else None,
                "q01_sample": quantiles.get(0.01), "q25_sample": quantiles.get(0.25),
                "q50_sample": quantiles.get(0.5), "q75_sample": quantiles.get(0.75),
                "q95_sample": quantiles.get(0.95), "q99_sample": quantiles.get(0.99),
                "quantile_sample_rows": len(values), "quantile_method": "deterministic temporal stride; max 5000 rows per day",
            })
            missing_rows.append({"symbol": symbol, "feature": left, "rows": s["rows"], "nonmissing": n, "missing": s["rows"] - n, "missing_rate": 1 - n / s["rows"] if s["rows"] else None})
        correlation_rows.extend([
            {"symbol": symbol, "feature_x": left, "feature_y": right, "correlation": correlations.loc[left, right], "sample_rows": len(sampled_numeric)}
            for left in LOB_CLASSIFICATION_FEATURES for right in LOB_CLASSIFICATION_FEATURES
        ])

    feature_dir = output_root / "outputs" / "features"
    _write_csv(feature_dir / "feature_summary.csv", summary_rows)
    _write_csv(feature_dir / "missingness_by_feature.csv", missing_rows)
    _write_csv(feature_dir / "feature_correlation.csv", correlation_rows)

    target_rows: list[dict[str, Any]] = []
    for symbol in settings.symbols:
        metrics_path = metrics_root / symbol / "classification_metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _read_json(metrics_path)
        for horizon, item in metrics.get("horizons", {}).items():
            target_rows.append({
                "symbol": symbol, "horizon_seconds": int(horizon), "target": "direction",
                "train_distribution_json": json.dumps(item.get("direction_class_distribution_train", {}), sort_keys=True),
                "validation_model": item.get("direction_model"), "holdout_rows": item.get("test_rows_direction"),
                "holdout_scored_once": metrics.get("final_test_scored_once"),
                "status": "class distribution from saved pilot metrics; pointwise labels are not persisted",
            })
            target_rows.append({
                "symbol": symbol, "horizon_seconds": int(horizon), "target": "composite_stress",
                "train_distribution_json": json.dumps(item.get("stress_class_distribution_train", {}), sort_keys=True),
                "validation_model": item.get("stress_model"), "holdout_rows": item.get("test_rows_stress"),
                "holdout_scored_once": metrics.get("final_test_scored_once"),
                "status": "class distribution from saved pilot metrics; pointwise labels are not persisted",
            })
    _write_csv(output_root / "outputs" / "features" / "feature_target_summary.csv", target_rows)


def _confusion_distribution(metric: dict[str, Any]) -> dict[str, float]:
    matrix = metric.get("confusion_matrix") or []
    labels = metric.get("labels") or []
    totals = [sum(row) for row in matrix]
    denominator = sum(totals)
    return {str(label): total / denominator for label, total in zip(labels, totals)} if denominator else {}


def _append_metric(rows: list[dict[str, Any]], *, symbol: str, horizon: int, split: str, task: str, model: str, metric: dict[str, Any]) -> None:
    rows.append({
        "symbol": symbol, "horizon_seconds": horizon, "split": split, "task": task, "model": model,
        "n": metric.get("n"), "accuracy": metric.get("accuracy"), "macro_f1": metric.get("macro_f1"),
        "balanced_accuracy": metric.get("balanced_accuracy"), "mcc": metric.get("mcc"),
        "pr_auc": metric.get("pr_auc"), "brier": metric.get("brier"), "positive_rate": metric.get("positive_rate"),
    })


def build_saved_result_tables(metrics_root: Path, output_root: Path, settings) -> None:
    metric_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    direction_classes: list[dict[str, Any]] = []
    stress_classes: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    obi_rows: list[dict[str, Any]] = []
    stress_sensitivity_rows: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {"study_label": "historical_archive_pilot", "symbols": {}}
    fold_rows: list[dict[str, Any]] = []
    leakage: dict[str, Any] = {
        "study_label": "historical_archive_pilot",
        "configured_embargo_seconds": settings.lob_embargo_seconds,
        "minimum_required_embargo_seconds": max(max(settings.lob_classification_horizons_seconds), settings.lob_feature_lookback_seconds, settings.lob_deeplob_lookback_seconds),
        "random_shuffling_used": False,
        "final_test_re_scored_by_this_audit": False,
        "final_test_metrics_used_for_model_selection": False,
        "notes": ["This script reads saved pilot metrics only; it does not train or score models."],
    }

    for symbol in settings.symbols:
        path = metrics_root / symbol / "classification_metrics.json"
        if not path.is_file():
            continue
        metrics = _read_json(path)
        repro = metrics.get("reproducibility", {})
        thresholds["symbols"][symbol] = {"data_sha256": metrics.get("final_test_data_sha256"), "horizons": {}}
        for horizon_text, item in metrics.get("horizons", {}).items():
            horizon = int(horizon_text)
            thresholds["symbols"][symbol]["horizons"][horizon_text] = {
                "direction_delta": item.get("delta"),
                "stress_thresholds_by_fold": item.get("stress_thresholds_by_fold", []),
                "final_train_stress_thresholds": item.get("final_train_stress_thresholds"),
            }
            for label, rate in (item.get("direction_class_distribution_train") or {}).items():
                direction_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "train", "class": label, "proportion": rate, "n": None})
            for label, rate in (item.get("stress_class_distribution_train") or {}).items():
                stress_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "train", "class": label, "proportion": rate, "n": None})
            for metric in item.get("walk_forward_direction", []):
                row = {"symbol": symbol, "horizon_seconds": horizon, "task": "direction", **metric}
                weekly_rows.append(row)
                _append_metric(metric_rows, symbol=symbol, horizon=horizon, split=f"validation:{metric.get('week_utc')}", task="direction", model=metric.get("model", "unknown"), metric=metric)
                for label, rate in _confusion_distribution(metric).items():
                    direction_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "validation", "class": label, "proportion": rate, "n": metric.get("n")})
            for metric in item.get("walk_forward_stress", []):
                row = {"symbol": symbol, "horizon_seconds": horizon, "task": "stress", **metric}
                weekly_rows.append(row)
                _append_metric(metric_rows, symbol=symbol, horizon=horizon, split=f"validation:{metric.get('week_utc')}", task="stress", model=metric.get("model", "unknown"), metric=metric)
                if metric.get("positive_rate") is not None:
                    stress_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "validation", "class": "1", "proportion": metric.get("positive_rate"), "n": metric.get("n")})
            for task, model_key, result_key in (("direction", "direction_model", "direction_test"), ("stress", "stress_model", "stress_test")):
                result = item.get(result_key) or {}
                _append_metric(metric_rows, symbol=symbol, horizon=horizon, split="holdout_once", task=task, model=item.get(model_key, "unknown"), metric=result)
                if task == "direction":
                    for label, rate in _confusion_distribution(result).items():
                        direction_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "holdout_once", "class": label, "proportion": rate, "n": result.get("n")})
                elif result.get("positive_rate") is not None:
                    stress_classes.append({"symbol": symbol, "horizon_seconds": horizon, "split": "holdout_once", "class": "1", "proportion": result.get("positive_rate"), "n": result.get("n")})
            for entry in item.get("direction_delta_sensitivity", []):
                delta_rows.append({"symbol": symbol, "horizon_seconds": horizon, **{key: value for key, value in entry.items() if not isinstance(value, (dict, list))}})
            for entry in item.get("direction_L_sensitivity", []):
                obi_rows.append({"symbol": symbol, "horizon_seconds": horizon, **{key: value for key, value in entry.items() if not isinstance(value, (dict, list))}})
            for task, key in (("direction", "direction_paired_tests_vs_persistence"), ("stress", "stress_paired_tests_vs_persistence")):
                paired_rows.append({"symbol": symbol, "horizon_seconds": horizon, "task": task, "result_json": json.dumps(item.get(key, {}), sort_keys=True)})
            for task, key in (("direction", "direction_test_by_regime"), ("stress", "stress_test_by_regime")):
                for regime, payload in (item.get(key) or {}).items():
                    regime_rows.append({"symbol": symbol, "horizon_seconds": horizon, "task": task, "regime": regime, "result_json": json.dumps(payload, sort_keys=True)})
            stress_sensitivity_rows.append({"symbol": symbol, "horizon_seconds": horizon, **{key: value for key, value in (item.get("stress_spread_only_sensitivity") or {}).items() if not isinstance(value, (dict, list))}})
            for task, key in (("direction", "direction_runtime_profile"), ("stress", "stress_runtime_profile")):
                latency_rows.append({"symbol": symbol, "horizon_seconds": horizon, "task": task, "profile_json": json.dumps(item.get(key, {}), sort_keys=True), "artifact_bytes": item.get("artifact_bytes")})

        timestamps = []
        for feature_file in sorted((settings.data_dir / "features" / f"symbol={symbol}").glob("date=*/features_1s.parquet")):
            timestamps.append(pd.read_parquet(feature_file, columns=["timestamp_utc"])["timestamp_utc"])
        ts = pd.to_datetime(pd.concat(timestamps, ignore_index=True), utc=True, errors="coerce") if timestamps else pd.Series([], dtype="datetime64[ns, UTC]")
        embargo = settings.lob_embargo_seconds
        val_start = pd.Timestamp(settings.lob_first_validation_week_utc)
        dev_end = pd.Timestamp(settings.lob_development_end_utc) + pd.Timedelta(1, unit="s")
        test_start = pd.Timestamp(settings.lob_untouched_test_start_utc)
        test_end = pd.Timestamp(settings.lob_untouched_test_end_utc) + pd.Timedelta(1, unit="s")
        train_start = pd.Timestamp("2026-09-28T00:00:00Z")
        train_boundary = val_start
        train_cutoff = train_boundary - pd.Timedelta(embargo, unit="s")
        development_train_cutoff = test_start - pd.Timedelta(embargo, unit="s")
        fold_rows.extend([
            {"symbol": symbol, "fold": "validation_2026-10-05", "train_start_utc": train_start, "train_end_utc_exclusive": train_boundary, "train_rows_before_embargo": int(((ts >= train_start) & (ts < train_boundary)).sum()), "train_rows_after_embargo": int(((ts >= train_start) & (ts < train_cutoff)).sum()), "validation_start_utc": val_start, "validation_end_utc_exclusive": dev_end, "validation_origin_rows": int(((ts >= val_start) & (ts < dev_end)).sum()), "test_start_utc": None, "test_end_utc_exclusive": None, "test_origin_rows": None, "purge_embargo_seconds": embargo, "status": "validation fold from frozen pilot"},
            {"symbol": symbol, "fold": "holdout_2026-10-12", "train_start_utc": train_start, "train_end_utc_exclusive": test_start, "train_rows_before_embargo": int(((ts >= train_start) & (ts < test_start)).sum()), "train_rows_after_embargo": int(((ts >= train_start) & (ts < development_train_cutoff)).sum()), "validation_start_utc": val_start, "validation_end_utc_exclusive": dev_end, "validation_origin_rows": int(((ts >= val_start) & (ts < dev_end)).sum()), "test_start_utc": test_start, "test_end_utc_exclusive": test_end, "test_origin_rows": int(((ts >= test_start) & (ts < test_end)).sum()), "purge_embargo_seconds": embargo, "status": "holdout already scored once; no re-score by this script"},
        ])
        leakage.setdefault("symbols", {})[symbol] = {
            "metrics_final_test_scored_once": metrics.get("final_test_scored_once"),
            "final_test_data_sha256": metrics.get("final_test_data_sha256"),
            "configured_split": {"validation_start_utc": settings.lob_first_validation_week_utc, "development_end_utc": repro.get("development_end_utc"), "test_start_utc": repro.get("final_test_start_utc"), "test_end_utc": repro.get("final_test_end_utc")},
        }

    out = output_root / "outputs"
    _write_csv(out / "metrics" / "model_comparison.csv", metric_rows)
    _write_csv(out / "metrics" / "direction_validation.csv", [row for row in metric_rows if row["task"] == "direction" and row["split"].startswith("validation:")])
    _write_csv(out / "metrics" / "stress_validation.csv", [row for row in metric_rows if row["task"] == "stress" and row["split"].startswith("validation:")])
    _write_csv(out / "robustness" / "metrics_by_week.csv", weekly_rows)
    _write_csv(out / "robustness" / "metrics_by_regime.csv", regime_rows)
    _write_csv(out / "robustness" / "delta_sensitivity.csv", delta_rows)
    _write_csv(out / "robustness" / "obi_depth_sensitivity.csv", obi_rows)
    _write_csv(out / "robustness" / "stress_label_sensitivity.csv", stress_sensitivity_rows)
    _write_csv(out / "robustness" / "paired_tests.csv", paired_rows)
    _write_csv(out / "robustness" / "latency_model_size.csv", latency_rows)
    _write_csv(out / "labels" / "direction_class_distribution.csv", direction_classes)
    _write_csv(out / "labels" / "stress_class_distribution.csv", stress_classes)
    split_class_rows = [{"symbol": row["symbol"], "horizon_seconds": row["horizon_seconds"], "split": row["split"], "target": "direction", "class": row["class"], "proportion": row["proportion"], "n": row["n"]} for row in direction_classes]
    split_class_rows.extend({"symbol": row["symbol"], "horizon_seconds": row["horizon_seconds"], "split": row["split"], "target": "stress", "class": row["class"], "proportion": row["proportion"], "n": row["n"]} for row in stress_classes)
    _write_csv(out / "splits" / "fold_class_distribution.csv", split_class_rows)
    _write_json(out / "labels" / "direction_thresholds.json", {
        "study_label": "historical_archive_pilot",
        "symbols": {symbol: {h: item["direction_delta"] for h, item in values["horizons"].items()} for symbol, values in thresholds["symbols"].items()},
        "source": "saved once-scored pilot metrics; no evaluation was rerun",
    })
    _write_json(out / "labels" / "stress_thresholds.json", thresholds)
    _write_csv(out / "splits" / "fold_manifest.csv", fold_rows)
    _write_json(out / "splits" / "leakage_audit.json", leakage)
    _write_json(out / "deeplob" / "status.json", {
        "status": "not_evaluated_in_submission_pilot",
        "reason": "No saved verified DeepLOB fit/predict report was found; holdout has already been scored once.",
        "do_not_run_against_holdout": True,
        "configured_train_cap": settings.lob_deeplob_max_train_rows,
    })


def build_artifact_inventory(output_root: Path) -> None:
    artifact_root = REPOSITORY_ROOT / "backend" / "artifacts" / "meter"
    data_manifest = output_root / "outputs" / "data_quality" / "data_manifest.csv"
    audit_manifest_sha = _sha256(data_manifest) if data_manifest.is_file() else None
    rows: list[dict[str, Any]] = []
    for path in sorted(artifact_root.glob("*/horizon_*.joblib")):
        bundle = joblib.load(path)
        provenance = bundle.get("reproducibility") or {}
        label_schemas = bundle.get("label_schema_versions") or {
            "direction": provenance.get("direction_label_schema_version"),
            "stress": provenance.get("stress_label_schema_version"),
        }
        target_names = bundle.get("target_names") or {"direction": "midprice_direction (inferred from bundle slot)", "stress": "liquidity_stress (inferred from bundle slot)"}
        metrics = bundle.get("metrics") or {}
        metric_summary = {
            task: {name: (metrics.get(task) or {}).get(name) for name in ("macro_f1", "balanced_accuracy", "mcc", "pr_auc", "brier", "positive_rate") if name in (metrics.get(task) or {})}
            for task in ("direction", "stress")
        }
        missing = [
            name for name, value in {
                "python_version": provenance.get("python_version"),
                "dependency_versions": provenance.get("dependency_versions"),
                "data_manifest_sha256_at_training": provenance.get("data_manifest_sha256"),
                "explicit_target_names": bundle.get("target_names"),
            }.items() if value is None
        ]
        rows.append({
            "symbol": bundle.get("symbol"),
            "horizon_seconds": bundle.get("horizon_seconds"),
            "model_version": bundle.get("model_version"),
            "direction_model_name": bundle.get("direction_model_name"),
            "stress_model_name": bundle.get("stress_model_name"),
            "target_names_json": json.dumps(target_names, sort_keys=True),
            "feature_schema_version": bundle.get("feature_schema_version") or provenance.get("feature_schema_version"),
            "direction_label_schema_version": label_schemas.get("direction"),
            "stress_label_schema_version": label_schemas.get("stress"),
            "configuration_file": provenance.get("configuration_file"),
            "configuration_sha256": provenance.get("configuration_sha256"),
            "training_manifest_sha256": provenance.get("data_manifest_sha256"),
            "manifest_sha256_at_audit": audit_manifest_sha,
            "observed_data_start_utc": provenance.get("observed_data_start_utc"),
            "observed_data_end_utc": provenance.get("observed_data_end_utc"),
            "training_period_json": json.dumps(provenance.get("training_period_utc", {}), sort_keys=True),
            "validation_period_json": json.dumps(provenance.get("validation_period_utc", {}), sort_keys=True),
            "trained_at_utc": bundle.get("trained_at_utc"),
            "stress_probability_threshold": bundle.get("stress_probability_threshold"),
            "training_thresholds_json": json.dumps(bundle.get("thresholds", {}), sort_keys=True),
            "metrics_summary_json": json.dumps(metric_summary, sort_keys=True),
            "random_seeds_json": json.dumps(provenance.get("random_seeds", {}), sort_keys=True),
            "python_version": provenance.get("python_version"),
            "dependency_versions_json": json.dumps(provenance.get("dependency_versions", {}), sort_keys=True),
            "artifact_bytes": path.stat().st_size,
            "artifact_sha256": _sha256(path),
            "metadata_complete_for_current_plan": not missing,
            "metadata_fields_missing": ";".join(missing),
            "artifact_path": str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        })
    _write_csv(output_root / "outputs" / "reproducibility" / "artifact_inventory.csv", rows)
    _write_json(output_root / "outputs" / "reproducibility" / "artifact_inventory.json", {
        "schema_version": "artifact_inventory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifacts": rows,
        "notes": [
            "Inventory reads existing joblib bundles and hashes files; it does not change or retrain artifacts.",
            "Missing dependency and training-manifest metadata are reported honestly for existing pilot artifacts.",
            "The model code now writes those fields for future training runs.",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.yaml")
    parser.add_argument("--qa-root", type=Path, default=REPOSITORY_ROOT / "outputs" / "manifests" / "historical_archive_pilot")
    parser.add_argument("--metrics-root", type=Path, default=REPOSITORY_ROOT / "outputs" / "tables" / "historical_archive_pilot")
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--quality-only", action="store_true", help="Refresh only data quality manifests; skip feature and saved-result tables")
    parser.add_argument("--artifact-only", action="store_true", help="Refresh only the saved artifact inventory")
    args = parser.parse_args()
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    qa_root = args.qa_root if args.qa_root.is_absolute() else REPOSITORY_ROOT / args.qa_root
    metrics_root = args.metrics_root if args.metrics_root.is_absolute() else REPOSITORY_ROOT / args.metrics_root
    output_root = args.output_root if args.output_root.is_absolute() else REPOSITORY_ROOT / args.output_root
    if args.quality_only and args.artifact_only:
        parser.error("--quality-only and --artifact-only cannot be combined")
    manifest = []
    if not args.artifact_only:
        manifest = build_data_quality(settings, qa_root, output_root)
    if not args.quality_only and not args.artifact_only:
        build_feature_audits(settings, metrics_root, output_root)
        build_saved_result_tables(metrics_root, output_root, settings)
    if not args.quality_only:
        build_artifact_inventory(output_root)
    if args.artifact_only:
        print("Artifact inventory refreshed from saved bundles.")
    elif args.quality_only:
        print(f"Data quality evidence refreshed from {len(manifest)} saved archive partitions.")
    else:
        print(f"Submission evidence built from {len(manifest)} saved archive partitions.")
    print(f"Outputs: {output_root / 'outputs'}")
    print("No model fitting or final-test scoring was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
