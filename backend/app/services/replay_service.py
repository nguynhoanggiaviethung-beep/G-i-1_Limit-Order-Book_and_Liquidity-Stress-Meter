"""Read-only replay, model discovery, and Liquidity Stress Meter service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.grid_store import GridStore
from app.features.microstructure import compute_microstructure_features
from app.labels.direction import DIRECTION_LABEL_SCHEMA_VERSION
from app.labels.stress import STRESS_LABEL_SCHEMA_VERSION
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES, LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _training_impact_notional(bundle: dict[str, Any], symbol: str) -> float | None:
    """Impact notional the artifact was trained with, read from its own training configuration.

    Returns None when it cannot be verified: no configuration recorded, the file is missing, or
    its SHA-256 no longer matches the one stored at training time.
    """
    if bundle.get("impact_notional_usdt") is not None:
        return _number(bundle["impact_notional_usdt"])
    reproducibility = bundle.get("reproducibility") or {}
    configuration_file = reproducibility.get("configuration_file")
    if not configuration_file:
        return None
    path = Path(configuration_file)
    path = path if path.is_absolute() else REPOSITORY_ROOT / path
    if not path.is_file():
        return None
    expected = reproducibility.get("configuration_sha256")
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        return None
    return load_settings(path).impact_notional_for(symbol)


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _snapshot(row: dict[str, Any], depth: int, impact_notional: float = 100000.0, tick_size: float | None = None) -> dict[str, Any]:
    derived = compute_microstructure_features(row, impact_notional, tick_size)
    bids, asks = [], []
    for level in range(1, depth + 1):
        bp, bq = row.get(f"bid_price_{level}"), row.get(f"bid_qty_{level}")
        ap, aq = row.get(f"ask_price_{level}"), row.get(f"ask_qty_{level}")
        if bp is not None and bq is not None:
            bids.append([str(bp), str(bq)])
        if ap is not None and aq is not None:
            asks.append([str(ap), str(aq)])
    return {
        "schema": "replay_snapshot.v1",
        "symbol": row.get("symbol"),
        "timestamp_utc": row.get("timestamp_utc"),
        "book_valid": _truthy(row.get("book_valid"), True),
        "bids": bids,
        "asks": asks,
        "features": {**row, **derived},
        "quality": {
            "book_valid": _truthy(row.get("book_valid"), True),
            "sequence_gap": _truthy(row.get("sequence_gap"), False),
            "seconds_since_gap": row.get("seconds_since_gap"),
        },
    }


class ReplayService:
    def __init__(
        self,
        store: GridStore | None = None,
        artifact_root: str | Path | None = None,
        max_staleness_seconds: int | None = None,
        max_missing_feature_fraction: float | None = None,
        impact_notional_usdt: float | None = None,
    ) -> None:
        self.store = store or GridStore()
        self.feature_store = GridStore(self.store.root.parent / "features")
        settings = load_settings()
        self.artifact_root = Path(artifact_root) if artifact_root else settings.lob_artifact_dir
        self.evaluation_root = settings.lob_evaluation_dir
        self.max_staleness_seconds = settings.lob_max_staleness_seconds if max_staleness_seconds is None else max_staleness_seconds
        self.max_missing_feature_fraction = settings.lob_max_missing_feature_fraction if max_missing_feature_fraction is None else max_missing_feature_fraction
        self.impact_notionals = settings.lob_stress_impact_notional_by_symbol or {}
        self.tick_sizes = settings.lob_tick_size_by_symbol or {}
        self.default_impact_notional = settings.lob_stress_impact_notional_usdt if impact_notional_usdt is None else impact_notional_usdt
        self._bundles: dict[tuple[str, int], dict[str, Any] | None] = {}
        self._bundle_reasons: dict[tuple[str, int], str] = {}

    def data_status(self, symbols: tuple[str, ...]) -> dict[str, Any]:
        return self.store.catalog(symbols)

    def replay(self, symbol: str, start=None, end=None, depth: int = 20, limit: int = 1000) -> dict[str, Any]:
        frame = self.store.read(symbol, start=start, end=end, limit=limit)
        return {
            "schema": "replay_snapshot.v1",
            "symbol": symbol,
            "status": "ready" if not frame.empty else "unavailable",
            "reason": None if not frame.empty else f"No valid replay rows found for {symbol}",
            "count": int(len(frame)),
            "snapshots": [
                _snapshot(
                    self.store.json_row(row), depth,
                    float(self.impact_notionals.get(symbol.upper(), self.default_impact_notional)),
                    self.tick_sizes.get(symbol.upper()),
                )
                for _, row in frame.iterrows()
            ],
        }

    def at(self, symbol: str, timestamp: str, depth: int = 20) -> dict[str, Any]:
        requested = _utc_timestamp(timestamp)
        frame = self.store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
        if frame.empty:
            return {
                "schema": "replay_snapshot.v1", "symbol": symbol, "status": "unavailable",
                "reason": f"No replay book for {symbol.upper()} at the requested time; capture and replay this symbol, then choose a time inside its capture window",
                "snapshot_age_seconds": None, "snapshot": None,
            }
        row = frame.iloc[-1]
        age_seconds = max(0.0, (requested - pd.Timestamp(row["timestamp_utc"])).total_seconds())
        if age_seconds > self.max_staleness_seconds:
            return {
                "schema": "replay_snapshot.v1", "symbol": symbol, "status": "unavailable",
                "reason": f"Latest valid book is {age_seconds:.3f}s old (limit {self.max_staleness_seconds}s)",
                "snapshot_age_seconds": age_seconds, "snapshot": None,
            }
        return {
            "schema": "replay_snapshot.v1", "symbol": symbol, "status": "ready",
            "snapshot_age_seconds": age_seconds,
            "snapshot": _snapshot(
                self.store.json_row(row), depth,
                float(self.impact_notionals.get(symbol.upper(), self.default_impact_notional)),
                self.tick_sizes.get(symbol.upper()),
            ),
        }

    def _bundle(self, symbol: str, horizon_seconds: int) -> dict[str, Any] | None:
        key = symbol.upper(), int(horizon_seconds)
        if key not in self._bundles:
            path = self.artifact_root / key[0] / f"horizon_{key[1]}.joblib"
            try:
                bundle = joblib.load(path) if path.is_file() else None
                reproducibility = (bundle.get("reproducibility") or {}) if isinstance(bundle, dict) else {}
                feature_schema = bundle.get("feature_schema_version") or reproducibility.get("feature_schema_version")
                label_schemas = bundle.get("label_schema_versions") or {
                    "direction": reproducibility.get("direction_label_schema_version"),
                    "stress": reproducibility.get("stress_label_schema_version"),
                }
                target_names = bundle.get("target_names")
                if (
                    not isinstance(bundle, dict)
                    or bundle.get("symbol") != key[0]
                    or int(bundle.get("horizon_seconds", -1)) != key[1]
                    or bundle.get("feature_columns") != list(LOB_CLASSIFICATION_FEATURES)
                    or feature_schema != LOB_CLASSIFICATION_FEATURE_SCHEMA_VERSION
                    or label_schemas.get("direction") != DIRECTION_LABEL_SCHEMA_VERSION
                    or label_schemas.get("stress") != STRESS_LABEL_SCHEMA_VERSION
                    or (target_names is not None and target_names != {"direction": "midprice_direction", "stress": "liquidity_stress"})
                    or bundle.get("direction_model") is None
                    or bundle.get("stress_model") is None
                ):
                    bundle = None
                if bundle is not None:
                    # price_impact_bps is a model feature, so runtime must sweep the same notional as training.
                    trained = _training_impact_notional(bundle, key[0])
                    runtime = float(self.impact_notionals.get(key[0], self.default_impact_notional))
                    if trained is not None and not np.isclose(trained, runtime):
                        self._bundle_reasons[key] = (
                            f"The model was trained with a {trained:,.0f} USDT impact notional, "
                            f"but the runtime configuration uses {runtime:,.0f} USDT"
                        )
                        bundle = None
                self._bundles[key] = bundle
            except Exception:
                self._bundles[key] = None
        return self._bundles[key]

    @staticmethod
    def _direction_probabilities(model: Any, feature_values: dict[str, float | None], columns: list[str]) -> dict[str, float] | None:
        if not hasattr(model, "predict_proba"):
            return None
        matrix = pd.DataFrame([[feature_values.get(column, np.nan) for column in columns]], columns=columns)
        values = np.asarray(model.predict_proba(matrix))[0]
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            final = list(model.named_steps.values())[-1]
            classes = list(getattr(final, "classes_", []))
        output = {"up": 0.0, "flat": 0.0, "down": 0.0}
        aliases = {"up": "up", "+1": "up", "1": "up", "flat": "flat", "0": "flat", "down": "down", "-1": "down"}
        for label, value in zip(classes, values):
            target = aliases.get(str(label).lower())
            if target:
                output[target] = float(value)
        return output if classes else None

    def predict_features(self, symbol: str, timestamp: str | None, horizon_seconds: int, feature_values: dict[str, Any], quality: dict[str, Any] | None = None) -> dict[str, Any]:
        bundle = self._bundle(symbol, horizon_seconds)
        stress_probability_threshold = _number(bundle.get("stress_probability_threshold")) if bundle is not None else None
        columns = list((bundle or {}).get("feature_columns") or LOB_CLASSIFICATION_FEATURES)
        values = {column: _number(feature_values.get(column)) for column in columns}
        missing_features = [column for column, value in values.items() if value is None]
        missing_fraction = len(missing_features) / len(columns) if columns else 1.0
        supplied_quality = quality or {}
        book_valid_value = supplied_quality.get("book_valid")
        if book_valid_value is None:
            book_valid_value = feature_values.get("book_valid")
        book_valid = _truthy(book_valid_value, True)
        quality_details = {
            **(supplied_quality or {"book_valid": book_valid}),
            "book_valid": book_valid,
            "missing_features": missing_features,
            "missing_feature_count": len(missing_features),
            "missing_feature_fraction": missing_fraction,
        }
        base = {
            "schema": "liquidity_meter.v1",
            "symbol": symbol.upper(),
            "timestamp_utc": timestamp,
            "horizon_seconds": int(horizon_seconds),
            "spread": str(feature_values.get("absolute_spread")) if feature_values.get("absolute_spread") is not None else None,
            "depth_imbalance": {f"L{depth}": _number(feature_values.get(f"obi_qty_{depth}")) for depth in (1, 5, 10, 20)},
            "ofi_5s": _number(feature_values.get("ofi_5s")),
            "quality": quality_details,
            "p_up": None,
            "p_flat": None,
            "p_down": None,
            "p_stress": None,
            "stress_probability_threshold": stress_probability_threshold,
            "model_version": bundle.get("model_version") if bundle is not None else None,
        }
        if bundle is None:
            reason = self._bundle_reasons.get((symbol.upper(), int(horizon_seconds)), "No trained artifact is available for this symbol and horizon")
            return {**base, "status": "unavailable", "reason": reason}
        if not book_valid:
            return {**base, "status": "unavailable", "reason": "The supplied order book is invalid"}
        if missing_fraction > self.max_missing_feature_fraction:
            return {
                **base,
                "status": "unavailable",
                "reason": f"{len(missing_features)} of {len(columns)} model features are missing; maximum allowed fraction is {self.max_missing_feature_fraction:.0%}",
            }
        direction_model, stress_model = bundle.get("direction_model"), bundle.get("stress_model")
        if direction_model is None or stress_model is None:
            return {**base, "status": "unavailable", "reason": "The model artifact is missing a direction or stress classifier"}
        direction = self._direction_probabilities(direction_model, values, columns)
        stress_matrix = pd.DataFrame([[values.get(column, np.nan) for column in columns]], columns=columns)
        if not hasattr(stress_model, "predict_proba"):
            return {**base, "status": "unavailable", "reason": "The stress classifier does not provide calibrated probabilities"}
        stress_probabilities = np.asarray(stress_model.predict_proba(stress_matrix))[0]
        stress_classes = list(getattr(stress_model, "classes_", []))
        if not stress_classes and hasattr(stress_model, "named_steps"):
            final = list(stress_model.named_steps.values())[-1]
            stress_classes = list(getattr(final, "classes_", []))
        positive_index = next((i for i, label in enumerate(stress_classes) if str(label).lower() in {"1", "true", "stress"}), None)
        if direction is None or positive_index is None:
            return {**base, "status": "unavailable", "reason": "The trained models do not expose the required class probabilities"}
        class_order = list(bundle.get("direction_class_order") or ["down", "flat", "up"])
        raw_direction = [direction.get(label, 0.0) for label in class_order]
        direction_calibrators = bundle.get("direction_calibrators") or []
        if len(direction_calibrators) == len(raw_direction):
            for index, calibrator in enumerate(direction_calibrators):
                if calibrator is not None:
                    raw_direction[index] = float(calibrator.predict([raw_direction[index]])[0])
        direction_total = sum(raw_direction)
        if direction_total > 0:
            raw_direction = [value / direction_total for value in raw_direction]
        direction = {label: float(value) for label, value in zip(class_order, raw_direction)}
        p_stress = float(stress_probabilities[positive_index])
        stress_calibrator = bundle.get("stress_calibrator")
        if stress_calibrator is not None:
            p_stress = float(stress_calibrator.predict([p_stress])[0])
        return {
            **base,
            "status": "ok",
            "p_up": direction["up"],
            "p_flat": direction["flat"],
            "p_down": direction["down"],
            "p_stress": min(1.0, max(0.0, p_stress)),
            "model_version": bundle.get("model_version"),
            "thresholds": bundle.get("thresholds"),
            "probability_calibrated": any(calibrator is not None for calibrator in direction_calibrators) or stress_calibrator is not None,
            "reason": None,
        }

    def meter(self, symbol: str, timestamp: str | None, horizon_seconds: int) -> dict[str, Any]:
        if timestamp is None:
            frame = self.feature_store.read(symbol, limit=1)
            if frame.empty:
                frame = self.store.read(symbol, limit=1)
        else:
            requested = _utc_timestamp(timestamp)
            frame = self.feature_store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
            if frame.empty:
                frame = self.store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
        if frame.empty:
            bundle = self._bundle(symbol, horizon_seconds)
            return {
                "schema": "liquidity_meter.v1", "symbol": symbol.upper(), "timestamp_utc": timestamp,
                "horizon_seconds": int(horizon_seconds), "status": "unavailable",
                "reason": f"No replay data is available for {symbol.upper()} at the requested time; capture and replay {symbol.upper()} data first",
                "quality": {"book_valid": False, "reason": "No replay data is available"},
                "p_up": None, "p_flat": None, "p_down": None, "p_stress": None,
                "stress_probability_threshold": _number(bundle.get("stress_probability_threshold")) if bundle is not None else None,
                "model_version": bundle.get("model_version") if bundle is not None else None,
            }
        row = self.store.json_row(frame.iloc[-1])
        snapshot_timestamp = pd.Timestamp(row["timestamp_utc"])
        snapshot_age = max(0.0, (_utc_timestamp(timestamp) - snapshot_timestamp).total_seconds()) if timestamp is not None else 0.0
        if timestamp is not None and snapshot_age > self.max_staleness_seconds:
            bundle = self._bundle(symbol, horizon_seconds)
            return {
                "schema": "liquidity_meter.v1", "symbol": symbol.upper(), "timestamp_utc": timestamp,
                "horizon_seconds": int(horizon_seconds), "status": "unavailable",
                "reason": f"Latest valid book is {snapshot_age:.3f}s old (limit {self.max_staleness_seconds}s)",
                "quality": {"book_valid": False, "snapshot_age_seconds": snapshot_age},
                "p_up": None, "p_flat": None, "p_down": None, "p_stress": None,
                "stress_probability_threshold": _number(bundle.get("stress_probability_threshold")) if bundle is not None else None,
                "model_version": bundle.get("model_version") if bundle is not None else None,
            }
        impact_notional = float(self.impact_notionals.get(symbol.upper(), self.default_impact_notional))
        derived = compute_microstructure_features(row, impact_notional, self.tick_sizes.get(symbol.upper()))
        # Impact always comes from the runtime notional, which the artifact check matched to training;
        # a feature partition built under an older notional must not leak into the model input.
        impact = {key: derived.get(key) for key in ("impact_buy_bps", "impact_sell_bps", "price_impact_bps")}
        features = {**derived, **row, **impact}
        return self.predict_features(
            symbol, row.get("timestamp_utc"), horizon_seconds, features,
            {"book_valid": _truthy(row.get("book_valid"), True), "seconds_since_gap": row.get("seconds_since_gap"), "snapshot_age_seconds": snapshot_age},
        )

    def models(self, symbols: tuple[str, ...], horizons: tuple[int, ...]) -> dict[str, Any]:
        items = []
        for symbol in symbols:
            for horizon in horizons:
                path = self.artifact_root / symbol / f"horizon_{horizon}.joblib"
                if path.is_file():
                    bundle = self._bundle(symbol, horizon)
                    if bundle is not None:
                        items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "ready", "model_version": bundle.get("model_version"), "study_label": bundle.get("study_label"), "metrics": bundle.get("metrics"), "trained_at_utc": bundle.get("trained_at_utc")})
                    else:
                        reason = self._bundle_reasons.get((symbol.upper(), int(horizon)), "The model artifact is invalid or incompatible")
                        items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "unavailable", "reason": reason})
                else:
                    items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "unavailable", "reason": "Model artifact is not present"})
        return {"schema": "models.v1", "status": "ready" if any(item["status"] == "ready" for item in items) else "unavailable", "models": items}

    def evaluation(self, symbol: str) -> dict[str, Any]:
        path = self.evaluation_root / symbol / "classification_metrics.json"
        if not path.is_file():
            return {"schema": "evaluation.v1", "symbol": symbol, "status": "unavailable", "reason": "No out-of-sample classification results are available"}
        return {"schema": "evaluation.v1", "symbol": symbol, "status": "ready", "results": json.loads(path.read_text(encoding="utf-8"))}

