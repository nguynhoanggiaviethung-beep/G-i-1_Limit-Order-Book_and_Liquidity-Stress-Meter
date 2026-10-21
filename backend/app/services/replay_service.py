"""Read-only replay, model discovery, and Liquidity Stress Meter service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import REPOSITORY_ROOT
from app.data.grid_store import GridStore
from app.features.microstructure import compute_microstructure_features
from app.models.lob_classifiers import LOB_CLASSIFICATION_FEATURES


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _snapshot(row: dict[str, Any], depth: int) -> dict[str, Any]:
    derived = compute_microstructure_features(row)
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
        "book_valid": bool(row.get("book_valid", True)),
        "bids": bids,
        "asks": asks,
        "features": {**row, **derived},
        "quality": {
            "book_valid": bool(row.get("book_valid", True)),
            "sequence_gap": bool(row.get("sequence_gap", False)),
            "seconds_since_gap": row.get("seconds_since_gap"),
        },
    }


class ReplayService:
    def __init__(self, store: GridStore | None = None, artifact_root: str | Path | None = None) -> None:
        self.store = store or GridStore()
        self.feature_store = GridStore(self.store.root.parent / "features")
        self.artifact_root = Path(artifact_root) if artifact_root else REPOSITORY_ROOT / "backend" / "artifacts" / "meter"
        self._bundles: dict[tuple[str, int], dict[str, Any] | None] = {}

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
            "snapshots": [_snapshot(self.store.json_row(row), depth) for _, row in frame.iterrows()],
        }

    def at(self, symbol: str, timestamp: str, depth: int = 20) -> dict[str, Any]:
        requested = pd.Timestamp(timestamp)
        frame = self.store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
        if frame.empty:
            return {"schema": "replay_snapshot.v1", "symbol": symbol, "status": "unavailable", "reason": "No valid book at or before the requested time", "snapshot": None}
        row = frame.iloc[-1]
        return {"schema": "replay_snapshot.v1", "symbol": symbol, "status": "ready", "snapshot": _snapshot(self.store.json_row(row), depth)}

    def _bundle(self, symbol: str, horizon_seconds: int) -> dict[str, Any] | None:
        key = symbol.upper(), int(horizon_seconds)
        if key not in self._bundles:
            path = self.artifact_root / key[0] / f"horizon_{key[1]}.joblib"
            try:
                bundle = joblib.load(path) if path.is_file() else None
                if not isinstance(bundle, dict) or bundle.get("symbol") != key[0] or int(bundle.get("horizon_seconds", -1)) != key[1]:
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
        base = {
            "schema": "liquidity_meter.v1",
            "symbol": symbol.upper(),
            "timestamp_utc": timestamp,
            "horizon_seconds": int(horizon_seconds),
            "spread": str(feature_values.get("absolute_spread")) if feature_values.get("absolute_spread") is not None else None,
            "depth_imbalance": {f"L{depth}": _number(feature_values.get(f"obi_qty_{depth}")) for depth in (1, 5, 10, 20)},
            "ofi_5s": _number(feature_values.get("ofi_5s")),
            "quality": quality or {"book_valid": True},
            "p_up": None,
            "p_flat": None,
            "p_down": None,
            "p_stress": None,
            "model_version": None,
        }
        if bundle is None:
            return {**base, "status": "unavailable", "reason": "No trained artifact is available for this symbol and horizon"}
        columns = list(bundle.get("feature_columns") or LOB_CLASSIFICATION_FEATURES)
        direction_model, stress_model = bundle.get("direction_model"), bundle.get("stress_model")
        if direction_model is None or stress_model is None:
            return {**base, "status": "unavailable", "reason": "The model artifact is missing a direction or stress classifier"}
        values = {column: _number(feature_values.get(column)) for column in columns}
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
            requested = pd.Timestamp(timestamp)
            frame = self.feature_store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
            if frame.empty:
                frame = self.store.read(symbol, start=requested.normalize().isoformat(), end=timestamp, limit=10000)
        if frame.empty:
            return self.predict_features(symbol, timestamp, horizon_seconds, {}, {"book_valid": False, "reason": "No replay data is available"})
        row = self.store.json_row(frame.iloc[-1])
        derived = compute_microstructure_features(row)
        features = {**row, **derived}
        for name, value in row.items():
            if name not in features:
                features[name] = value
        return self.predict_features(
            symbol, row.get("timestamp_utc"), horizon_seconds, features,
            {"book_valid": bool(row.get("book_valid", True)), "seconds_since_gap": row.get("seconds_since_gap")},
        )

    def models(self, symbols: tuple[str, ...], horizons: tuple[int, ...]) -> dict[str, Any]:
        items = []
        for symbol in symbols:
            for horizon in horizons:
                path = self.artifact_root / symbol / f"horizon_{horizon}.joblib"
                if path.is_file():
                    bundle = self._bundle(symbol, horizon)
                    if bundle is not None:
                        items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "ready", "model_version": bundle.get("model_version"), "metrics": bundle.get("metrics"), "trained_at_utc": bundle.get("trained_at_utc")})
                    else:
                        items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "unavailable", "reason": "The model artifact is invalid or incompatible"})
                else:
                    items.append({"symbol": symbol, "horizon_seconds": horizon, "status": "unavailable", "reason": "Model artifact is not present"})
        return {"schema": "models.v1", "status": "ready" if any(item["status"] == "ready" for item in items) else "unavailable", "models": items}

    def evaluation(self, symbol: str) -> dict[str, Any]:
        path = REPOSITORY_ROOT / "outputs" / "tables" / symbol / "classification_metrics.json"
        if not path.is_file():
            return {"schema": "evaluation.v1", "symbol": symbol, "status": "unavailable", "reason": "No out-of-sample classification results are available"}
        return {"schema": "evaluation.v1", "symbol": symbol, "status": "ready", "results": json.loads(path.read_text(encoding="utf-8"))}

