from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from starlette.requests import Request

from app.api.routes.health import health
from app.api.routes.replay import (
    LiquidityMeterResponse,
    PredictRequest,
    get_data_status,
    get_evaluation,
    get_meter,
    get_models,
    get_replay,
    get_replay_at,
    predict,
)
from app.data.grid_store import GridStore
from app.main import app
from app.services.replay_service import ReplayService


class ReplayStub:
    def __init__(self):
        self.store = SimpleNamespace(catalog=lambda symbols: {"status": "ready", "symbols": {symbol: {} for symbol in symbols}})

    def data_status(self, symbols):
        return {"schema": "data_status.v1", "status": "ready", "symbols": list(symbols)}

    def replay(self, symbol, **kwargs):
        return {"schema": "replay_snapshot.v1", "symbol": symbol, "status": "ready", "snapshots": []}

    def at(self, symbol, timestamp, depth):
        return {"schema": "replay_snapshot.v1", "symbol": symbol, "status": "ready", "snapshot": {"timestamp_utc": timestamp}}

    def meter(self, symbol, timestamp, horizon_seconds):
        return {"schema": "liquidity_meter.v1", "symbol": symbol, "status": "ok", "horizon_seconds": horizon_seconds}

    def predict_features(self, symbol, timestamp, horizon_seconds, features, quality):
        return {"schema": "liquidity_meter.v1", "symbol": symbol, "status": "ok", "horizon_seconds": horizon_seconds}

    def models(self, symbols, horizons):
        return {"schema": "models.v1", "status": "ready", "models": []}

    def evaluation(self, symbol):
        return {"schema": "evaluation.v1", "symbol": symbol, "status": "ready", "results": {}}


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b"", "app": app})


@pytest.fixture
def service(monkeypatch):
    replay_service = ReplayStub()
    monkeypatch.setattr(app.state, "replay_service", replay_service, raising=False)
    return replay_service


def test_active_api_contract_replaces_collector_routes(service):
    openapi = app.openapi()
    paths = openapi["paths"]
    assert app.title == "Limit Order Book & Liquidity Stress Meter API"
    for path in (
        "/health",
        "/api/v1/data-status",
        "/api/v1/replay/{symbol}",
        "/api/v1/replay/{symbol}/at",
        "/api/v1/meter/{symbol}",
        "/api/v1/predict",
        "/api/v1/models",
        "/api/v1/evaluation/{symbol}",
    ):
        assert path in paths
    assert "/api/v1/collector/status" not in paths
    assert "/api/v1/lob/latest" not in paths
    timestamp_parameter = next(
        parameter for parameter in paths["/api/v1/replay/{symbol}/at"]["get"]["parameters"]
        if parameter["name"] == "timestamp_utc"
    )
    assert timestamp_parameter["schema"]["format"] == "date-time"
    meter_ref = paths["/api/v1/meter/{symbol}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    meter_name = meter_ref.rsplit("/", 1)[-1]
    meter_properties = openapi["components"]["schemas"][meter_name]["properties"]
    assert {"p_up", "p_flat", "p_down", "p_stress", "stress_probability_threshold", "model_version", "quality"}.issubset(meter_properties)


def test_health_reports_replay_availability(service):
    assert health(_request()) == {"status": "ok", "replay_status": "ready"}


def test_data_status_and_replay_routes(service):
    request = _request()
    status = get_data_status(request)
    replay = get_replay(request, "BTCUSDT", start=None, end=None, depth=20, limit=1)
    at = get_replay_at(request, "BTCUSDT", datetime(2025, 12, 1, tzinfo=timezone.utc), depth=20)
    assert status["schema"] == "data_status.v1"
    assert replay["schema"] == "replay_snapshot.v1"
    assert at["snapshot"]["timestamp_utc"].startswith("2025-12-01T00:00:00")


def test_meter_predict_models_and_evaluation_routes(service):
    request = _request()
    meter = get_meter(request, "BTCUSDT", timestamp_utc=None, horizon_seconds=5)
    prediction = predict(request, PredictRequest(symbol="BTCUSDT", horizon_seconds=5, features={"obi_qty_5": 0.2}))
    models = get_models(request)
    evaluation = get_evaluation(request, "BTCUSDT")
    assert meter["schema"] == prediction["schema"] == "liquidity_meter.v1"
    assert models["schema"] == "models.v1"
    assert evaluation["schema"] == "evaluation.v1"


def test_predict_request_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        PredictRequest.model_validate({"symbol": "BTCUSDT", "features": {}, "timestamp_utc": "not-a-date"})


def test_meter_response_model_preserves_artifact_threshold_and_schema_alias():
    response = LiquidityMeterResponse.model_validate({
        "schema": "liquidity_meter.v1",
        "symbol": "BTCUSDT",
        "timestamp_utc": None,
        "horizon_seconds": 5,
        "status": "unavailable",
        "quality": {"book_valid": False},
        "stress_probability_threshold": 0.37,
    })
    serialized = response.model_dump(by_alias=True)
    assert serialized["schema"] == "liquidity_meter.v1"
    assert serialized["stress_probability_threshold"] == 0.37


def test_prediction_refuses_excessive_missing_features(monkeypatch):
    replay_service = ReplayService(max_missing_feature_fraction=0.1)
    bundle = {"feature_columns": ["feature_a", "feature_b"], "direction_model": object(), "stress_model": object(), "stress_probability_threshold": 0.37}
    monkeypatch.setattr(replay_service, "_bundle", lambda *_: bundle)
    response = replay_service.predict_features("BTCUSDT", None, 5, {})
    assert response["status"] == "unavailable"
    assert response["stress_probability_threshold"] == 0.37
    assert response["quality"]["missing_features"] == ["feature_a", "feature_b"]
    assert response["quality"]["missing_feature_count"] == 2


def test_at_and_meter_refuse_stale_valid_rows(monkeypatch):
    requested = "2025-12-01T00:00:10Z"
    row_time = "2025-12-01T00:00:07Z"
    rows = pd.DataFrame({"timestamp_utc": [pd.Timestamp(row_time)], "book_valid": [True]})

    class StoreStub:
        root = Path("test-grid")
        json_row = staticmethod(GridStore.json_row)

        def read(self, symbol, **kwargs):
            return rows.copy()

    replay_service = ReplayService(store=StoreStub(), max_staleness_seconds=2)
    monkeypatch.setattr(replay_service.feature_store, "read", lambda symbol, **kwargs: rows.copy())
    monkeypatch.setattr(replay_service, "_bundle", lambda *_: {"stress_probability_threshold": 0.37})
    replay = replay_service.at("BTCUSDT", requested)
    meter = replay_service.meter("BTCUSDT", requested, 5)
    assert replay["status"] == "unavailable"
    assert replay["snapshot_age_seconds"] == 3
    assert meter["status"] == "unavailable"
    assert meter["quality"]["snapshot_age_seconds"] == 3
    assert meter["stress_probability_threshold"] == 0.37


def test_meter_reports_missing_flow_features_from_grid(monkeypatch):
    row = {
        "timestamp_utc": pd.Timestamp("2025-12-01T00:00:00Z"),
        "symbol": "BTCUSDT",
        "book_valid": True,
        "bid_price_1": 100.0,
        "bid_qty_1": 1.0,
        "ask_price_1": 101.0,
        "ask_qty_1": 1.0,
    }
    rows = pd.DataFrame([row])

    class StoreStub:
        root = Path("test-grid")
        json_row = staticmethod(GridStore.json_row)

        def read(self, symbol, **kwargs):
            return rows.copy()

    replay_service = ReplayService(store=StoreStub(), max_missing_feature_fraction=0.1)
    monkeypatch.setattr(replay_service.feature_store, "read", lambda symbol, **kwargs: rows.copy())
    monkeypatch.setattr(replay_service, "_bundle", lambda *_: {"feature_columns": ["ofi_1s", "trade_imbalance_5s"]})
    response = replay_service.meter("BTCUSDT", None, 5)
    assert response["status"] == "unavailable"
    assert response["quality"]["missing_features"] == ["ofi_1s", "trade_imbalance_5s"]


def test_grid_store_caches_partition_reads_and_returns_copies(monkeypatch, tmp_path):
    path = tmp_path / "symbol=BTCUSDT" / "date=2025-12-01" / "book_1s.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partition")
    calls = []

    def read_parquet(_path):
        calls.append(_path)
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    store = GridStore(tmp_path)
    first = store._read_partition(path)
    first.loc[0, "value"] = 99
    second = store._read_partition(path)
    assert len(calls) == 1
    assert second.loc[0, "value"] == 1
