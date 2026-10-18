import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes.forecasts import create_forecast
from app.api.routes.health import health
from app.api.schemas.forecast import ForecastRequest
from app.main import app
from app.api.routes import forecasts


def test_health_handler_returns_ok_and_is_registered():
    assert app.title == "Crypto Volatility and Tail Risk API"
    assert health() == {"status": "ok"}
    assert "/health" in app.openapi()["paths"]


def test_forecast_route_is_registered():
    assert "post" in app.openapi()["paths"]["/api/v1/forecasts"]


def test_research_forecast_route(monkeypatch):
    expected = {"forecast_id": "btc-h1-random_walk-20261001", "mode": "research"}
    monkeypatch.setattr(forecasts, "load_forecast", lambda *args: expected)
    response = create_forecast(ForecastRequest(symbol="BTCUSDT"))
    assert response == expected


def test_production_forecast_route(monkeypatch):
    expected = {"forecast_id": "production-btc-h1-har", "mode": "production"}
    monkeypatch.setattr(forecasts, "create_production_forecast", lambda *args: expected)
    response = create_forecast(
        ForecastRequest(symbol="BTCUSDT", model="har", mode="production")
    )
    assert response == expected


def test_forecast_request_rejects_invalid_inputs():
    with pytest.raises(ValidationError):
        ForecastRequest(symbol="BTCUSDT", horizon_days=2)
    with pytest.raises(ValidationError):
        ForecastRequest(symbol="BTCUSDT", confidence_levels=[0.9])


def test_forecast_route_maps_missing_artifact_to_404(monkeypatch):
    def missing(*_args):
        raise FileNotFoundError("forecast artifact missing")

    monkeypatch.setattr(forecasts, "load_forecast", missing)
    with pytest.raises(HTTPException) as error:
        create_forecast(ForecastRequest(symbol="BTCUSDT"))
    assert error.value.status_code == 404
    assert error.value.detail == "forecast artifact missing"


def test_forecast_route_maps_runtime_failure_to_503(monkeypatch):
    def unavailable(*_args):
        raise RuntimeError("forecast service unavailable")

    monkeypatch.setattr(forecasts, "load_forecast", unavailable)
    with pytest.raises(HTTPException) as error:
        create_forecast(ForecastRequest(symbol="BTCUSDT"))
    assert error.value.status_code == 503
