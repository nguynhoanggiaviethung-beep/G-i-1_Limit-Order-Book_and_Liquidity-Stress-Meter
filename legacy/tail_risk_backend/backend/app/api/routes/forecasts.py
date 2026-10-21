"""Routes for immutable research and current-data production forecasts."""

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas.forecast import ForecastRequest
from app.services.forecast_service import (
    create_forecast as load_forecast,
    create_production_forecast,
    get_forecast_history as load_forecast_history,
    get_forecast as load_forecast_by_id,
)

router = APIRouter(prefix="/api/v1/forecasts", tags=["forecasts"])


@router.post("")
def create_forecast(request: ForecastRequest):
    try:
        if request.mode == "production":
            return create_production_forecast(
                request.symbol, request.model, request.horizon_days, request.confidence_levels
            )
        return load_forecast(request.symbol, request.model, request.horizon_days, request.confidence_levels)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/history")
def get_forecast_history(
    symbol: str = Query(...),
    model: str = Query(...),
    horizon_days: int = Query(..., ge=1, le=30),
    mode: str = Query(default="research"),
    risk_method: str = Query(default="filtered_historical_simulation"),
    confidence: float = Query(default=0.95),
    limit: int = Query(default=365, ge=1, le=2000),
):
    try:
        if horizon_days not in (1, 7, 30):
            raise ValueError("horizon_days must be 1, 7, or 30")
        if confidence not in (0.95, 0.975, 0.99):
            raise ValueError("confidence must be one of 0.95, 0.975, or 0.99")
        return load_forecast_history(
            symbol, model, horizon_days, mode, risk_method, confidence, limit
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{forecast_id}")
def get_forecast(forecast_id: str):
    try:
        return load_forecast_by_id(forecast_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
