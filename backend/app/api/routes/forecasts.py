"""Routes for immutable research and current-data production forecasts."""

from fastapi import APIRouter, HTTPException

from app.api.schemas.forecast import ForecastRequest
from app.services.forecast_service import (
    create_forecast as load_forecast,
    create_production_forecast,
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


@router.get("/{forecast_id}")
def get_forecast(forecast_id: str):
    try:
        return load_forecast_by_id(forecast_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
