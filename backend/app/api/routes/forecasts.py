"""Routes for the latest forecasts saved by the model training script."""

from fastapi import APIRouter, HTTPException

from app.api.schemas.forecast import ForecastRequest
from app.services.forecast_service import create_forecast as load_forecast, get_forecast as load_forecast_by_id

router = APIRouter(prefix="/api/v1/forecasts", tags=["forecasts"])


@router.post("")
def create_forecast(request: ForecastRequest):
    try:
        return load_forecast(request.symbol, request.model, request.horizon_minutes, request.confidence_levels)
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
