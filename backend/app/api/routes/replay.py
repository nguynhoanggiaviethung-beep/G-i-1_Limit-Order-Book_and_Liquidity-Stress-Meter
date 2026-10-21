"""Read-only historical replay and Liquidity Stress Meter API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.config import load_settings


router = APIRouter(prefix="/api/v1", tags=["LOB replay and liquidity meter"])


def _service(request: Request):
    service = getattr(request.app.state, "replay_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Replay service is unavailable")
    return service


def _symbol(symbol: str) -> str:
    value = symbol.upper()
    if value not in load_settings().symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {value} is not configured")
    return value


@router.get("/data-status")
def get_data_status(request: Request) -> dict[str, Any]:
    return _service(request).data_status(load_settings().symbols)


@router.get("/replay/{symbol}/at")
def get_replay_at(
    request: Request,
    symbol: str,
    timestamp_utc: str = Query(...),
    depth: int = Query(20, ge=1, le=20),
) -> dict[str, Any]:
    return _service(request).at(_symbol(symbol), timestamp_utc, depth)


@router.get("/replay/{symbol}")
def get_replay(
    request: Request,
    symbol: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    depth: int = Query(20, ge=1, le=20),
    limit: int = Query(1000, ge=1, le=5000),
) -> dict[str, Any]:
    return _service(request).replay(_symbol(symbol), start=start, end=end, depth=depth, limit=limit)


@router.get("/meter/{symbol}")
def get_meter(
    request: Request,
    symbol: str,
    timestamp_utc: str | None = Query(None),
    horizon_seconds: int = Query(5, ge=1, le=60),
) -> dict[str, Any]:
    return _service(request).meter(_symbol(symbol), timestamp_utc, horizon_seconds)


class PredictRequest(BaseModel):
    symbol: str
    horizon_seconds: int = Field(5, ge=1, le=60)
    timestamp_utc: str | None = None
    features: dict[str, Any]
    quality: dict[str, Any] = Field(default_factory=dict)


@router.post("/predict")
def predict(request: Request, payload: PredictRequest) -> dict[str, Any]:
    symbol = _symbol(payload.symbol)
    return _service(request).predict_features(
        symbol,
        payload.timestamp_utc,
        payload.horizon_seconds,
        payload.features,
        payload.quality or {"book_valid": True},
    )


@router.get("/models")
def get_models(request: Request) -> dict[str, Any]:
    settings = load_settings()
    return _service(request).models(settings.symbols, settings.lob_classification_horizons_seconds)


@router.get("/evaluation/{symbol}")
def get_evaluation(request: Request, symbol: str) -> dict[str, Any]:
    return _service(request).evaluation(_symbol(symbol))

