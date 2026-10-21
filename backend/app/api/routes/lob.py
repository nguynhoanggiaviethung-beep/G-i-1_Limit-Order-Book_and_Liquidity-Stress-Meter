"""Read-only API for live local books and LOB-native features."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import load_settings


router = APIRouter(prefix="/api/v1", tags=["LOB data"])


def _collector(request: Request):
    collector = getattr(request.app.state, "lob_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="LOB collector is not running")
    return collector


def _symbol(symbol: str) -> str:
    value = symbol.upper()
    if value not in load_settings().symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {value} is not configured")
    return value


@router.get("/collector/status")
def get_collector_status(request: Request) -> dict:
    """Connection, event-rate, sequence, freshness, resync, and storage status."""
    collector = _collector(request)
    return collector.get_status()


@router.get("/lob/latest")
def get_latest_lob(request: Request, symbol: str = Query(...)) -> dict:
    """Return the latest valid sampled top-20 local book for one configured symbol."""
    value = _symbol(symbol)
    collector = _collector(request)
    status = collector.get_status()["symbol_status"][value]
    if not status["book_valid"]:
        raise HTTPException(status_code=503, detail={"message": "No fresh synchronized book is available", "status": status})
    latest = collector.get_latest_book(value)
    if latest is None:
        raise HTTPException(status_code=503, detail="Waiting for the first clock-time LOB sample")
    return latest


@router.get("/features/latest")
def get_latest_features(request: Request, symbol: str = Query(...)) -> dict:
    """Return the most recent LOB feature row and its data-quality flags."""
    value = _symbol(symbol)
    collector = _collector(request)
    status = collector.get_status()["symbol_status"][value]
    if not status["book_valid"]:
        raise HTTPException(status_code=503, detail={"message": "No fresh synchronized feature is available", "status": status})
    features = collector.get_latest_features(value)
    if features is None:
        raise HTTPException(status_code=503, detail="Waiting for the first LOB feature sample")
    return features


@router.get("/forecasts/latest")
def get_latest_lob_forecasts(request: Request, symbol: str = Query(...)) -> dict:
    """Return live multi-horizon RV/risk forecasts and prequential model comparisons."""
    value = _symbol(symbol)
    collector = _collector(request)
    forecast = collector.get_latest_forecasts(value)
    forecast["long_horizons"] = collector.get_latest_long_forecasts(value)
    status = collector.get_status()["symbol_status"][value]
    forecast["feed_status"] = status["status"]
    forecast["book_valid"] = status["book_valid"]
    forecast["last_error"] = status.get("last_error")
    forecast["quality_flags"] = status.get("quality_flags", [])
    return forecast
