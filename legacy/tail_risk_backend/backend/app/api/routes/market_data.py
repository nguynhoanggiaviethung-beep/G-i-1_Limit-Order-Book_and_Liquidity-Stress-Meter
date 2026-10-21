"""Read-only routes for locally stored market candles."""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone

from app.api.schemas.market import MarketDataResponse
from app.services.data_service import load_market_data

router = APIRouter(prefix="/api/v1/market-data", tags=["market data"])


@router.get("/{symbol}", response_model=MarketDataResponse)
def get_market_data(
    symbol: str,
    interval: str = Query(default="5m"),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(default=1000, ge=1, le=10000),
):
    try:
        frame = load_market_data(symbol, interval)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        frame = frame[frame["open_time"] >= start_time]
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        frame = frame[frame["open_time"] <= end_time]
    frame = frame.tail(limit)
    return {"symbol": symbol.upper(), "interval": interval, "rows": len(frame), "candles": frame.to_dict(orient="records")}
