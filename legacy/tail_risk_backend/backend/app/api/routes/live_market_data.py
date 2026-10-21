"""Routes for live, in-progress Binance candles."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/v1", tags=["live market data"])


@router.get("/live-market-data")
def get_all_live_market_data(request: Request) -> dict:
    service = getattr(request.app.state, "live_price_service", None)
    if service is None:
        return {"status": "not_started", "candles": {}}
    return service.get_status()


@router.get("/live-market-data/{symbol}")
def get_live_market_data(symbol: str, request: Request) -> dict:
    service = getattr(request.app.state, "live_price_service", None)
    if service is None:
        return {"symbol": symbol.upper(), "connection_status": "not_started", "candle": None}
    try:
        return service.get_symbol(symbol)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol.upper()} is not configured") from exc
