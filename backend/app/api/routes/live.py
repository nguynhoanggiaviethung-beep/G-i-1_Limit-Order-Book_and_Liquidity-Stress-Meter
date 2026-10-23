"""Server-sent live snapshots from the read-only Binance collector."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.core.config import load_settings

router = APIRouter(prefix="/api/v1", tags=["Live market data"])


def _live_service(request: Request):
    service = getattr(request.app.state, "live_market_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Live market service is unavailable")
    return service


def _symbol(value: str) -> str:
    symbol = value.upper()
    if symbol not in load_settings().symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} is not configured")
    return symbol


@router.get("/live/status")
def get_live_status(request: Request) -> dict:
    service = _live_service(request)
    return {"schema": "live_status.v1", "symbols": {symbol: service.current(symbol) for symbol in load_settings().symbols}}


@router.get("/live/{symbol}/stream")
async def stream_live_market(
    request: Request,
    symbol: str,
    horizon_seconds: int = Query(5, ge=1, le=60),
) -> StreamingResponse:
    symbol = _symbol(symbol)
    service = _live_service(request)
    queue = service.subscribe(symbol)

    async def events():
        try:
            yield "retry: 1500\n\n"
            while not await request.is_disconnected():
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                message = service.for_horizon(payload, horizon_seconds)
                yield f"data: {json.dumps(message, ensure_ascii=False, allow_nan=False)}\n\n"
        finally:
            service.unsubscribe(symbol, queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
