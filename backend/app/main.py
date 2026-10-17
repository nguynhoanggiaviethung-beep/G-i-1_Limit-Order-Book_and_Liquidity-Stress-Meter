"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import evaluation, forecasts, health, live_market_data, market_data, regimes, sync
from app.core.config import load_settings
from app.core.logging import configure_logging
from app.services.auto_sync_service import AutoSyncService
from app.services.live_price_service import BinanceLiveKlineService

configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = load_settings()
    if settings.auto_sync_interval_seconds < 60:
        raise ValueError("auto_sync.interval_seconds must be at least 60 seconds")
    service = AutoSyncService(settings)
    live_service = BinanceLiveKlineService(settings)
    application.state.auto_sync_service = service
    application.state.live_price_service = live_service
    tasks = []
    if settings.auto_sync_enabled:
        tasks.append(asyncio.create_task(service.run_forever(), name="binance-auto-sync"))
    tasks.append(asyncio.create_task(live_service.run_forever(), name="binance-live-kline"))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Crypto Volatility and Tail Risk API", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(market_data.router)
app.include_router(forecasts.router)
app.include_router(evaluation.router)
app.include_router(sync.router)
app.include_router(live_market_data.router)
app.include_router(regimes.router)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    """Send browser visits to the interactive API documentation."""
    return RedirectResponse(url="/docs")
