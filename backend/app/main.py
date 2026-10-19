"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.api.routes import (
    evaluation,
    forecasts,
    health,
    live_market_data,
    market_data,
    regimes,
    research_data,
    sync,
)
from app.core.config import REPOSITORY_ROOT, load_settings
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(market_data.router)
app.include_router(forecasts.router)
app.include_router(evaluation.router)
app.include_router(sync.router)
app.include_router(live_market_data.router)
app.include_router(regimes.router)
app.include_router(research_data.router)


@app.get("/", include_in_schema=False)
def home() -> Response:
    """Serve the built frontend when present, otherwise open API documentation."""
    index = REPOSITORY_ROOT / "frontend" / "dist" / "index.html"
    if index.is_file():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_fallback(frontend_path: str) -> Response:
    """Serve static frontend assets and route deep links to the SPA entry point."""
    if frontend_path.split("/", 1)[0] in {"api", "docs", "redoc", "openapi.json", "health"}:
        raise HTTPException(status_code=404, detail="Not Found")
    dist = (REPOSITORY_ROOT / "frontend" / "dist").resolve()
    index = dist / "index.html"
    candidate = (dist / frontend_path).resolve()
    if dist in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend is not built; run npm --prefix frontend run build")
