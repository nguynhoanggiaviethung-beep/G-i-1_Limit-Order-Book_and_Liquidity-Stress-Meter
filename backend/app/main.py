"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.api.routes import health, live, replay
from app.core.config import REPOSITORY_ROOT
from app.core.logging import configure_logging
from app.services.replay_service import ReplayService
from app.services.live_market_service import LiveMarketService

configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    live_service = LiveMarketService(application.state.replay_service)
    application.state.live_market_service = live_service
    await live_service.start()
    try:
        yield
    finally:
        await live_service.stop()


app = FastAPI(title="Limit Order Book & Liquidity Stress Meter API", version="1.0.0", lifespan=lifespan)
app.state.replay_service = ReplayService()
app.state.live_market_service = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(replay.router)
app.include_router(live.router)


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
    if frontend_path.split("/", 1)[0] in {"api", "docs", "redoc", "openapi.json", "health", "collector", "lob", "features", "live"}:
        raise HTTPException(status_code=404, detail="Not Found")
    dist = (REPOSITORY_ROOT / "frontend" / "dist").resolve()
    index = dist / "index.html"
    candidate = (dist / frontend_path).resolve()
    if dist in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend is not built; run npm --prefix frontend run build")
