"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.api.routes import health, lob
from app.core.config import REPOSITORY_ROOT, load_settings
from app.core.logging import configure_logging
from app.ingestion.collector import BinanceLOBCollector

configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = load_settings()
    if settings.lob_sampling_interval_seconds < 1:
        raise ValueError("lob.sampling_interval_seconds must be at least one second")
    if settings.lob_depth_snapshot_limit < 1 or settings.lob_depth_snapshot_limit > 5000:
        raise ValueError("lob.depth_snapshot_limit must be between 1 and 5000")
    collector = BinanceLOBCollector(settings)
    application.state.lob_collector = collector
    tasks = []
    if settings.lob_collection_enabled:
        tasks.append(asyncio.create_task(collector.run_forever(), name="binance-lob-collector"))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="LOB-Only Crypto Risk Research API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(lob.router)


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
    if frontend_path.split("/", 1)[0] in {"api", "docs", "redoc", "openapi.json", "health", "collector", "lob", "features"}:
        raise HTTPException(status_code=404, detail="Not Found")
    dist = (REPOSITORY_ROOT / "frontend" / "dist").resolve()
    index = dist / "index.html"
    candidate = (dist / frontend_path).resolve()
    if dist in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend is not built; run npm --prefix frontend run build")
