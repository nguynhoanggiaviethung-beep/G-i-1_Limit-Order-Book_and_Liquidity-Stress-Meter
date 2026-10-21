"""Application health route."""

from fastapi import APIRouter, Request
from app.core.config import load_settings

router = APIRouter()


@router.get("/health", tags=["health"])
def health(request: Request) -> dict:
    """Report process health and whether replay data is indexed."""
    service = getattr(request.app.state, "replay_service", None)
    return {"status": "ok", "replay_status": service.store.catalog(load_settings().symbols)["status"] if service else "unavailable"}
