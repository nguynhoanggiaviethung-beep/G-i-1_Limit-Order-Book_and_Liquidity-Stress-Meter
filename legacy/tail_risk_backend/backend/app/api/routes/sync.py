"""Automatic data synchronization status route."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1", tags=["data synchronization"])


@router.get("/sync-status")
def get_sync_status(request: Request) -> dict:
    """Return the latest background Binance sync state."""
    service = getattr(request.app.state, "auto_sync_service", None)
    if service is None:
        return {"enabled": False, "status": "not_started"}
    return service.get_status()
