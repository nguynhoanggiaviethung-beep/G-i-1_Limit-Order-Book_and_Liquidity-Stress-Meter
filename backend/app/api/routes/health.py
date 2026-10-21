"""Application health route."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["health"])
def health(request: Request) -> dict:
    """Report process health and the current state of each configured LOB feed."""
    collector = getattr(request.app.state, "lob_collector", None)
    collector_status = collector.get_status() if collector is not None else {"status": "not_started"}
    return {"status": "ok", "collector_status": collector_status}
