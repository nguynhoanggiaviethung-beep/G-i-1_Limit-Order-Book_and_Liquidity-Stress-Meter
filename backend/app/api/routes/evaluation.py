"""Evaluation result routes."""

from fastapi import APIRouter, HTTPException, Query
from app.services.evaluation_service import get_evaluation as load_evaluation

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.get("/{symbol}")
def get_evaluation(symbol: str, horizon_days: int | None = Query(default=None, ge=1, le=30)):
    try:
        if horizon_days not in (None, 1, 7, 30):
            raise ValueError("horizon_days must be 1, 7, or 30")
        return load_evaluation(symbol, horizon_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
