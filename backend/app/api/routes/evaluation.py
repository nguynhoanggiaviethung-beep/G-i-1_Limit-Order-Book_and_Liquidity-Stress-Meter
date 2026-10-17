"""Evaluation result routes."""

from fastapi import APIRouter, HTTPException
from app.services.evaluation_service import get_evaluation as load_evaluation

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.get("/{symbol}")
def get_evaluation(symbol: str):
    try:
        return load_evaluation(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
