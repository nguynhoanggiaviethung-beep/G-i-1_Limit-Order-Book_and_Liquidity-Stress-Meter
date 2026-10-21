"""Evaluation API schemas."""

from typing import Any

from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    symbol: str
    status: str
    metrics: dict[str, Any] = {}
