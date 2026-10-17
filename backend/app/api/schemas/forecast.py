"""Forecast API schemas."""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    symbol: str
    horizon_minutes: int = Field(gt=0)
    confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    model: str = "benchmark"
