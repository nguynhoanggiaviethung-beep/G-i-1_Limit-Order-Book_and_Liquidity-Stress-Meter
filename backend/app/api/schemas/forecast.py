"""Research forecast request schema, using the frozen day horizons."""

from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    symbol: Literal["BTCUSDT", "ETHUSDT"]
    horizon_days: Literal[1, 7, 30] = 1
    model: str = "random_walk"
    confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
