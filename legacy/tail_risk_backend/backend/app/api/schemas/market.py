"""Market data API schemas."""

from datetime import datetime

from pydantic import BaseModel


class MarketCandle(BaseModel):
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataResponse(BaseModel):
    symbol: str
    interval: str
    rows: int
    candles: list[MarketCandle]
