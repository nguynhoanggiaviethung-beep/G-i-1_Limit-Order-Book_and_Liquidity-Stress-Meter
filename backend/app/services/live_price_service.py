"""Live in-progress Binance kline updates through the public WebSocket stream."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from app.core.config import Settings


LOGGER = logging.getLogger(__name__)
STREAM_BASE_URL = "wss://stream.binance.com:9443/stream?streams="


def _utc_from_ms(value: int | str | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()


class BinanceLiveKlineService:
    """Maintain the latest live kline event for each configured symbol."""

    def __init__(self, settings: Settings) -> None:
        self.symbols = tuple(symbol.upper() for symbol in settings.symbols)
        self.interval = settings.interval
        streams = "/".join(f"{symbol.lower()}@kline_{self.interval}" for symbol in self.symbols)
        self.url = STREAM_BASE_URL + streams
        self._status: dict = {
            "status": "connecting",
            "provider": "Binance Spot WebSocket",
            "interval": self.interval,
            "symbols": list(self.symbols),
            "connected_at_utc": None,
            "last_message_at_utc": None,
            "last_error": None,
            "candles": {},
        }

    def get_status(self) -> dict:
        return {**self._status, "candles": dict(self._status["candles"])}

    def get_symbol(self, symbol: str) -> dict:
        symbol = symbol.upper()
        if symbol not in self.symbols:
            raise KeyError(symbol)
        return {
            "symbol": symbol,
            "interval": self.interval,
            "connection_status": self._status["status"],
            "connected_at_utc": self._status["connected_at_utc"],
            "last_message_at_utc": self._status["last_message_at_utc"],
            "last_error": self._status["last_error"],
            "candle": self._status["candles"].get(symbol),
        }

    def _handle_message(self, raw_message: str) -> None:
        message = json.loads(raw_message)
        event = message.get("data", message)
        kline = event.get("k")
        if not kline:
            return
        symbol = str(event.get("s", kline.get("s", ""))).upper()
        if symbol not in self.symbols:
            return
        received_at = datetime.now(timezone.utc).isoformat()
        self._status["last_message_at_utc"] = received_at
        self._status["last_error"] = None
        self._status["candles"][symbol] = {
            "symbol": symbol,
            "interval": kline.get("i", self.interval),
            "open_time_utc": _utc_from_ms(kline.get("t")),
            "close_time_utc": _utc_from_ms(kline.get("T")),
            "open": kline.get("o"),
            "high": kline.get("h"),
            "low": kline.get("l"),
            "price": kline.get("c"),
            "base_volume": kline.get("v"),
            "quote_volume": kline.get("q"),
            "trade_count": int(kline.get("n", 0)),
            "is_closed": bool(kline.get("x", False)),
            "event_time_utc": _utc_from_ms(event.get("E")),
            "received_at_utc": received_at,
        }

    async def run_forever(self) -> None:
        """Consume the combined stream and reconnect automatically with backoff."""
        backoff = 1
        while True:
            self._status["status"] = "connecting"
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    open_timeout=15,
                ) as connection:
                    self._status.update(
                        status="connected",
                        connected_at_utc=datetime.now(timezone.utc).isoformat(),
                        last_error=None,
                    )
                    LOGGER.info("Connected to Binance live kline stream for %s", self.symbols)
                    backoff = 1
                    async for raw_message in connection:
                        try:
                            self._handle_message(raw_message)
                        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                            LOGGER.exception("Ignored malformed Binance WebSocket message")
                    raise ConnectionError("Binance live kline stream closed")
            except asyncio.CancelledError:
                self._status["status"] = "stopped"
                raise
            except Exception as exc:
                self._status.update(
                    status="reconnecting",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                LOGGER.warning("Binance live stream disconnected; retrying in %s seconds: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
