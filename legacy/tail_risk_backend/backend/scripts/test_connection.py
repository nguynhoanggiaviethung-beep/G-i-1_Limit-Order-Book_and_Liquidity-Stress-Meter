"""Check public Binance depth REST and Spot diff-depth WebSocket connectivity."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_settings
from app.ingestion.binance_rest import BinanceDepthClient
from app.ingestion.binance_ws import connect_depth_stream, depth_stream_url


async def main() -> None:
    settings = load_settings()
    rest = BinanceDepthClient(
        settings.binance_base_url,
        settings.request_timeout_seconds,
        settings.request_retries,
        fallback_base_urls=settings.binance_fallback_urls,
    )
    for symbol in settings.symbols:
        snapshot = await asyncio.to_thread(rest.fetch_snapshot, symbol, 100)
        url = depth_stream_url(settings.lob_ws_url, symbol, settings.lob_update_speed)
        async with connect_depth_stream(url) as websocket:
            event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15))
        payload = event.get("data", event)
        if payload.get("e") != "depthUpdate" or payload.get("s", "").upper() != symbol:
            raise RuntimeError(f"Unexpected depth stream response for {symbol}: {payload}")
        print(f"{symbol}: REST snapshot updateId={snapshot['lastUpdateId']}; WebSocket event {payload['U']}-{payload['u']}")


if __name__ == "__main__":
    asyncio.run(main())
