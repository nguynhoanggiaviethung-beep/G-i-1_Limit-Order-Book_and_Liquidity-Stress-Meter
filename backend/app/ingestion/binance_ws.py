"""Binance Spot diff-depth WebSocket connection helpers."""

from __future__ import annotations

import websockets


def depth_stream_url(base_url: str, symbol: str, update_speed: str = "100ms") -> str:
    suffix = f"@depth@{update_speed}" if update_speed else "@depth"
    return f"{base_url.rstrip('/')}/ws/{symbol.lower()}{suffix}"


def connect_depth_stream(url: str):
    return websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        open_timeout=8,
        max_queue=2048,
    )
