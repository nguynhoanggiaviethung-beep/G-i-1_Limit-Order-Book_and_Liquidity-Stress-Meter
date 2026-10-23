"""Continuous, read-only Binance market feed for the live dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import Settings, load_settings
from app.data.cryptohft_replay import OneSecondBookReplay
from app.data.grid_store import GridStore
from app.data.websocket_collector import BinanceWebSocketCollector
from app.features.classification_features import build_classification_features
from app.features.microstructure import compute_microstructure_features
from app.services.replay_service import ReplayService, _snapshot

logger = logging.getLogger(__name__)


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _SymbolState:
    symbol: str
    status: str = "starting"
    reason: str | None = "Starting live market-data collector"
    snapshot: dict[str, Any] | None = None
    model_features: dict[str, Any] | None = None
    engine: OneSecondBookReplay | None = None
    history: deque[dict[str, Any]] = field(default_factory=deque)
    queues: set[asyncio.Queue] = field(default_factory=set)
    last_event_monotonic: float | None = None
    version: int = 0
    updated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class LiveMarketService:
    """Own one validated collector and publish latest-only updates to SSE clients."""

    def __init__(
        self,
        replay_service: ReplayService,
        settings: Settings | None = None,
        collector_factory=BinanceWebSocketCollector,
    ) -> None:
        self.settings = settings or load_settings()
        self.replay_service = replay_service
        self.collector_factory = collector_factory
        self.states = {
            symbol: _SymbolState(
                symbol=symbol,
                history=deque(maxlen=max(self.settings.lob_feature_lookback_seconds + 10, 310)),
            )
            for symbol in self.settings.symbols
        }
        self.collector: BinanceWebSocketCollector | None = None
        self.collector_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.settings.lob_collection_enabled:
            for symbol in self.states:
                self._set_status(symbol, "disabled", "Live collection is disabled in runtime settings")
            return
        self.collector = self.collector_factory(
            self.settings,
            on_book_sync=self._on_book_sync,
            on_market_event=self._on_market_event,
            on_status=self._on_collector_status,
        )
        self.collector_task = asyncio.create_task(self._run_collector(), name="binance-live-collector")
        self.monitor_task = asyncio.create_task(self._monitor_freshness(), name="binance-live-freshness")

    async def stop(self) -> None:
        tasks = [task for task in (self.collector_task, self.monitor_task) if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for symbol in self.states:
            self._set_status(symbol, "stopped", "Backend stopped the live collector")
        self._started = False

    async def _run_collector(self) -> None:
        try:
            assert self.collector is not None
            await self.collector.run(tuple(self.states))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Live Binance collector stopped unexpectedly")
            for symbol in self.states:
                self._set_status(symbol, "error", str(exc))

    async def _monitor_freshness(self) -> None:
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            for state in self.states.values():
                if (
                    state.snapshot is not None
                    and state.last_event_monotonic is not None
                    and now - state.last_event_monotonic > self.settings.lob_stale_after_seconds
                    and state.status not in {"stale", "stopped"}
                ):
                    state.snapshot = None
                    state.model_features = None
                    self._set_status(
                        state.symbol,
                        "stale",
                        f"No Binance market events received for more than {self.settings.lob_stale_after_seconds} seconds",
                    )

    def current(self, symbol: str) -> dict[str, Any]:
        state = self._state(symbol)
        return self._payload(state)

    def subscribe(self, symbol: str) -> asyncio.Queue:
        state = self._state(symbol)
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        state.queues.add(queue)
        queue.put_nowait(self._payload(state))
        return queue

    def unsubscribe(self, symbol: str, queue: asyncio.Queue) -> None:
        self._state(symbol).queues.discard(queue)

    def for_horizon(self, payload: dict[str, Any], horizon_seconds: int) -> dict[str, Any]:
        snapshot = payload.get("snapshot")
        symbol = str(payload["symbol"])
        if snapshot is None:
            meter = {
                "schema": "liquidity_meter.v1",
                "symbol": symbol,
                "timestamp_utc": None,
                "horizon_seconds": int(horizon_seconds),
                "status": "unavailable",
                "reason": payload.get("reason") or "Waiting for a synchronized live order book",
                "quality": {"book_valid": False},
                "p_up": None,
                "p_flat": None,
                "p_down": None,
                "p_stress": None,
                "stress_probability_threshold": None,
                "model_version": None,
            }
        else:
            meter = self.replay_service.predict_features(
                symbol,
                snapshot["timestamp_utc"],
                int(horizon_seconds),
                payload.get("model_features") or snapshot.get("features") or {},
                snapshot.get("quality") or {"book_valid": True},
            )
        return {key: value for key, value in payload.items() if key != "model_features"} | {
            "meter": meter,
            "horizon_seconds": int(horizon_seconds),
        }

    def _state(self, symbol: str) -> _SymbolState:
        normalized = symbol.upper()
        if normalized not in self.states:
            raise KeyError(normalized)
        return self.states[normalized]

    def _payload(self, state: _SymbolState) -> dict[str, Any]:
        return {
            "schema": "live_market.v1",
            "symbol": state.symbol,
            "status": state.status,
            "reason": state.reason,
            "updated_at_utc": state.updated_at_utc,
            "snapshot": state.snapshot,
            "model_features": state.model_features,
            "version": state.version,
        }

    def _publish(self, state: _SymbolState) -> None:
        state.version += 1
        state.updated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = self._payload(state)
        for queue in tuple(state.queues):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def _set_status(self, symbol: str, status: str, reason: str | None) -> None:
        state = self._state(symbol)
        if status in {"connecting", "reconnecting", "syncing", "error", "disabled", "stopped"}:
            state.snapshot = None
            state.model_features = None
            state.engine = None
            state.history.clear()
        state.status = status
        state.reason = reason
        self._publish(state)

    def _on_collector_status(self, symbol: str, status: str, reason: str | None) -> None:
        self._set_status(symbol, status, reason)

    def _on_book_sync(self, checkpoint: dict[str, Any]) -> None:
        symbol = str(checkpoint["symbol"]).upper()
        state = self._state(symbol)
        tick_size = float((self.settings.lob_tick_size_by_symbol or {}).get(symbol, 0.01))
        state.engine = OneSecondBookReplay(
            symbol,
            tick_size=tick_size,
            max_event_age_seconds=self.settings.lob_stale_after_seconds,
            levels=20,
        )
        state.history.clear()
        snapshot_event = {
            "kind": "snapshot",
            "symbol": symbol,
            "event_time_ms": int(checkpoint["event_time_ms"]),
            "book_event_time_ms": int(checkpoint["book_event_time_ms"]),
            "received_at_utc": checkpoint.get("received_at_utc"),
            "lastUpdateId": int(checkpoint["lastUpdateId"]),
            "bids": checkpoint["bids"],
            "asks": checkpoint["asks"],
        }
        list(state.engine.feed(snapshot_event))
        state.last_event_monotonic = time.monotonic()
        row = self._checkpoint_row(checkpoint)
        self._publish_row(state, row, "live", None, append_history=False)

    def _on_market_event(self, event: dict[str, Any]) -> None:
        symbol = str(event.get("symbol", "")).upper()
        if symbol not in self.states:
            return
        state = self.states[symbol]
        state.last_event_monotonic = time.monotonic()
        if state.engine is None:
            return
        try:
            rows = list(state.engine.feed(event))
        except (KeyError, TypeError, ValueError) as exc:
            state.snapshot = None
            state.model_features = None
            state.status = "syncing"
            state.reason = f"Waiting for a clean stream after event error: {exc}"
            self._publish(state)
            return
        for row in rows:
            state.history.append(row)
            self._publish_row(state, row, "live", None, append_history=False)

    @staticmethod
    def _checkpoint_row(checkpoint: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "symbol": checkpoint["symbol"],
            "timestamp_utc": _iso_utc(int(checkpoint["book_event_time_ms"])),
            "event_time_utc": _iso_utc(int(checkpoint["book_event_time_ms"])),
            "received_at_utc": checkpoint.get("received_at_utc"),
            "last_update_id": int(checkpoint["lastUpdateId"]),
            "book_valid": True,
            "sequence_gap": False,
            "seconds_since_gap": None,
        }
        for side in ("bid", "ask"):
            levels = checkpoint["bids"] if side == "bid" else checkpoint["asks"]
            for index in range(20):
                price, quantity = levels[index] if index < len(levels) else (None, None)
                row[f"{side}_price_{index + 1}"] = price
                row[f"{side}_qty_{index + 1}"] = quantity
        return row

    def _publish_row(
        self,
        state: _SymbolState,
        row: dict[str, Any],
        status: str,
        reason: str | None,
        *,
        append_history: bool,
    ) -> None:
        if append_history:
            state.history.append(row)
        if state.history:
            raw = pd.DataFrame(list(state.history))
            featured = build_classification_features(
                raw,
                impact_notional_usdt=float(self.settings.impact_notional_for(state.symbol)),
                tick_size=float(self.settings.tick_size_for(state.symbol)),
            )
            feature_row = GridStore.json_row(featured.iloc[-1])
        else:
            feature_row = {
                **row,
                **compute_microstructure_features(
                    row,
                    float(self.settings.impact_notional_for(state.symbol)),
                    float(self.settings.tick_size_for(state.symbol)),
                ),
            }
            feature_row = GridStore.json_row(pd.Series(feature_row))
        snapshot = _snapshot(
            feature_row,
            depth=20,
            impact_notional=float(self.settings.impact_notional_for(state.symbol)),
            tick_size=float(self.settings.tick_size_for(state.symbol)),
        )
        state.snapshot = snapshot
        state.model_features = feature_row
        state.status = status
        state.reason = reason
        state.last_event_monotonic = time.monotonic()
        self._publish(state)
