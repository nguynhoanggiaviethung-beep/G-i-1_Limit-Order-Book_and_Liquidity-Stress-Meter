"""Independent, reconnecting LOB collectors for configured Binance symbols."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import websockets

from app.core.config import Settings
from app.features.lob_book_features import compute_book_features
from app.features.lob_temporal_features import TemporalFeatureEngine
from app.ingestion.binance_rest import BinanceDepthClient
from app.ingestion.binance_ws import connect_depth_stream, depth_stream_url
from app.orderbook.book_state import InvalidBook, LocalOrderBook, SequenceGap
from app.orderbook.integrity import event_age_ms
from app.orderbook.synchronizer import bootstrap_book
from app.services.lob_forecast import LOBOnlineForecaster
from app.services.lob_long_forecast import LOBLongHorizonForecaster
from app.storage.parquet_writer import LOBParquetWriter
from app.storage.raw_writer import RawLOBWriter


LOGGER = logging.getLogger(__name__)
COLLECTOR_VERSION = "lob-collector-1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_from_message(raw: str | bytes) -> tuple[dict, str | None]:
    message = json.loads(raw)
    event = message.get("data", message)
    stream = message.get("stream")
    if not isinstance(event, dict) or event.get("e") != "depthUpdate":
        raise ValueError("Expected a Binance diff-depth event")
    for required in ("s", "E", "U", "u", "b", "a"):
        if required not in event:
            raise ValueError(f"Depth event is missing {required}")
    if int(event["E"]) < 0 or int(event["U"]) < 0 or int(event["u"]) < int(event["U"]):
        raise ValueError("Depth event has invalid event time or update-ID range")
    if not isinstance(event["b"], list) or not isinstance(event["a"], list):
        raise ValueError("Depth event bid/ask changes must be arrays")
    return event, stream


async def _next_event(queue: asyncio.Queue, receiver: asyncio.Task, timeout: float):
    queued = asyncio.create_task(queue.get())
    done, _ = await asyncio.wait({queued, receiver}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    if queued in done:
        return queued.result()
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)
    if receiver in done:
        receiver.result()
        raise ConnectionError("Binance depth WebSocket closed")
    return None


class BinanceLOBCollector:
    """Capture raw depth, synchronize the book, and emit fresh clock-time samples."""

    def __init__(self, settings: Settings, client: BinanceDepthClient | None = None) -> None:
        self.settings = settings
        self.client = client
        self._clients = {
            symbol: client or BinanceDepthClient(
                settings.binance_base_url,
                settings.request_timeout_seconds,
                settings.request_retries,
                fallback_base_urls=settings.binance_fallback_urls,
            )
            for symbol in settings.symbols
        }
        self._websocket_urls = tuple(dict.fromkeys((settings.lob_ws_url, *settings.lob_ws_fallback_urls)))
        self._websocket_url_index = {symbol: 0 for symbol in settings.symbols}
        self.raw_writer = RawLOBWriter(settings.data_dir, settings.lob_raw_file_max_bytes)
        self._states = {
            symbol: {
                "symbol": symbol,
                "collector_started_at_utc": _utc_now(),
                "status": "starting",
                "connection_id": None,
                "connected_at_utc": None,
                "last_event_time_utc": None,
                "last_received_at_utc": None,
                "last_sample_time_utc": None,
                "last_update_id": None,
                "book_valid": False,
                "quality_flags": ["awaiting_snapshot"],
                "event_rate_per_second": 0.0,
                "latency_ms": None,
                "reconnect_count": 0,
                "resync_count": 0,
                "sample_count": 0,
                "expected_sample_count": 0,
                "valid_sample_count": 0,
                "book_valid_seconds": 0,
                "book_valid_ratio": 0.0,
                "last_error": None,
                "latest": None,
                "disk_free_bytes": None,
            }
            for symbol in settings.symbols
        }
        self._event_times: dict[str, deque[float]] = {symbol: deque() for symbol in settings.symbols}
        self._latest_features: dict[str, dict | None] = {symbol: None for symbol in settings.symbols}
        self._latest_snapshots: dict[str, dict | None] = {symbol: None for symbol in settings.symbols}
        self._forecasters = {
            symbol: LOBOnlineForecaster(
                symbol,
                settings.lob_target_horizons_seconds,
                settings.lob_sampling_interval_seconds,
                settings.lob_online_minimum_train_examples,
                settings.lob_online_maximum_train_examples,
                settings.lob_online_refit_every_examples,
            )
            for symbol in settings.symbols
        }
        self._long_forecasters = {
            symbol: LOBLongHorizonForecaster(
                symbol,
                settings.lob_long_target_horizons_days,
                settings.lob_sampling_interval_seconds,
                settings.lob_long_online_minimum_train_examples,
                settings.lob_long_online_maximum_train_examples,
                settings.lob_long_minimum_daily_coverage,
            )
            for symbol in settings.symbols
        }

    def get_status(self) -> dict:
        now = time.monotonic()
        result = {}
        for symbol, state in self._states.items():
            events = self._event_times[symbol]
            while events and now - events[0] > 60:
                events.popleft()
            copied = dict(state)
            copied["event_rate_per_second"] = round(len(events) / 60, 3)
            started = datetime.fromisoformat(copied["collector_started_at_utc"])
            elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 1.0)
            copied["book_valid_time_ratio"] = min(1.0, round(copied["book_valid_seconds"] / elapsed, 6))
            result[symbol] = copied
        symbol_states = [value["status"] for value in result.values()]
        overall = "connected" if symbol_states and all(value == "connected" for value in symbol_states) else (
            "degraded" if any(value == "connected" for value in symbol_states) else "reconnecting"
        )
        return {
            "status": overall,
            "provider": "Binance Spot REST depth + diff-depth WebSocket",
            "symbols": list(self.settings.symbols),
            "sampling_interval_seconds": self.settings.lob_sampling_interval_seconds,
            "stale_after_seconds": self.settings.lob_stale_after_seconds,
            "symbol_status": result,
        }

    def get_latest_book(self, symbol: str) -> dict | None:
        return self._latest_snapshots.get(symbol.upper())

    def get_latest_features(self, symbol: str) -> dict | None:
        return self._latest_features.get(symbol.upper())

    def get_latest_forecasts(self, symbol: str) -> dict:
        return self._forecasters[symbol.upper()].get_latest()

    def get_latest_long_forecasts(self, symbol: str) -> dict:
        return self._long_forecasters[symbol.upper()].get_latest()

    async def run_forever(self) -> None:
        """Run each symbol in its own task so a symbol failure is isolated."""
        await asyncio.gather(*(
            asyncio.to_thread(self._long_forecasters[symbol].load_history, self.settings.data_dir)
            for symbol in self.settings.symbols
        ))
        await asyncio.gather(*(self._run_symbol_forever(symbol) for symbol in self.settings.symbols))

    async def _run_symbol_forever(self, symbol: str) -> None:
        retry = 0
        while True:
            state = self._states[symbol]
            connection_id = uuid.uuid4().hex
            connection_started = time.monotonic()
            base_url = self._websocket_urls[self._websocket_url_index[symbol]]
            state.update(status="connecting", book_valid=False, quality_flags=["connecting"], connection_id=connection_id)
            try:
                url = depth_stream_url(base_url, symbol, self.settings.lob_update_speed)
                async with connect_depth_stream(url) as websocket:
                    state.update(status="synchronizing", connected_at_utc=_utc_now(), last_error=None)
                    await self._consume_connection(symbol, websocket, connection_id)
                    raise ConnectionError("Binance depth stream closed")
            except asyncio.CancelledError:
                self._forecasters[symbol].mark_gap()
                self._long_forecasters[symbol].mark_gap()
                state.update(status="stopped", book_valid=False, quality_flags=["collector_stopped"])
                raise
            except Exception as exc:
                self._forecasters[symbol].mark_gap()
                self._long_forecasters[symbol].mark_gap()
                was_healthy = state.get("status") == "connected"
                retry = 0 if was_healthy and time.monotonic() - connection_started >= 30 else retry + 1
                state["reconnect_count"] += 1
                if isinstance(exc, (SequenceGap, InvalidBook)):
                    state["resync_count"] += 1
                state.update(
                    status="reconnecting",
                    book_valid=False,
                    quality_flags=["reconnecting"],
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                self._websocket_url_index[symbol] = (self._websocket_url_index[symbol] + 1) % len(self._websocket_urls)
                LOGGER.warning("LOB collector %s disconnected from %s: %s", symbol, base_url, exc)
                delay = min(2 ** min(retry - 1, 5), 15) + random.random()
                await asyncio.sleep(delay)

    async def _receive(
        self,
        symbol: str,
        websocket,
        connection_id: str,
        queue: asyncio.Queue,
    ) -> None:
        async for raw in websocket:
            received = datetime.now(timezone.utc)
            try:
                event, stream = _event_from_message(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                LOGGER.warning("Ignoring malformed diff-depth message for %s: %s", symbol, exc)
                continue
            if str(event.get("s", "")).upper() != symbol:
                LOGGER.warning("Ignoring depth event for unexpected symbol %s on %s", event.get("s"), symbol)
                continue
            received_iso = received.isoformat()
            await asyncio.to_thread(
                self.raw_writer.write_depth_event,
                symbol,
                event,
                stream or f"{symbol.lower()}@depth@{self.settings.lob_update_speed}",
                connection_id,
                COLLECTOR_VERSION,
                received_iso,
            )
            await queue.put((event, received))

    async def _consume_connection(self, symbol: str, websocket, connection_id: str) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        receiver = asyncio.create_task(self._receive(symbol, websocket, connection_id, queue))
        writer = LOBParquetWriter(
            self.settings.data_dir,
            self.settings.lob_flush_rows,
            self.settings.lob_flush_seconds,
        )
        temporal = TemporalFeatureEngine(max_history=self.settings.lob_temporal_history_samples)
        book: LocalOrderBook | None = None
        buffered: list[tuple[dict, datetime]] = []
        snapshot_resync_attempts = 0
        try:
            while book is None:
                snapshot_received = datetime.now(timezone.utc)
                snapshot = await asyncio.to_thread(
                    self._clients[symbol].fetch_snapshot, symbol, self.settings.lob_depth_snapshot_limit
                )
                await asyncio.to_thread(
                    self.raw_writer.write_snapshot,
                    symbol,
                    snapshot,
                    connection_id,
                    COLLECTOR_VERSION,
                    snapshot_received.isoformat(),
                )
                while True:
                    result = bootstrap_book(symbol, snapshot, [item[0] for item in buffered])
                    if result.waiting_for_event:
                        next_item = await _next_event(queue, receiver, timeout=30)
                        if next_item is None:
                            raise TimeoutError("No depth event arrived within 30 seconds")
                        buffered.append(next_item)
                        continue
                    if result.snapshot_too_old:
                        self._states[symbol]["resync_count"] += 1
                        snapshot_resync_attempts += 1
                        client = self._clients[symbol]
                        rotate = getattr(client, "rotate_endpoint", None)
                        next_endpoint = rotate() if rotate is not None else "configured REST endpoint"
                        self._states[symbol]["last_error"] = (
                            "Binance depth snapshot arrived behind the buffered update stream; "
                            f"switching REST endpoint to {next_endpoint} "
                            f"({snapshot_resync_attempts}/5)"
                        )
                        if snapshot_resync_attempts >= 5:
                            raise RuntimeError(
                                "Binance depth snapshot stayed behind the buffered WebSocket stream "
                                "after 5 attempts; check network latency or Binance REST access"
                            )
                        await asyncio.sleep(min(0.25 * snapshot_resync_attempts, 1.0))
                        break
                    book = result.book
                    assert book is not None
                    # bootstrap_book applies the first spanning event itself.
                    spanning = next(
                        i for i, (item, _) in enumerate(buffered)
                        if int(item["u"]) > int(snapshot["lastUpdateId"])
                    )
                    first_event, first_received = buffered[spanning]
                    book.received_at_utc = first_received.isoformat()
                    self._mark_event(symbol, book, first_event, first_received)
                    for item, received_time in buffered[spanning + 1 :]:
                        book.apply_update(item)
                        book.received_at_utc = received_time.isoformat()
                        self._mark_event(symbol, book, item, received_time)
                    buffered.clear()
                    break
            state = self._states[symbol]
            state.update(status="connected", book_valid=True, quality_flags=[], last_update_id=book.last_update_id, last_error=None)
            LOGGER.info("Synchronized Binance local book for %s at update ID %s", symbol, book.last_update_id)

            next_sample = asyncio.get_running_loop().time() + self.settings.lob_sampling_interval_seconds
            while True:
                wait = max(0.01, next_sample - asyncio.get_running_loop().time())
                next_item = await _next_event(queue, receiver, timeout=wait)
                if next_item is not None:
                    event, received = next_item
                    book.apply_update(event)
                    book.received_at_utc = received.isoformat()
                    self._mark_event(symbol, book, event, received)
                loop_now = asyncio.get_running_loop().time()
                if loop_now >= next_sample:
                    await self._sample(symbol, book, writer, temporal)
                    # Skip missed ticks instead of creating duplicate clock samples.
                    next_sample += (int((loop_now - next_sample) // self.settings.lob_sampling_interval_seconds) + 1) * self.settings.lob_sampling_interval_seconds
                for path in writer.flush_if_due():
                    self._states[symbol]["last_persisted_file"] = str(path)
        finally:
            if not receiver.done():
                receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            for path in writer.flush():
                self._states[symbol]["last_persisted_file"] = str(path)
            try:
                await asyncio.to_thread(self.raw_writer.finish_connection, symbol, connection_id)
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.exception("Could not write raw-data manifest for %s", symbol)

    def _mark_event(self, symbol: str, book: LocalOrderBook, event: dict, received: datetime) -> None:
        state = self._states[symbol]
        now = time.monotonic()
        self._event_times[symbol].append(now)
        state.update(
            last_event_time_utc=datetime.fromtimestamp(int(event["E"]) / 1000, tz=timezone.utc).isoformat(),
            last_received_at_utc=received.isoformat(),
            last_update_id=book.last_update_id,
            latency_ms=event_age_ms(received, event.get("E")),
            book_valid=bool(book.valid),
            quality_flags=[] if book.valid else [book.invalid_reason or "invalid_book"],
            last_error=None,
        )

    async def _sample(
        self,
        symbol: str,
        book: LocalOrderBook,
        writer: LOBParquetWriter,
        temporal: TemporalFeatureEngine,
    ) -> None:
        state = self._states[symbol]
        state["expected_sample_count"] += 1
        now = datetime.now(timezone.utc)
        last_received = datetime.fromisoformat(book.received_at_utc) if book.received_at_utc else None
        age_seconds = (now - last_received).total_seconds() if last_received else float("inf")
        if not book.valid or age_seconds > self.settings.lob_stale_after_seconds:
            self._forecasters[symbol].mark_gap()
            self._long_forecasters[symbol].mark_gap()
            temporal.reset()
            state.update(
                book_valid=False,
                quality_flags=["stale_feed" if age_seconds > self.settings.lob_stale_after_seconds else "invalid_book"],
                book_valid_ratio=round(state["valid_sample_count"] / max(state["expected_sample_count"], 1), 6),
            )
            return
        base = compute_book_features(book)
        event_time = datetime.fromtimestamp(book.event_time_ms / 1000, tz=timezone.utc).isoformat() if book.event_time_ms else None
        event_age = (
            max(0.0, (now - datetime.fromtimestamp(book.event_time_ms / 1000, tz=timezone.utc)).total_seconds() * 1000)
            if book.event_time_ms else None
        )
        row = {
            "symbol": symbol,
            "sample_time_utc": now.isoformat(),
            "event_time_utc": event_time,
            "received_at_utc": book.received_at_utc,
            "event_age_ms": event_age,
            "last_update_id": book.last_update_id,
            "book_valid": True,
            "quality_flags": [],
            "sampling_interval_seconds": self.settings.lob_sampling_interval_seconds,
            **base,
        }
        feature = temporal.transform(row)
        self._forecasters[symbol].update(feature)
        self._long_forecasters[symbol].update(feature)
        writer.append(feature, book)
        self._latest_features[symbol] = feature
        self._latest_snapshots[symbol] = {
            "symbol": symbol,
            "sample_time_utc": now.isoformat(),
            "event_time_utc": event_time,
            "received_at_utc": book.received_at_utc,
            "last_update_id": book.last_update_id,
            "book_valid": True,
            "bids": [[str(price), str(quantity)] for price, quantity in book.top_levels("bid", 20)],
            "asks": [[str(price), str(quantity)] for price, quantity in book.top_levels("ask", 20)],
            "features": feature,
        }
        state.update(
            status="connected",
            book_valid=True,
            quality_flags=[],
            last_sample_time_utc=now.isoformat(),
            sample_count=state["sample_count"] + 1,
            valid_sample_count=state["valid_sample_count"] + 1,
            book_valid_seconds=state["book_valid_seconds"] + self.settings.lob_sampling_interval_seconds,
            latest=self._latest_snapshots[symbol],
        )
        state["book_valid_ratio"] = round(
            state["valid_sample_count"] / max(state["expected_sample_count"], 1), 6
        )
        try:
            import shutil

            disk = shutil.disk_usage(self.settings.data_dir)
            state["disk_free_bytes"] = disk.free
            state["disk_total_bytes"] = disk.total
            state["disk_used_ratio"] = round(disk.used / max(disk.total, 1), 6)
        except OSError:
            state["disk_free_bytes"] = None
