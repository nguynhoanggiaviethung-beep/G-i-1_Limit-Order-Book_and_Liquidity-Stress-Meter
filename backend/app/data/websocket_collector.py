"""Capture and synchronize live Binance Spot depth and trade streams."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.core.config import REPOSITORY_ROOT, Settings, load_settings
from app.data.binance_client import BinancePublicClient
from app.data.storage import PartitionedGzipJsonlWriter
from app.orderbook.book_state import InvalidBook, LocalOrderBook, SequenceGap
from app.orderbook.synchronizer import bootstrap_book


logger = logging.getLogger(__name__)


def _utc_iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _event_record(message: dict[str, Any], received_ms: int) -> dict[str, Any] | None:
    payload = message.get("data", message)
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("e", payload.get("kind", ""))).lower()
    symbol = str(payload.get("s", payload.get("symbol", ""))).upper()
    if kind in {"depthupdate", "depth_update", "depth"} or all(key in payload for key in ("U", "u", "b", "a")):
        if not all(key in payload for key in ("U", "u", "b", "a")):
            return None
        event_time = int(payload.get("E", received_ms))
        return {
            **payload,
            "kind": "depth_update",
            "symbol": symbol,
            "event_time_ms": event_time,
            "receive_time_ms": received_ms,
            "received_at_utc": _utc_iso(received_ms),
        }
    if kind in {"trade", "aggtrade", "agg_trade"} or all(key in payload for key in ("p", "q", "m")):
        trade_time = int(payload.get("T", payload.get("time", payload.get("E", received_ms))))
        event_time = int(payload.get("E", trade_time))
        price = payload.get("p", payload.get("price"))
        quantity = payload.get("q", payload.get("quantity"))
        buyer_maker = payload.get("m", payload.get("is_buyer_maker"))
        if price is None or quantity is None or buyer_maker is None:
            return None
        buyer_maker = _as_bool(buyer_maker)
        return {
            "kind": "trade",
            "symbol": symbol,
            "event_time_ms": event_time,
            "trade_timestamp_ms": trade_time,
            "receive_time_ms": received_ms,
            "received_at_utc": _utc_iso(received_ms),
            "price": str(price),
            "quantity": str(quantity),
            "is_buyer_maker": buyer_maker,
            "aggressor_side": "sell" if buyer_maker else "buy",
            "trade_id": payload.get("a", payload.get("t", payload.get("id"))),
            "raw_event": payload,
        }
    return None


def _snapshot_checkpoint(book: LocalOrderBook, event_time_ms: int, received_ms: int) -> dict[str, Any]:
    return {
        "kind": "snapshot",
        "symbol": book.symbol,
        "event_time_ms": int(event_time_ms),
        "receive_time_ms": int(received_ms),
        "received_at_utc": _utc_iso(received_ms),
        "book_event_time_ms": int(book.event_time_ms if book.event_time_ms is not None else event_time_ms),
        "lastUpdateId": int(book.last_update_id),
        "bids": [[str(price), str(quantity)] for price, quantity in book.top_levels("bid", len(book.bids))],
        "asks": [[str(price), str(quantity)] for price, quantity in book.top_levels("ask", len(book.asks))],
        "snapshot_source": "rest_snapshot_plus_buffered_depth_updates",
    }


def _rebuild_from_buffer(
    symbol: str, snapshot: dict[str, Any], pending: list[dict[str, Any]]
) -> tuple[LocalOrderBook | None, str]:
    depth_events = [event for event in pending if event.get("kind") == "depth_update"]
    try:
        result = bootstrap_book(symbol, snapshot, depth_events)
        if result.book is None:
            return None, "old" if result.snapshot_too_old else "waiting"
        book = result.book
        for event in result.remaining_events:
            book.apply_update(event)
            book.received_at_utc = event.get("received_at_utc")
        return book, "ready"
    except (InvalidBook, SequenceGap, KeyError, TypeError, ValueError):
        return None, "gap"


class BinanceWebSocketCollector:
    """Collect one or more public streams into auditable, replayable partitions."""

    def __init__(
        self,
        settings: Settings | None = None,
        output_root: Path | str | None = None,
        interim_root: Path | str | None = None,
        duration_seconds: float = 0,
        reconnect_max_seconds: float = 30,
    ) -> None:
        self.settings = settings or load_settings()
        root = Path(output_root) if output_root is not None else self.settings.raw_data_dir
        self.raw_root = root if root.is_absolute() else REPOSITORY_ROOT / root
        intermediate = Path(interim_root) if interim_root is not None else self.settings.interim_data_dir
        self.interim_root = intermediate if intermediate.is_absolute() else REPOSITORY_ROOT / intermediate
        self.capture_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.capture_root = self.raw_root / "captures" / self.capture_id
        self.interim_capture_root = self.interim_root / "captures" / self.capture_id
        self.duration_seconds = float(duration_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        if self.duration_seconds < 0 or self.reconnect_max_seconds < 1:
            raise ValueError("Capture duration must be non-negative and reconnect backoff at least one second")
        self.client = BinancePublicClient(self.settings)

    def _stream_url(self, base_url: str, symbol: str) -> str:
        speed = f"@{self.settings.lob_update_speed}" if self.settings.lob_update_speed else ""
        streams = f"{symbol.lower()}@depth{speed}/{symbol.lower()}@aggTrade"
        return f"{base_url.rstrip('/')}/stream?streams={quote(streams, safe='@/') }"

    async def _fetch_snapshot(self, symbol: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.client.depth_snapshot, symbol, self.settings.lob_depth_snapshot_limit)

    async def _reader(self, websocket, queue: asyncio.Queue) -> None:
        async for message in websocket:
            await queue.put(message)

    async def collect_symbol(self, symbol: str, stop_at: float | None = None) -> dict[str, Any]:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Install backend requirements to use the Binance WebSocket collector") from exc

        symbol = symbol.upper()
        if symbol not in self.settings.symbols:
            raise ValueError(f"{symbol} is not in the configured symbol list {self.settings.symbols}")
        symbol_root = self.capture_root
        event_writer = PartitionedGzipJsonlWriter(
            self.interim_capture_root / "events", symbol, self.settings.lob_raw_file_max_bytes,
            flush_rows=self.settings.lob_flush_rows, flush_seconds=self.settings.lob_flush_seconds,
        )
        websocket_writer = PartitionedGzipJsonlWriter(
            symbol_root / "websocket_raw", symbol, self.settings.lob_raw_file_max_bytes,
            flush_rows=self.settings.lob_flush_rows, flush_seconds=self.settings.lob_flush_seconds,
        )
        snapshot_writer = PartitionedGzipJsonlWriter(
            symbol_root / "rest_snapshots", symbol, self.settings.lob_raw_file_max_bytes,
            flush_rows=self.settings.lob_flush_rows, flush_seconds=self.settings.lob_flush_seconds,
        )
        audit_writer = PartitionedGzipJsonlWriter(
            symbol_root / "collector_logs", symbol, self.settings.lob_raw_file_max_bytes,
            flush_rows=self.settings.lob_flush_rows, flush_seconds=self.settings.lob_flush_seconds,
        )

        def audit(event_type: str, **details: Any) -> None:
            timestamp_ms = int(time.time() * 1000)
            audit_writer.write({
                "event_time_ms": timestamp_ms,
                "receive_time_ms": timestamp_ms,
                "timestamp_utc": _utc_iso(timestamp_ms),
                "symbol": symbol,
                "event_type": event_type,
                "details": details,
            })

        start_ms = int(time.time() * 1000)
        audit("capture_started", capture_id=self.capture_id)
        counters = {
            "websocket_messages": 0, "depth_events": 0, "trade_events": 0,
            "snapshot_requests": 0, "snapshot_bootstraps": 0, "snapshot_recoveries": 0,
            "sequence_gaps": 0, "invalid_book_events": 0, "reconnects": 0,
        }
        exchange_metadata: dict[str, Any] | None = None
        try:
            exchange_metadata = await asyncio.to_thread(self.client.exchange_symbol_metadata, symbol)
        except Exception as exc:
            logger.warning("Could not load exchange metadata for %s: %s", symbol, exc)
            audit("exchange_metadata_unavailable", error=str(exc))

        retry = 0
        base_urls = (self.settings.lob_ws_url, *self.settings.lob_ws_fallback_urls)
        latest_book: LocalOrderBook | None = None
        reader_task: asyncio.Task | None = None
        snapshot_task: asyncio.Task | None = None
        cancelled = False
        try:
            while stop_at is None or time.monotonic() < stop_at:
                base_url = base_urls[retry % len(base_urls)]
                url = self._stream_url(base_url, symbol)
                pending: list[dict[str, Any]] = []
                pending_snapshot: dict[str, Any] | None = None
                snapshot_task: asyncio.Task | None = None
                reader_task: asyncio.Task | None = None
                queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20000)
                connected_at = time.monotonic()
                try:
                    async with websockets.connect(
                        url, open_timeout=self.settings.request_timeout_seconds,
                        close_timeout=2, ping_interval=20, ping_timeout=20,
                        max_queue=4096,
                    ) as websocket:
                        logger.info("Connected to Binance Spot streams for %s via %s", symbol, base_url)
                        if counters["websocket_messages"] > 0:
                            counters["reconnects"] += 1
                            audit("websocket_reconnected", endpoint=base_url, reconnect_count=counters["reconnects"])
                        else:
                            audit("websocket_connected", endpoint=base_url)
                        reader_task = asyncio.create_task(self._reader(websocket, queue))
                        latest_book = None
                        while stop_at is None or time.monotonic() < stop_at:
                            if snapshot_task is not None and snapshot_task.done():
                                try:
                                    snapshot = snapshot_task.result()
                                except Exception as exc:
                                    logger.warning("Depth snapshot failed for %s: %s", symbol, exc)
                                    audit("snapshot_request_failed", error=str(exc))
                                    snapshot_task = None
                                    await asyncio.sleep(0.5)
                                else:
                                    counters["snapshot_requests"] += 1
                                    snapshot_received_ms = int(time.time() * 1000)
                                    audit(
                                        "snapshot_received", last_update_id=snapshot.get("lastUpdateId"),
                                        buffered_event_count=len(pending),
                                    )
                                    snapshot_writer.write({
                                        "symbol": symbol,
                                        "receive_time_ms": snapshot_received_ms,
                                        "received_at_utc": _utc_iso(snapshot_received_ms),
                                        "payload": snapshot,
                                    })
                                    pending_snapshot = snapshot
                                    candidate, bootstrap_status = _rebuild_from_buffer(symbol, snapshot, pending)
                                    if candidate is not None:
                                        latest_book = candidate
                                        pending_times = [int(item["event_time_ms"]) for item in pending]
                                        checkpoint_time = max(pending_times, default=candidate.event_time_ms or snapshot_received_ms)
                                        # During a bootstrap/resync the buffered records are archived first;
                                        # the full reconstructed checkpoint then anchors future replay.
                                        for item in sorted(pending, key=lambda item: (int(item["event_time_ms"]), item.get("kind") == "depth_update")):
                                            event_writer.write(item)
                                        event_writer.write(_snapshot_checkpoint(candidate, checkpoint_time, snapshot_received_ms))
                                        counters["snapshot_bootstraps" if counters["snapshot_bootstraps"] == 0 else "snapshot_recoveries"] += 1
                                        audit(
                                            "book_synchronized", last_update_id=candidate.last_update_id,
                                            recovery_count=counters["snapshot_recoveries"],
                                        )
                                        logger.info("Synchronized %s order book at update %s", symbol, candidate.last_update_id)
                                        pending.clear()
                                        pending_snapshot = None
                                        snapshot_task = None
                                    else:
                                        audit(
                                            "snapshot_alignment_pending", status=bootstrap_status,
                                            last_update_id=snapshot.get("lastUpdateId"),
                                            buffered_event_count=len(pending),
                                        )
                                        if bootstrap_status in {"old", "gap"}:
                                            # The REST snapshot predates the stream buffer; request a newer one.
                                            snapshot_task = asyncio.create_task(self._fetch_snapshot(symbol))
                                        else:
                                            snapshot_task = None

                            try:
                                raw_message = await asyncio.wait_for(queue.get(), timeout=0.2)
                            except asyncio.TimeoutError:
                                if reader_task.done():
                                    exception = reader_task.exception()
                                    if exception:
                                        raise exception
                                    break
                                continue
                            received_ms = int(time.time() * 1000)
                            try:
                                message = json.loads(raw_message)
                            except json.JSONDecodeError:
                                counters["invalid_book_events"] += 1
                                continue
                            counters["websocket_messages"] += 1
                            websocket_writer.write({
                                "receive_time_ms": received_ms,
                                "received_at_utc": _utc_iso(received_ms),
                                "payload": message,
                            })
                            event = _event_record(message, received_ms)
                            if event is None or event.get("symbol") not in {"", symbol}:
                                continue
                            if event["kind"] == "depth_update":
                                counters["depth_events"] += 1
                            else:
                                counters["trade_events"] += 1

                            if latest_book is None:
                                pending.append(event)
                                if len(pending) > 100000:
                                    raise RuntimeError("Bootstrap buffer exceeded 100,000 events; reconnecting to avoid unbounded memory use")
                                if event["kind"] == "depth_update":
                                    if pending_snapshot is not None:
                                        latest_book, bootstrap_status = _rebuild_from_buffer(symbol, pending_snapshot, pending)
                                        if latest_book is not None:
                                            received = int(time.time() * 1000)
                                            checkpoint_time = max(int(item["event_time_ms"]) for item in pending)
                                            for item in sorted(pending, key=lambda item: (int(item["event_time_ms"]), item.get("kind") == "depth_update")):
                                                event_writer.write(item)
                                            event_writer.write(_snapshot_checkpoint(latest_book, checkpoint_time, received))
                                            counters["snapshot_bootstraps" if counters["snapshot_bootstraps"] == 0 else "snapshot_recoveries"] += 1
                                            audit(
                                                "book_synchronized", last_update_id=latest_book.last_update_id,
                                                recovery_count=counters["snapshot_recoveries"],
                                            )
                                            pending.clear()
                                            pending_snapshot = None
                                        elif bootstrap_status in {"old", "gap"} and snapshot_task is None:
                                            snapshot_task = asyncio.create_task(self._fetch_snapshot(symbol))
                                    if snapshot_task is None and pending_snapshot is None:
                                        snapshot_task = asyncio.create_task(self._fetch_snapshot(symbol))
                                continue

                            checkpoint = _snapshot_checkpoint(latest_book, int(event["event_time_ms"]), received_ms)
                            if event["kind"] == "trade":
                                event_writer.write(event, checkpoint=checkpoint)
                                continue
                            try:
                                latest_book.apply_update(event)
                                latest_book.received_at_utc = event["received_at_utc"]
                            except SequenceGap:
                                counters["sequence_gaps"] += 1
                                audit(
                                    "sequence_gap", last_update_id=latest_book.last_update_id,
                                    first_update_id=event.get("U"), final_update_id=event.get("u"),
                                )
                                event_writer.write(event, checkpoint=checkpoint)
                                latest_book = None
                                pending = []
                                snapshot_task = asyncio.create_task(self._fetch_snapshot(symbol))
                                continue
                            except (InvalidBook, KeyError, TypeError, ValueError) as exc:
                                counters["invalid_book_events"] += 1
                                logger.warning("Invalid book update for %s: %s", symbol, exc)
                                audit("invalid_book_update", error=str(exc), update_id=event.get("u"))
                                event_writer.write(event, checkpoint=checkpoint)
                                latest_book = None
                                pending = []
                                snapshot_task = asyncio.create_task(self._fetch_snapshot(symbol))
                                continue
                            event_writer.write(event, checkpoint=checkpoint)
                            if counters["websocket_messages"] % 10000 == 0:
                                logger.info("%s: %s depth updates, %s trades", symbol, counters["depth_events"], counters["trade_events"])

                    if stop_at is not None and time.monotonic() >= stop_at:
                        break
                    if time.monotonic() - connected_at > 1:
                        retry = 0
                    retry += 1
                    wait = min(self.reconnect_max_seconds, float(2 ** min(retry, 5)))
                    logger.warning("Stream for %s closed; reconnecting in %.1fs", symbol, wait)
                    audit("websocket_closed", retry_in_seconds=wait)
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if reader_task is not None and not reader_task.done():
                        reader_task.cancel()
                        await asyncio.gather(reader_task, return_exceptions=True)
                    if snapshot_task is not None and not snapshot_task.done():
                        snapshot_task.cancel()
                        await asyncio.gather(snapshot_task, return_exceptions=True)
                    latest_book = None
                    retry += 1
                    wait = min(self.reconnect_max_seconds, float(2 ** min(retry, 5)))
                    logger.warning("Stream error for %s via %s: %s; reconnecting in %.1fs", symbol, base_url, exc, wait)
                    audit("websocket_error", endpoint=base_url, error=str(exc), retry_in_seconds=wait)
                    if stop_at is None or time.monotonic() + wait < stop_at:
                        await asyncio.sleep(wait)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            for task in (reader_task, snapshot_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (reader_task, snapshot_task) if task is not None), return_exceptions=True)
            audit("capture_interrupted" if cancelled else "capture_stopped")
            event_files = event_writer.file_metadata()
            websocket_files = websocket_writer.file_metadata()
            snapshot_files = snapshot_writer.file_metadata()
            audit_files = audit_writer.file_metadata()
        end_ms = int(time.time() * 1000)
        manifest = {
            "schema_version": "binance_lob_capture.v1",
            "capture_id": self.capture_id,
            "status": "interrupted" if cancelled else "complete",
            "source": "Binance Spot public WebSocket and REST market data",
            "symbol": symbol,
            "started_at_utc": _utc_iso(start_ms),
            "ended_at_utc": _utc_iso(end_ms),
            "websocket_streams": [f"{symbol.lower()}@depth@{self.settings.lob_update_speed or 'default'}", f"{symbol.lower()}@aggTrade"],
            "depth_snapshot_limit": self.settings.lob_depth_snapshot_limit,
            "sampling_interval_seconds": self.settings.lob_sampling_interval_seconds,
            "exchange_metadata": exchange_metadata,
            "counts": counters,
            "files": {
                "replay_events": event_files, "raw_websocket": websocket_files,
                "rest_snapshots": snapshot_files, "collector_logs": audit_files,
            },
            "interim_event_root": str(self.interim_capture_root / "events"),
            "historical_limit": "Live collection starts at capture time; it does not backfill historical L2 order-book events.",
            "book_limit": "The REST snapshot and diff-depth stream expose price-level changes, not order-ID add/cancel events.",
        }
        manifest_path = self.capture_root / "manifests" / f"symbol={symbol}" / "capture.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest_path)
        if cancelled:
            raise asyncio.CancelledError
        return manifest

    async def run(self, symbols: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
        if not self.settings.lob_collection_enabled:
            raise RuntimeError("Binance collection is disabled in config/settings.yaml")
        loop = asyncio.get_running_loop()
        stop_at = loop.time() + self.duration_seconds if self.duration_seconds > 0 else None
        tasks = [asyncio.create_task(self.collect_symbol(symbol, stop_at)) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return results
