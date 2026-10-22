"""Replay normalized raw L2/trade events into a validated one-second book grid."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Iterable, Iterator

from app.features.ofi import event_ofi
from app.orderbook.book_state import InvalidBook, LocalOrderBook, SequenceGap

MAX_EVENT_TIME_REORDER_MS = 100


def _iso_utc(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_event(record: dict[str, Any]) -> dict[str, Any]:
    """Map a native Binance-style event or canonical event record to one schema.

    Canonical records use ``kind`` values ``snapshot``, ``depth_update`` and
    ``trade``. Native ``e``/``E``/``U``/``u``/``b``/``a`` and aggTrade fields
    are accepted directly. Provider-specific CSV/parquet columns should be
    mapped to this small schema before replay.
    """
    payload = record.get("data", record)
    if not isinstance(payload, dict):
        raise ValueError("Each event record must be a JSON object")
    raw_kind = str(payload.get("kind", payload.get("event_type", payload.get("e", "")))).lower()
    if raw_kind in {"snapshot", "depthsnapshot", "depth_snapshot"} or ("lastUpdateId" in payload and "bids" in payload):
        kind = "snapshot"
    elif raw_kind in {"depthupdate", "depth_update", "depth"} or all(name in payload for name in ("U", "u", "b", "a")):
        kind = "depth_update"
    elif (
        raw_kind in {"aggtrade", "trade", "agg_trade"}
        or all(name in payload for name in ("p", "q", "m"))
        or all(name in payload for name in ("price", "quantity", "is_buyer_maker"))
    ):
        kind = "trade"
    else:
        raise ValueError(f"Unsupported raw event kind: {raw_kind or 'missing'}")

    timestamp = payload.get("event_time_ms", payload.get("E", payload.get("T", payload.get("trade_timestamp_ms", payload.get("timestamp_ms")))))
    if timestamp is None:
        raise ValueError("Each snapshot, depth update, and trade needs an exchange event timestamp in milliseconds")
    event = dict(payload)
    event["kind"] = kind
    event["event_time_ms"] = int(timestamp)
    if payload.get("s", payload.get("symbol")) is not None:
        event["symbol"] = str(payload.get("s", payload.get("symbol"))).upper()
    if kind == "depth_update":
        event["E"] = int(payload.get("E", timestamp))
        event["T"] = int(payload["T"]) if payload.get("T") is not None else None
    elif kind == "trade":
        event["p"] = payload.get("p", payload.get("price"))
        event["q"] = payload.get("q", payload.get("quantity"))
        event["m"] = payload.get("m", payload.get("is_buyer_maker"))
        event["trade_timestamp_ms"] = int(payload.get("trade_timestamp_ms", payload.get("T", payload.get("trade_ts", timestamp))))
    return event


class OneSecondBookReplay:
    """Stateful sequence-checked event replay with exact 1-second sampling."""

    def __init__(self, symbol: str, tick_size: float, max_event_age_seconds: int = 2, levels: int = 20) -> None:
        if tick_size <= 0 or max_event_age_seconds < 0 or levels < 1:
            raise ValueError("tick_size and levels must be positive and max_event_age_seconds non-negative")
        self.symbol = symbol.upper()
        self.tick_size = float(tick_size)
        self.max_event_age_ms = int(max_event_age_seconds * 1000)
        self.levels = int(levels)
        self.book: LocalOrderBook | None = None
        self.current_second: int | None = None
        self.last_input_timestamp_ms: int | None = None
        self.last_gap_second: int | None = None
        self.recovered_second: int | None = None
        self._snapshot_seen = False
        self._reset_bucket()
        self.qa = {
            "expected_seconds": 0, "valid_seconds": 0, "dropped_seconds": 0,
            "sequence_gap_count": 0, "invalid_book_event_count": 0,
            "snapshot_bootstraps": 0, "snapshot_recoveries": 0,
            "event_time_order_adjustments": 0, "max_event_time_lateness_ms": 0,
        }
        self.daily_qa: dict[str, dict[str, int]] = {}

    def _daily_counter(self, second: int | None = None, timestamp_ms: int | None = None) -> dict[str, int]:
        if timestamp_ms is None:
            timestamp_ms = int(second or 0) * 1000
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()
        return self.daily_qa.setdefault(day, {
            "expected_seconds": 0, "valid_seconds": 0, "dropped_seconds": 0,
            "sequence_gap_count": 0, "invalid_book_event_count": 0,
            "snapshot_bootstraps": 0, "snapshot_recoveries": 0,
            "event_time_order_adjustments": 0, "max_event_time_lateness_ms": 0,
        })

    def _reset_bucket(self) -> None:
        self.ofi_sum = 0.0
        self.addition_intensity = 0.0
        self.cancellation_intensity = 0.0
        self.aggressive_buy_volume = 0.0
        self.aggressive_sell_volume = 0.0
        self.trade_count = 0
        self.last_trade_price: str | None = None
        self.last_trade_time_ms: int | None = None
        self.first_update_id: int | None = None
        self.last_update_id: int | None = None

    def feed(self, raw_record: dict[str, Any]) -> Iterator[dict[str, Any]]:
        event = normalize_event(raw_record)
        if event.get("symbol") and event["symbol"] != self.symbol:
            raise ValueError(f"Event symbol {event['symbol']} does not match replay symbol {self.symbol}")
        timestamp = event["event_time_ms"]
        if self.last_input_timestamp_ms is not None and timestamp < self.last_input_timestamp_ms:
            lateness_ms = self.last_input_timestamp_ms - timestamp
            if lateness_ms > MAX_EVENT_TIME_REORDER_MS:
                raise ValueError(
                    f"Raw event time moved backwards by {lateness_ms} ms, exceeding the "
                    f"{MAX_EVENT_TIME_REORDER_MS} ms cross-stream tolerance"
                )
            self.qa["event_time_order_adjustments"] += 1
            self.qa["max_event_time_lateness_ms"] = max(self.qa["max_event_time_lateness_ms"], lateness_ms)
            daily_qa = self._daily_counter(timestamp_ms=timestamp)
            daily_qa["event_time_order_adjustments"] += 1
            daily_qa["max_event_time_lateness_ms"] = max(daily_qa["max_event_time_lateness_ms"], lateness_ms)
            # Depth sequence is applied in stream order. Clamp only the replay
            # bucket clock; preserve the original exchange timestamp on event.
            timestamp = self.last_input_timestamp_ms
        self.last_input_timestamp_ms = timestamp
        # Exchange events are accumulated in (t-1s, t] and sampled on the
        # next UTC second boundary. An event exactly at a boundary belongs to it.
        second = (timestamp + 999) // 1000
        if self.current_second is None:
            self.current_second = second
        elif second < self.current_second:
            raise ValueError("Event time moved behind the replay sampling cursor")
        elif second > self.current_second:
            for completed_second in range(self.current_second, second):
                row = self._sample(completed_second)
                if row is not None:
                    yield row
                self._reset_bucket()
            self.current_second = second

        if event["kind"] == "snapshot":
            self._install_snapshot(event)
        elif event["kind"] == "depth_update":
            self._apply_depth_update(event)
        else:
            self._apply_trade(event)

    def finish(self) -> Iterator[dict[str, Any]]:
        if self.current_second is not None:
            row = self._sample(self.current_second)
            if row is not None:
                yield row
            self.current_second = None

    def _install_snapshot(self, event: dict[str, Any]) -> None:
        was_invalid = self.book is None or not self.book.valid
        payload = {
            "lastUpdateId": event["lastUpdateId"],
            "bids": event.get("bids", event.get("b", [])),
            "asks": event.get("asks", event.get("a", [])),
        }
        try:
            self.book = LocalOrderBook.from_snapshot(self.symbol, payload)
        except (KeyError, TypeError, ValueError, InvalidBook) as exc:
            self.book = None
            raise ValueError(f"Invalid order-book snapshot: {exc}") from exc
        self.book.event_time_ms = int(event.get("book_event_time_ms", event["event_time_ms"]))
        self.book.received_at_utc = event.get("received_at_utc")
        if not self._snapshot_seen:
            self.qa["snapshot_bootstraps"] += 1
            self._daily_counter(timestamp_ms=event["event_time_ms"])["snapshot_bootstraps"] += 1
            self._snapshot_seen = True
        elif was_invalid:
            self.qa["snapshot_recoveries"] += 1
            self._daily_counter(timestamp_ms=event["event_time_ms"])["snapshot_recoveries"] += 1
            self.recovered_second = self.current_second

    @staticmethod
    def _change_intensity(book: LocalOrderBook, changes: list[list[Any]], side: str) -> tuple[float, float]:
        levels = book.bids if side == "bid" else book.asks
        additions = deletions = 0.0
        for price_raw, quantity_raw, *_ in changes:
            from decimal import Decimal, InvalidOperation

            try:
                price = Decimal(str(price_raw))
                current = Decimal(str(quantity_raw))
            except (InvalidOperation, ValueError) as exc:
                raise ValueError("Depth update contains a non-numeric level") from exc
            previous = levels.get(price, Decimal(0))
            if not price.is_finite() or not current.is_finite() or price <= 0 or current < 0:
                raise ValueError("Depth update contains a non-finite or invalid level")
            difference = current - previous
            if difference > 0:
                additions += float(difference)
            elif difference < 0:
                # Public diff-depth deletions mix cancels and executions; this
                # field is a visible-book depletion proxy, not order-ID truth.
                deletions += float(-difference)
        return additions, deletions

    def _apply_depth_update(self, event: dict[str, Any]) -> None:
        if self.book is None or not self.book.valid:
            return
        try:
            previous = {
                "best_bid": str(self.book.best_bid[0]),
                "best_bid_qty": str(self.book.best_bid[1]),
                "best_ask": str(self.book.best_ask[0]),
                "best_ask_qty": str(self.book.best_ask[1]),
            }
            update = {"U": event["U"], "u": event["u"], "b": event.get("b", []), "a": event.get("a", []), "E": event.get("E"), "T": event.get("T")}
            bid_add, bid_delete = self._change_intensity(self.book, update["b"], "bid")
            ask_add, ask_delete = self._change_intensity(self.book, update["a"], "ask")
            applied = self.book.apply_update(update)
            if not applied:
                return
            current = {
                "best_bid": str(self.book.best_bid[0]),
                "best_bid_qty": str(self.book.best_bid[1]),
                "best_ask": str(self.book.best_ask[0]),
                "best_ask_qty": str(self.book.best_ask[1]),
            }
            self.ofi_sum += event_ofi(previous, current)
            self.addition_intensity += bid_add + ask_add
            self.cancellation_intensity += bid_delete + ask_delete
            self.first_update_id = int(update["U"]) if self.first_update_id is None else min(self.first_update_id, int(update["U"]))
            self.last_update_id = int(update["u"]) if self.last_update_id is None else max(self.last_update_id, int(update["u"]))
        except SequenceGap:
            self.qa["sequence_gap_count"] += 1
            self._daily_counter(self.current_second)["sequence_gap_count"] += 1
            self.last_gap_second = self.current_second
            self.book = None
        except (InvalidBook, KeyError, TypeError, ValueError):
            self.qa["invalid_book_event_count"] += 1
            self._daily_counter(self.current_second)["invalid_book_event_count"] += 1
            self.book = None

    def _apply_trade(self, event: dict[str, Any]) -> None:
        try:
            quantity = float(event["q"])
            price = float(event["p"])
            if not isfinite(quantity) or not isfinite(price) or quantity <= 0 or price <= 0:
                return
            # Binance buyer-maker true means the buyer was passive, so the
            # aggressive trade was a sell.
            buyer_maker = event["m"]
            if isinstance(buyer_maker, str):
                normalized = buyer_maker.strip().lower()
                if normalized not in {"true", "1", "yes", "false", "0", "no"}:
                    return
                buyer_maker = normalized in {"true", "1", "yes"}
            elif not isinstance(buyer_maker, bool):
                return
            if bool(buyer_maker):
                self.aggressive_sell_volume += quantity
            else:
                self.aggressive_buy_volume += quantity
            self.last_trade_price = str(event["p"])
            self.last_trade_time_ms = int(event.get("trade_timestamp_ms", event["event_time_ms"]))
            self.trade_count += 1
        except (KeyError, TypeError, ValueError):
            return

    def _sample(self, second: int) -> dict[str, Any] | None:
        self.qa["expected_seconds"] += 1
        daily_qa = self._daily_counter(second)
        daily_qa["expected_seconds"] += 1
        sample_time_ms = int(second * 1000)
        if self.book is None or not self.book.valid or self.book.event_time_ms is None:
            self.qa["dropped_seconds"] += 1
            daily_qa["dropped_seconds"] += 1
            return None
        age_ms = sample_time_ms - self.book.event_time_ms
        if age_ms < 0 or age_ms > self.max_event_age_ms:
            self.qa["dropped_seconds"] += 1
            daily_qa["dropped_seconds"] += 1
            return None
        try:
            bid_levels = self.book.top_levels("bid", self.levels)
            ask_levels = self.book.top_levels("ask", self.levels)
        except InvalidBook:
            self.qa["dropped_seconds"] += 1
            daily_qa["dropped_seconds"] += 1
            return None
        row: dict[str, Any] = {
            "symbol": self.symbol,
            "timestamp_utc": _iso_utc(sample_time_ms),
            "event_time_utc": _iso_utc(self.book.event_time_ms),
            "received_at_utc": self.book.received_at_utc,
            "first_update_id": self.first_update_id,
            "last_update_id": self.last_update_id or self.book.last_update_id,
            "book_valid": True,
            "sequence_gap": self.last_gap_second == second,
            "seconds_since_gap": second - self.recovered_second if self.recovered_second is not None else None,
            "ofi_1s": self.ofi_sum,
            "aggressive_buy_volume_1s": self.aggressive_buy_volume,
            "aggressive_sell_volume_1s": self.aggressive_sell_volume,
            "trade_count_1s": self.trade_count,
            "last_trade_price": self.last_trade_price,
            "last_trade_time_utc": _iso_utc(self.last_trade_time_ms),
            "addition_intensity_1s": self.addition_intensity,
            "cancellation_intensity_1s": self.cancellation_intensity,
            "tick_size": self.tick_size,
            "qa_flags": [],
        }
        for side, levels in (("bid", bid_levels), ("ask", ask_levels)):
            for index in range(self.levels):
                price, quantity = levels[index] if index < len(levels) else (None, None)
                row[f"{side}_price_{index + 1}"] = str(price) if price is not None else None
                row[f"{side}_qty_{index + 1}"] = str(quantity) if quantity is not None else None
        self.qa["valid_seconds"] += 1
        daily_qa["valid_seconds"] += 1
        return row


def replay_event_records(
    records: Iterable[dict[str, Any]],
    symbol: str,
    tick_size: float,
    max_event_age_seconds: int = 2,
    levels: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convenience wrapper for small archives and deterministic unit tests."""
    replay = OneSecondBookReplay(symbol, tick_size, max_event_age_seconds, levels)
    output: list[dict[str, Any]] = []
    for record in records:
        output.extend(replay.feed(record))
    output.extend(replay.finish())
    return output, dict(replay.qa)
