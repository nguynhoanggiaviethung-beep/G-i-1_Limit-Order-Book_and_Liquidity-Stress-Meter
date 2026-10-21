"""Snapshot/diff-depth alignment following Binance's published procedure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.orderbook.book_state import LocalOrderBook


@dataclass
class BootstrapResult:
    book: LocalOrderBook | None
    remaining_events: list[dict[str, Any]]
    snapshot_too_old: bool
    waiting_for_event: bool


def bootstrap_book(
    symbol: str,
    snapshot: dict[str, Any],
    buffered_events: Iterable[dict[str, Any]],
) -> BootstrapResult:
    """Install a snapshot once the buffered stream spans its update ID.

    If the snapshot predates the first buffered event, fetch a newer snapshot.
    Events at or below the snapshot update ID are discarded. The first event
    applied must overlap the snapshot update ID, as required by Binance.
    """
    events = list(buffered_events)
    snapshot_id = int(snapshot["lastUpdateId"])
    candidate_index = next((i for i, event in enumerate(events) if int(event["u"]) > snapshot_id), None)
    if candidate_index is None:
        return BootstrapResult(None, [], False, True)
    first = events[candidate_index]
    if int(first["U"]) > snapshot_id:
        return BootstrapResult(None, events, True, False)
    book = LocalOrderBook.from_snapshot(symbol, snapshot)
    book.apply_update(first)
    remaining = events[candidate_index + 1 :]
    return BootstrapResult(book, remaining, False, False)
