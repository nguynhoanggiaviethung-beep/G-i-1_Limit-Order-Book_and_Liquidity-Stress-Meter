"""Event-level Order Flow Imbalance (OFI) calculations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def event_ofi(previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
    """Cont, Kukanov and Stoikov best-quote OFI contribution for one event."""
    pb0, qb0 = float(previous["best_bid"]), float(previous["best_bid_qty"])
    pa0, qa0 = float(previous["best_ask"]), float(previous["best_ask_qty"])
    pb1, qb1 = float(current["best_bid"]), float(current["best_bid_qty"])
    pa1, qa1 = float(current["best_ask"]), float(current["best_ask_qty"])
    bid = (qb1 if pb1 >= pb0 else 0.0) - (qb0 if pb1 <= pb0 else 0.0)
    ask = (qa1 if pa1 <= pa0 else 0.0) - (qa0 if pa1 >= pa0 else 0.0)
    return bid - ask


def aggregate_ofi_by_window(events: Iterable[Mapping[str, Any]]) -> dict[int, float]:
    """Sum event-level OFI by second; events must already be sequence ordered."""
    totals: dict[int, float] = {}
    previous: Mapping[str, Any] | None = None
    for event in events:
        timestamp = event.get("event_time_ms")
        if timestamp is None:
            continue
        second = int(timestamp) // 1000
        if previous is not None:
            totals[second] = totals.get(second, 0.0) + event_ofi(previous, event)
        previous = event
    return totals

