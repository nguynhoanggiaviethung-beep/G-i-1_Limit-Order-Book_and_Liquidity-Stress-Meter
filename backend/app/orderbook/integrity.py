"""Book validity and sequence quality helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from app.orderbook.book_state import LocalOrderBook


def event_age_ms(received_at: datetime, event_time_ms: int | None) -> float | None:
    if event_time_ms is None:
        return None
    event_at = datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc)
    return max(0.0, (received_at - event_at).total_seconds() * 1000)


def book_quality(book: LocalOrderBook) -> dict:
    valid = book.valid
    best_bid = book.best_bid[0] if valid else None
    best_ask = book.best_ask[0] if valid else None
    crossed = bool(valid and best_bid >= best_ask)
    return {
        "book_valid": bool(valid and not crossed),
        "quality_flags": [] if valid and not crossed else [book.invalid_reason or "invalid_book"],
        "sequence_id": book.last_update_id,
        "crossed_book": crossed,
        "bid_levels": len(book.bids),
        "ask_levels": len(book.asks),
    }
