"""Exact-decimal state for one Binance Spot local order book."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


class SequenceGap(RuntimeError):
    """Raised when a depth update cannot be applied without missing levels."""


class InvalidBook(RuntimeError):
    """Raised when a reconstructed book fails structural integrity checks."""


def _levels(values: list[list[str]]) -> dict[Decimal, Decimal]:
    result: dict[Decimal, Decimal] = {}
    for price_raw, quantity_raw, *_ in values:
        try:
            price, quantity = Decimal(str(price_raw)), Decimal(str(quantity_raw))
        except (InvalidOperation, ValueError) as exc:
            raise InvalidBook("Book contains a non-numeric price or quantity") from exc
        if not price.is_finite() or not quantity.is_finite() or price <= 0 or quantity < 0:
            raise InvalidBook("Book prices must be positive and quantities non-negative")
        if quantity:
            result[price] = quantity
    return result


@dataclass
class LocalOrderBook:
    """One synchronized book. Price and quantity keys retain Binance precision."""

    symbol: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    last_update_id: int | None = None
    valid: bool = False
    invalid_reason: str | None = "awaiting_snapshot"
    event_time_ms: int | None = None
    transaction_time_ms: int | None = None
    received_at_utc: str | None = None
    _bid_prices_sorted: list[Decimal] | None = field(default=None, init=False, repr=False)
    _ask_prices_sorted: list[Decimal] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_snapshot(cls, symbol: str, snapshot: dict[str, Any]) -> "LocalOrderBook":
        book = cls(
            symbol=symbol.upper(),
            bids=_levels(snapshot.get("bids", [])),
            asks=_levels(snapshot.get("asks", [])),
            last_update_id=int(snapshot["lastUpdateId"]),
            valid=True,
            invalid_reason=None,
        )
        book.validate()
        return book

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.invalid_reason = reason

    def validate(self) -> None:
        if not self.bids or not self.asks:
            self.invalidate("empty_side")
            raise InvalidBook("A valid book must have at least one bid and one ask")
        if self._bid_prices_sorted is None:
            self._bid_prices_sorted = sorted(self.bids)
        if self._ask_prices_sorted is None:
            self._ask_prices_sorted = sorted(self.asks)
        best_bid, best_ask = self._bid_prices_sorted[-1], self._ask_prices_sorted[0]
        if best_bid >= best_ask:
            self.invalidate("crossed_or_locked_book")
            raise InvalidBook("Best bid must be strictly below best ask")

    @property
    def best_bid(self) -> tuple[Decimal, Decimal]:
        if not self.valid or not self.bids:
            raise InvalidBook(self.invalid_reason or "book_invalid")
        if self._bid_prices_sorted is None:
            self._bid_prices_sorted = sorted(self.bids)
        price = self._bid_prices_sorted[-1]
        return price, self.bids[price]

    @property
    def best_ask(self) -> tuple[Decimal, Decimal]:
        if not self.valid or not self.asks:
            raise InvalidBook(self.invalid_reason or "book_invalid")
        if self._ask_prices_sorted is None:
            self._ask_prices_sorted = sorted(self.asks)
        price = self._ask_prices_sorted[0]
        return price, self.asks[price]

    def apply_update(self, event: dict[str, Any]) -> bool:
        """Apply one diff-depth event; return False for an already-applied event."""
        if not self.valid or self.last_update_id is None:
            raise InvalidBook(self.invalid_reason or "book_not_initialized")
        first_id, final_id = int(event["U"]), int(event["u"])
        if final_id <= self.last_update_id:
            return False
        if first_id > self.last_update_id + 1:
            missing_from = self.last_update_id + 1
            self.invalidate(f"sequence_gap:{missing_from}-{first_id - 1}")
            raise SequenceGap(
                f"{self.symbol}: expected update {missing_from}, received range {first_id}-{final_id}"
            )
        try:
            self._apply_side(self.bids, event.get("b", []), self._bid_prices_sorted)
            self._apply_side(self.asks, event.get("a", []), self._ask_prices_sorted)
            self.last_update_id = final_id
            self.event_time_ms = int(event["E"]) if event.get("E") is not None else None
            self.transaction_time_ms = int(event["T"]) if event.get("T") is not None else None
            if event.get("received_at_utc") is not None:
                self.received_at_utc = str(event["received_at_utc"])
            self.validate()
        except (InvalidBook, KeyError, TypeError, ValueError) as exc:
            self.invalidate(str(exc))
            raise
        return True

    @staticmethod
    def _apply_side(
        side: dict[Decimal, Decimal],
        changes: list[list[str]],
        sorted_prices: list[Decimal] | None,
    ) -> None:
        for price_raw, quantity_raw, *_ in changes:
            try:
                price, quantity = Decimal(str(price_raw)), Decimal(str(quantity_raw))
            except (InvalidOperation, ValueError) as exc:
                raise InvalidBook("Depth update contains a non-numeric level") from exc
            if not price.is_finite() or not quantity.is_finite() or price <= 0 or quantity < 0:
                raise InvalidBook("Depth update has invalid price or quantity")
            if quantity == 0:
                if price in side:
                    side.pop(price)
                    if sorted_prices is not None:
                        index = bisect_left(sorted_prices, price)
                        if index >= len(sorted_prices) or sorted_prices[index] != price:
                            raise InvalidBook("Sorted price index lost synchronization with the book")
                        sorted_prices.pop(index)
            else:
                if price not in side and sorted_prices is not None:
                    sorted_prices.insert(bisect_left(sorted_prices, price), price)
                side[price] = quantity

    def top_levels(self, side: str, count: int = 20) -> list[tuple[Decimal, Decimal]]:
        if not self.valid:
            raise InvalidBook(self.invalid_reason or "book_invalid")
        if count < 1:
            return []
        levels = self.bids if side == "bid" else self.asks
        cache_name = "_bid_prices_sorted" if side == "bid" else "_ask_prices_sorted"
        prices = getattr(self, cache_name)
        if prices is None:
            prices = sorted(levels)
            setattr(self, cache_name, prices)
        if side == "bid":
            prices = prices[-count:][::-1]
        else:
            prices = prices[:count]
        return [(price, levels[price]) for price in prices]

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "valid": self.valid,
            "bids": [[str(p), str(q)] for p, q in self.top_levels("bid", len(self.bids))],
            "asks": [[str(p), str(q)] for p, q in self.top_levels("ask", len(self.asks))],
        }
