"""Order-book-native instantaneous features; no OHLCV or trade data."""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

import numpy as np

from app.orderbook.book_state import LocalOrderBook


def _depth(levels: Iterable[tuple[Decimal, Decimal]]) -> tuple[float, float]:
    rows = list(levels)
    quantity = sum((q for _, q in rows), Decimal(0))
    notional = sum((p * q for p, q in rows), Decimal(0))
    return float(quantity), float(notional)


def compute_book_features(book: LocalOrderBook, tick_size: float | None = None) -> dict[str, float | int]:
    if not book.valid:
        raise ValueError("Cannot calculate features from an invalid local book")
    bids, asks = book.best_bid, book.best_ask
    bid_price, bid_qty = bids
    ask_price, ask_qty = asks
    mid = (bid_price + ask_price) / Decimal(2)
    spread = ask_price - bid_price
    denominator = bid_qty + ask_qty
    microprice = (
        (ask_price * bid_qty + bid_price * ask_qty) / denominator
        if denominator > 0 else mid
    )
    result: dict[str, float | int] = {
        "best_bid": float(bid_price),
        "best_bid_qty": float(bid_qty),
        "best_ask": float(ask_price),
        "best_ask_qty": float(ask_qty),
        "mid_price": float(mid),
        "absolute_spread": float(spread),
        "relative_spread_bps": float(spread / mid * Decimal(10000)),
        "spread_ticks": float(spread / Decimal(str(tick_size))) if tick_size and tick_size > 0 else float("nan"),
        "microprice": float(microprice),
        "microprice_minus_mid_bps": float((microprice - mid) / mid * Decimal(10000)),
        "bid_level_count": len(book.bids),
        "ask_level_count": len(book.asks),
    }
    for count in (5, 10, 20):
        bid_qty_n, bid_notional_n = _depth(book.top_levels("bid", count))
        ask_qty_n, ask_notional_n = _depth(book.top_levels("ask", count))
        result[f"bid_depth_qty_{count}"] = bid_qty_n
        result[f"ask_depth_qty_{count}"] = ask_qty_n
        result[f"bid_depth_notional_{count}"] = bid_notional_n
        result[f"ask_depth_notional_{count}"] = ask_notional_n
        total = bid_qty_n + ask_qty_n
        result[f"obi_qty_{count}"] = (bid_qty_n - ask_qty_n) / total if total else 0.0
        notional_total = bid_notional_n + ask_notional_n
        result[f"obi_notional_{count}"] = (
            (bid_notional_n - ask_notional_n) / notional_total if notional_total else 0.0
        )
        result[f"bid_top_level_share_{count}"] = (
            float(book.top_levels("bid", min(5, count))[0][1] / Decimal(str(bid_qty_n)))
            if bid_qty_n else 0.0
        )
        result[f"ask_top_level_share_{count}"] = (
            float(book.top_levels("ask", min(5, count))[0][1] / Decimal(str(ask_qty_n)))
            if ask_qty_n else 0.0
        )
    for side_name in ("bid", "ask"):
        levels = book.top_levels(side_name, 20)
        distances = np.asarray(
            [float(abs(price - mid) / mid * Decimal(10000)) for price, _ in levels], dtype=float
        )
        quantities = np.asarray([float(quantity) for _, quantity in levels], dtype=float)
        if len(levels) >= 2 and np.ptp(distances) > 0 and quantities.sum() > 0:
            normalized_quantities = quantities / quantities.sum()
            result[f"depth_profile_slope_{side_name}_20"] = float(
                np.polyfit(distances, normalized_quantities, 1)[0]
            )
        else:
            result[f"depth_profile_slope_{side_name}_20"] = 0.0
    return result
