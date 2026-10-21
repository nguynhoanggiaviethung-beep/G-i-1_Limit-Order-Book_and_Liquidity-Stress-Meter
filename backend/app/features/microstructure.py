"""Point-in-time LOB feature calculations used by replay and classification."""

from __future__ import annotations

import json
from typing import Any


def _levels(row: dict[str, Any], side: str, depth: int = 20) -> list[tuple[float, float]]:
    direct = row.get("bids" if side == "bid" else "asks")
    if direct:
        try:
            return [(float(price), float(qty)) for price, qty in direct[:depth]]
        except (TypeError, ValueError):
            pass
    encoded = row.get(f"{side}_levels_top_20_json")
    if encoded:
        try:
            values = json.loads(encoded) if isinstance(encoded, str) else encoded
            return [(float(price), float(qty)) for price, qty in values[:depth]]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    output = []
    for level in range(1, depth + 1):
        price = row.get(f"{side}_price_{level}")
        quantity = row.get(f"{side}_qty_{level}")
        if price is None or quantity is None:
            break
        try:
            price_value, quantity_value = float(price), float(quantity)
        except (TypeError, ValueError):
            break
        if price_value > 0 and quantity_value >= 0:
            output.append((price_value, quantity_value))
    return output


def _impact_bps(levels: list[tuple[float, float]], notional: float, mid: float) -> float | None:
    remaining = float(notional)
    spent = 0.0
    quantity = 0.0
    for price, available in levels:
        level_notional = price * available
        take_notional = min(remaining, level_notional)
        if price <= 0 or take_notional <= 0:
            continue
        take_quantity = take_notional / price
        spent += take_quantity * price
        quantity += take_quantity
        remaining -= take_notional
        if remaining <= 1e-8:
            break
    if remaining > 1e-8 or quantity <= 0:
        return None
    vwap = spent / quantity
    return (vwap - mid) / mid * 10000.0


def compute_microstructure_features(
    row: dict[str, Any], impact_notional_usdt: float = 100000.0, tick_size: float | None = None
) -> dict[str, float | None]:
    """Derive spread, depth, imbalance, microprice, and fixed-notional impact."""
    bids, asks = _levels(row, "bid"), _levels(row, "ask")
    if not bids or not asks:
        result = {"relative_spread_bps": None, "mid_price": None, "microprice_minus_mid_bps": None}
        result.update({f"obi_qty_{depth}": None for depth in (1, 5, 10, 20)})
        return result
    best_bid, bid_qty = bids[0]
    best_ask, ask_qty = asks[0]
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    denominator = bid_qty + ask_qty
    microprice = (best_ask * bid_qty + best_bid * ask_qty) / denominator if denominator > 0 else mid
    result: dict[str, float | None] = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "absolute_spread": spread,
        "relative_spread_bps": spread / mid * 10000.0 if mid else None,
        "spread_ticks": spread / tick_size if tick_size and tick_size > 0 else None,
        "microprice": microprice,
        "microprice_minus_mid_bps": (microprice - mid) / mid * 10000.0 if mid else None,
    }
    for depth in (1, 5, 10, 20):
        bid_depth = sum(quantity for _, quantity in bids[:depth])
        ask_depth = sum(quantity for _, quantity in asks[:depth])
        total = bid_depth + ask_depth
        result[f"bid_depth_qty_{depth}"] = bid_depth
        result[f"ask_depth_qty_{depth}"] = ask_depth
        result[f"depth_qty_{depth}"] = total
        result[f"bid_depth_notional_{depth}"] = sum(price * quantity for price, quantity in bids[:depth])
        result[f"ask_depth_notional_{depth}"] = sum(price * quantity for price, quantity in asks[:depth])
        result[f"depth_notional_{depth}"] = result[f"bid_depth_notional_{depth}"] + result[f"ask_depth_notional_{depth}"]
        result[f"obi_qty_{depth}"] = (bid_depth - ask_depth) / total if total else 0.0
    buy_impact = _impact_bps(asks, impact_notional_usdt, mid)
    sell_impact = _impact_bps(bids, impact_notional_usdt, mid)
    result["impact_buy_bps"] = buy_impact
    result["impact_sell_bps"] = -sell_impact if sell_impact is not None else None
    result["price_impact_bps"] = (
        max(buy_impact, result["impact_sell_bps"])
        if buy_impact is not None and result["impact_sell_bps"] is not None
        else None
    )
    return result

