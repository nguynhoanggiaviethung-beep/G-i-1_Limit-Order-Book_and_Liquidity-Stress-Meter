from math import isclose

import pandas as pd

from app.features.classification_features import build_classification_features
from app.features.lob_book_features import compute_book_features
from app.features.microstructure import compute_microstructure_features
from app.orderbook.book_state import LocalOrderBook


def test_top_of_book_depth_imbalance_and_microprice_match_hand_calculation():
    book = LocalOrderBook.from_snapshot("BTCUSDT", {
        "lastUpdateId": 8,
        "bids": [["100", "2"], ["99", "1"]],
        "asks": [["102", "1"], ["103", "1"]],
    })
    result = compute_book_features(book)
    assert result["mid_price"] == 101
    assert result["absolute_spread"] == 2
    assert isclose(result["relative_spread_bps"], 2 / 101 * 10000)
    assert result["bid_depth_qty_5"] == 3
    assert result["ask_depth_qty_5"] == 2
    assert result["obi_qty_5"] == 0.2
    assert result["microprice"] == (102 * 2 + 100 * 1) / 3


def test_invalid_book_does_not_produce_features():
    book = LocalOrderBook.from_snapshot("BTCUSDT", {
        "lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]
    })
    book.invalidate("sequence_gap")
    try:
        compute_book_features(book)
    except ValueError as error:
        assert "invalid" in str(error)
    else:
        raise AssertionError("Invalid books must not emit features")


def test_book_microstructure_and_vector_features_share_the_same_formulas():
    bids = [[str(100 - level), str(3 + level / 10)] for level in range(20)]
    asks = [[str(102 + level), str(2 + level / 10)] for level in range(20)]
    book = LocalOrderBook.from_snapshot("BTCUSDT", {"lastUpdateId": 8, "bids": bids, "asks": asks})
    row = {"symbol": "BTCUSDT", "timestamp_utc": "2025-01-01T00:00:00Z", "book_valid": True, "tick_size": 2}
    for side, levels in (("bid", bids), ("ask", asks)):
        for index, (price, quantity) in enumerate(levels, start=1):
            row[f"{side}_price_{index}"] = price
            row[f"{side}_qty_{index}"] = quantity

    canonical = compute_book_features(book, tick_size=2)
    row_features = compute_microstructure_features(row, impact_notional_usdt=5000, tick_size=2)
    vector = build_classification_features(pd.DataFrame([row]), impact_notional_usdt=5000, tick_size=2).iloc[0]
    for key in ("absolute_spread", "relative_spread_bps", "spread_ticks", "mid_price", "microprice", "microprice_minus_mid_bps", "obi_qty_5", "obi_qty_10", "obi_qty_20"):
        assert isclose(float(canonical[key]), float(row_features[key]))
        assert isclose(float(canonical[key]), float(vector[key]))
    assert isclose(float(row_features["price_impact_bps"]), float(vector["price_impact_bps"]))
    assert "bid_top_level_share_20" in canonical
    assert "bid_depth_concentration_20" not in canonical

    # Mid 101; level-20 bid 81 and ask 121 are both 20/101 of mid away.
    distance_bps = 20 / 101 * 10000
    bid_notional = sum(float(p) * float(q) for p, q in bids)
    ask_notional = sum(float(p) * float(q) for p, q in asks)
    assert isclose(float(vector["bid_depth_slope_20"]), bid_notional / distance_bps)
    assert isclose(float(vector["ask_depth_slope_20"]), ask_notional / distance_bps)
    assert isclose(float(vector["depth_slope_imbalance_20"]), (bid_notional - ask_notional) / (bid_notional + ask_notional))
