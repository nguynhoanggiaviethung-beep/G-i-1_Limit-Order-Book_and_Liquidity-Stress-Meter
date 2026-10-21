from math import isclose

from app.features.lob_book_features import compute_book_features
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
