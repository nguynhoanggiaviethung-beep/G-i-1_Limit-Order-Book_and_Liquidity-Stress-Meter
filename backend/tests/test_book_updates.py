import pytest

from app.orderbook.book_state import InvalidBook, LocalOrderBook, SequenceGap


def _book():
    return LocalOrderBook.from_snapshot("BTCUSDT", {
        "lastUpdateId": 100,
        "bids": [["100.00000001", "2.5"], ["99", "1"]],
        "asks": [["101", "3"], ["102", "1"]],
    })


def test_depth_update_replaces_levels_and_zero_quantity_deletes():
    book = _book()
    applied = book.apply_update({"U": 101, "u": 102, "E": 1000, "b": [["100.00000001", "0"], ["98", "4"]], "a": [["101", "0"]]})
    assert applied
    assert book.last_update_id == 102
    assert float(book.best_bid[0]) == 99.0
    assert "100.00000001" not in {str(price) for price in book.bids}
    assert float(book.best_ask[0]) == 102.0
    assert book.valid


def test_old_event_is_ignored():
    book = _book()
    assert not book.apply_update({"U": 99, "u": 100, "b": [], "a": []})
    assert book.last_update_id == 100


def test_sequence_gap_invalidates_book_and_raises():
    book = _book()
    with pytest.raises(SequenceGap):
        book.apply_update({"U": 102, "u": 103, "b": [], "a": []})
    assert not book.valid
    assert "sequence_gap" in book.invalid_reason


def test_crossed_book_is_not_published():
    book = _book()
    with pytest.raises(InvalidBook):
        book.apply_update({"U": 101, "u": 101, "b": [["102", "4"]], "a": [], "E": 1000})
    assert not book.valid
