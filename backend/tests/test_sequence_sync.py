from app.orderbook.synchronizer import bootstrap_book


def _snapshot(update_id=100):
    return {
        "lastUpdateId": update_id,
        "bids": [["100.00", "2.0"], ["99.00", "1.0"]],
        "asks": [["101.00", "3.0"], ["102.00", "1.0"]],
    }


def _event(first, final):
    return {"e": "depthUpdate", "s": "BTCUSDT", "E": 1000, "U": first, "u": final, "b": [], "a": []}


def test_snapshot_discards_old_and_applies_first_event_spanning_snapshot_id():
    result = bootstrap_book("BTCUSDT", _snapshot(), [_event(90, 100), _event(100, 102)])
    assert result.book is not None
    assert result.book.last_update_id == 102
    assert result.remaining_events == []
    assert not result.snapshot_too_old


def test_snapshot_older_than_first_stream_event_requests_new_snapshot():
    result = bootstrap_book("BTCUSDT", _snapshot(100), [_event(101, 102)])
    assert result.book is None
    assert result.snapshot_too_old


def test_no_event_after_snapshot_waits_for_stream():
    result = bootstrap_book("BTCUSDT", _snapshot(100), [_event(90, 100)])
    assert result.waiting_for_event
