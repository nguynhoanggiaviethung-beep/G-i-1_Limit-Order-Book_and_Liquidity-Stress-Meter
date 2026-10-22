from app.data.cryptohft_replay import replay_event_records


def test_raw_book_trade_events_replay_to_one_second_grid_with_event_ofi():
    events = [
        {"kind": "snapshot", "event_time_ms": 900, "lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"kind": "depth_update", "E": 1000, "U": 11, "u": 11, "b": [["100", "2"]], "a": []},
        {"kind": "trade", "T": 1500, "p": "101", "q": "0.5", "m": False},
        {"kind": "trade", "event_time_ms": 1750, "price": "102", "quantity": "0.25", "is_buyer_maker": "true"},
        {"kind": "depth_update", "E": 2000, "U": 12, "u": 12, "b": [["100", "3"]], "a": []},
    ]

    rows, qa = replay_event_records(events, "BTCUSDT", tick_size=0.01, levels=1)

    assert len(rows) == 2
    assert rows[0]["timestamp_utc"] == "1970-01-01T00:00:01Z"
    assert rows[0]["ofi_1s"] == 1.0
    assert rows[1]["aggressive_buy_volume_1s"] == 0.5
    assert rows[1]["aggressive_sell_volume_1s"] == 0.25
    assert rows[1]["trade_count_1s"] == 2
    assert rows[1]["last_trade_price"] == "102"
    assert rows[1]["addition_intensity_1s"] == 1.0
    assert rows[0]["book_valid"] is True
    assert qa["valid_seconds"] == qa["expected_seconds"] == 2


def test_sequence_gap_invalidates_samples_until_new_snapshot():
    events = [
        {"kind": "snapshot", "event_time_ms": 900, "lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"kind": "depth_update", "E": 1000, "U": 11, "u": 11, "b": [], "a": []},
        {"kind": "depth_update", "E": 2000, "U": 15, "u": 15, "b": [], "a": []},
        {"kind": "snapshot", "event_time_ms": 3000, "lastUpdateId": 20, "bids": [["100", "1"]], "asks": [["101", "1"]]},
    ]

    rows, qa = replay_event_records(events, "BTCUSDT", tick_size=0.01, levels=1)

    assert [row["timestamp_utc"] for row in rows] == ["1970-01-01T00:00:01Z", "1970-01-01T00:00:03Z"]
    assert qa["sequence_gap_count"] == 1
    assert qa["dropped_seconds"] == 1
    assert qa["snapshot_bootstraps"] == 1
    assert qa["snapshot_recoveries"] == 1


def test_invalid_book_update_is_not_counted_as_sequence_gap():
    events = [
        {"kind": "snapshot", "event_time_ms": 900, "lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"kind": "depth_update", "E": 1000, "U": 11, "u": 11, "b": [["102", "1"]], "a": []},
        {"kind": "snapshot", "event_time_ms": 2000, "lastUpdateId": 20, "bids": [["100", "1"]], "asks": [["101", "1"]]},
    ]

    rows, qa = replay_event_records(events, "BTCUSDT", tick_size=0.01, levels=1)

    assert len(rows) == 1
    assert qa["sequence_gap_count"] == 0
    assert qa["invalid_book_event_count"] == 1
