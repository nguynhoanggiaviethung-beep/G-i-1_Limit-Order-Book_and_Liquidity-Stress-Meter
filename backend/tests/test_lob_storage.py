import hashlib
import json
from decimal import Decimal

from app.orderbook.book_state import LocalOrderBook
from app.storage.parquet_writer import LOBParquetWriter
from app.storage.raw_writer import RawLOBWriter


def test_raw_writer_appends_payload_and_manifest_with_checksum(tmp_path):
    writer = RawLOBWriter(tmp_path, max_file_bytes=4096)
    event = {"e": "depthUpdate", "s": "BTCUSDT", "E": 1760000000000, "U": 10, "u": 11, "b": [], "a": []}
    path = writer.write_depth_event("BTCUSDT", event, "btcusdt@depth@100ms", "abc123456789", "test", "2025-10-09T08:00:00+00:00")
    writer.finish_connection("BTCUSDT", "abc123456789")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["payload"] == event
    manifest = tmp_path / "state" / "lob" / "btcusdt" / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert entry["artifact_type"] == "raw_events_jsonl"
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["rows_in_file"] == 1


def test_parquet_writer_persists_sampled_feature_and_top_book(tmp_path):
    book = LocalOrderBook.from_snapshot("BTCUSDT", {
        "lastUpdateId": 7, "bids": [["100", "2"]], "asks": [["101", "3"]]
    })
    writer = LOBParquetWriter(tmp_path, flush_rows=1)
    row = {
        "symbol": "BTCUSDT", "sample_time_utc": "2025-10-09T08:00:00+00:00",
        "event_time_utc": None, "received_at_utc": "2025-10-09T08:00:00+00:00",
        "book_valid": True, "mid_price": 100.5,
    }
    writer.append(row, book)
    paths = writer.flush()
    assert len(paths) == 2
    import pandas as pd
    features = pd.read_parquet(next(path for path in paths if "features" in str(path)))
    snapshots = pd.read_parquet(next(path for path in paths if "lob_snapshots" in str(path)))
    assert features.loc[0, "mid_price"] == 100.5
    assert json.loads(snapshots.loc[0, "bid_levels_top_20_json"]) == [["100", "2"]]
