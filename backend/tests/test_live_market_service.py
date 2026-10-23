from __future__ import annotations

from datetime import datetime, timezone

from app.services.live_market_service import LiveMarketService


class ReplayPredictor:
    def predict_features(self, symbol, timestamp, horizon_seconds, features, quality):
        return {
            "schema": "liquidity_meter.v1",
            "symbol": symbol,
            "timestamp_utc": timestamp,
            "horizon_seconds": horizon_seconds,
            "status": "unavailable",
            "reason": "live model warmup",
            "features_received": bool(features),
            "quality": quality,
        }


def _checkpoint(symbol: str = "ETHUSDT") -> dict:
    now_ms = (int(datetime.now(timezone.utc).timestamp() * 1000) // 1000) * 1000
    return {
        "symbol": symbol,
        "event_time_ms": now_ms,
        "book_event_time_ms": now_ms,
        "received_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lastUpdateId": 100,
        "bids": [[str(2000 - index), "1.25"] for index in range(20)],
        "asks": [[str(2001 + index), "1.50"] for index in range(20)],
    }


def test_live_book_sync_publishes_snapshot_and_meter_uses_live_features():
    service = LiveMarketService(ReplayPredictor(), collector_factory=lambda *_args, **_kwargs: None)
    queue = service.subscribe("ETHUSDT")
    assert queue.get_nowait()["status"] == "starting"

    service._on_book_sync(_checkpoint())
    payload = queue.get_nowait()
    assert payload["status"] == "live"
    assert payload["snapshot"]["symbol"] == "ETHUSDT"
    assert len(payload["snapshot"]["bids"]) == 20
    assert payload["snapshot"]["features"]["mid_price"] == 2000.5

    response = service.for_horizon(payload, 5)
    assert response["meter"]["reason"] == "live model warmup"
    assert response["meter"]["features_received"] is True
    assert "model_features" not in response


def test_live_stream_emits_one_second_book_and_flow_updates():
    service = LiveMarketService(ReplayPredictor(), collector_factory=lambda *_args, **_kwargs: None)
    queue = service.subscribe("ETHUSDT")
    queue.get_nowait()
    checkpoint = _checkpoint()
    service._on_book_sync(checkpoint)
    queue.get_nowait()

    for update_id, offset_ms, quantity in ((101, 1000, "1.50"), (102, 2000, "1.75")):
        event_time = checkpoint["event_time_ms"] + offset_ms
        service._on_market_event({
            "kind": "depth_update",
            "symbol": "ETHUSDT",
            "event_time_ms": event_time,
            "E": event_time,
            "T": event_time,
            "U": update_id,
            "u": update_id,
            "b": [["2000", quantity]],
            "a": [],
            "received_at_utc": datetime.fromtimestamp(event_time / 1000, timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        update = queue.get_nowait()

    assert update["snapshot"]["bids"][0] == ["2000", "1.50"]
    assert update["model_features"]["ofi_1s"] == 0.25
