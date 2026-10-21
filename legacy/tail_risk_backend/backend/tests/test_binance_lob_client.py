import pytest

from app.ingestion.binance_rest import BinanceDepthClient
from app.ingestion.binance_ws import depth_stream_url


def test_public_depth_client_requests_only_configured_snapshot_levels():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"lastUpdateId": 9, "bids": [["1", "2"]], "asks": [["2", "3"]]}

    class Session:
        def get(self, url, params, timeout):
            assert url == "https://api.binance.com/api/v3/depth"
            assert params == {"symbol": "BTCUSDT", "limit": 5000}
            assert timeout == 12
            return Response()

    snapshot = BinanceDepthClient("https://api.binance.com", 12, session=Session()).fetch_snapshot("btcusdt", 5000)
    assert snapshot["lastUpdateId"] == 9


def test_depth_snapshot_limit_obeys_documented_upper_bound():
    with pytest.raises(ValueError, match="between 1 and 5000"):
        BinanceDepthClient("https://api.binance.com").fetch_snapshot("BTCUSDT", 5001)


def test_combined_websocket_url_is_diff_depth_and_configurable_speed():
    assert depth_stream_url("wss://stream.binance.com:9443", "BTCUSDT", "100ms") == (
        "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    )
