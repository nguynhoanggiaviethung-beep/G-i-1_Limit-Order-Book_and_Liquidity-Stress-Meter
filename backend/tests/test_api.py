from types import SimpleNamespace

from starlette.requests import Request

from app.api.routes.health import health
from app.api.routes.lob import get_latest_features, get_latest_lob
from app.main import app


def test_health_and_lob_read_only_routes_are_registered():
    paths = app.openapi()["paths"]
    assert app.title == "LOB-Only Crypto Risk Research API"
    assert paths["/health"]["get"]
    assert paths["/api/v1/collector/status"]["get"]
    assert paths["/api/v1/lob/latest"]["get"]
    assert paths["/api/v1/features/latest"]["get"]
    assert "/api/v1/forecasts" not in paths


def test_health_is_read_only_and_reports_not_started():
    scope = {"type": "http", "method": "GET", "path": "/health", "headers": [], "query_string": b"", "app": app}
    assert health(Request(scope)) == {"status": "ok", "collector_status": {"status": "not_started"}}


def test_latest_book_and_features_return_lob_only_data():
    snapshot = {"symbol": "BTCUSDT", "book_valid": True, "bids": [["100", "2"]], "asks": [["101", "3"]]}
    feature = {"symbol": "BTCUSDT", "book_valid": True, "mid_price": 100.5, "obi_qty_5": -0.2}
    service = SimpleNamespace(
        get_status=lambda: {"symbol_status": {"BTCUSDT": {"book_valid": True}}},
        get_latest_book=lambda _symbol: snapshot,
        get_latest_features=lambda _symbol: feature,
    )
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b"", "app": app}
    request = Request(scope)
    app.state.lob_collector = service
    try:
        assert get_latest_lob(request, "BTCUSDT") == snapshot
        assert get_latest_features(request, "BTCUSDT") == feature
    finally:
        del app.state.lob_collector
