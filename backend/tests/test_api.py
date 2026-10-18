from app.main import app
from app.api.routes.health import health


def test_health_handler_returns_ok():
    assert app.title == "Crypto Volatility and Tail Risk API"
    assert health() == {"status": "ok"}
