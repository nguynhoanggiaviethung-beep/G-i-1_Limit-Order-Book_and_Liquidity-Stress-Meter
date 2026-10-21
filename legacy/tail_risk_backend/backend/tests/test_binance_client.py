from datetime import datetime, timezone

from app.data.binance_client import BinanceClient, to_milliseconds


def test_date_is_converted_as_utc():
    assert to_milliseconds("2020-01-01T00:00:00Z") == 1577836800000


def test_fetch_klines_preserves_binance_kline_fields():
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [[1577836800000, "1", "2", "0.5", "1.5", "10", 1577837099999, "15", 4, "5", "7.5", "0"]]

    class Session:
        def mount(self, *args):
            pass

        def get(self, *args, **kwargs):
            return Response()

    frame = BinanceClient(session=Session()).fetch_klines("BTCUSDT", start_time=1577836800000)
    assert frame.loc[0, "symbol"] == "BTCUSDT"
    assert frame.loc[0, "close"] == "1.5"
