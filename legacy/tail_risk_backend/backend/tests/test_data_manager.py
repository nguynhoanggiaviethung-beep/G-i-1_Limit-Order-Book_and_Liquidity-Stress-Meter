from datetime import datetime, timezone

from app.data.binance_client import RAW_COLUMNS
from app.data.data_manager import DataManager


def test_empty_binance_response_does_not_create_empty_snapshot(tmp_path):
    class Client:
        def server_time(self):
            return datetime(2020, 1, 1, tzinfo=timezone.utc)

        def fetch_klines(self, *args, **kwargs):
            import pandas as pd
            return pd.DataFrame(columns=RAW_COLUMNS + ["symbol", "interval"])

    manager = DataManager(Client(), tmp_path, start_date="2019-01-01")
    try:
        manager.sync("BTCUSDT")
    except RuntimeError as error:
        assert "no candles" in str(error)
    else:
        raise AssertionError("Expected an empty initial sync to fail")
