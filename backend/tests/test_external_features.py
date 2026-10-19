import pandas as pd

from app.features.external import attach_point_in_time_external
from app.data.external_client import ExternalDataClient


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def mount(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return _Response(self.payload)


def test_external_features_only_use_events_available_by_each_origin():
    daily = pd.DataFrame({
        "utc_date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "available_at_utc": pd.to_datetime(["2024-01-01T16:00:00Z", "2024-01-02T00:00:00Z"], utc=True),
        "is_complete_day": [True, True],
    })
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"],
        "feature_name": ["funding_rate"],
        "observation_time_utc": pd.to_datetime(["2024-01-01T16:00:00Z"], utc=True),
        "available_at_utc": pd.to_datetime(["2024-01-01T16:01:00Z"], utc=True),
        "value": [0.001],
    })

    result, report = attach_point_in_time_external(
        daily,
        events,
        "BTCUSDT",
        min_coverage=0.5,
        coverage_start="2024-01-02T00:00:00Z",
    )

    assert pd.isna(result.loc[0, "external_funding_rate_last"])
    assert result.loc[1, "external_funding_rate_last"] == 0.001
    assert report["funding_rate"]["included"]
    assert report["funding_rate"]["coverage"] == 1.0


def test_sparse_external_feature_is_excluded_by_coverage_gate():
    daily = pd.DataFrame({
        "utc_date": pd.date_range("2024-01-01", periods=4, tz="UTC"),
        "available_at_utc": pd.date_range("2024-01-02", periods=4, tz="UTC"),
        "is_complete_day": [True] * 4,
    })
    events = pd.DataFrame({
        "symbol": ["ETHUSDT"],
        "feature_name": ["open_interest_value"],
        "observation_time_utc": pd.to_datetime(["2024-01-03T00:00:00Z"], utc=True),
        "available_at_utc": pd.to_datetime(["2024-01-03T00:01:00Z"], utc=True),
        "value": [100.0],
    })

    result, report = attach_point_in_time_external(daily, events, "ETHUSDT", min_coverage=0.95)

    assert not report["open_interest_value"]["included"]
    assert "external_open_interest_value_last" not in result


def test_fred_initial_release_is_not_available_before_conservative_release_time():
    client = ExternalDataClient(session=_Session({"observations": [{
        "date": "2024-01-01",
        "realtime_start": "2024-01-02",
        "value": "13.2",
    }]}))

    result = client.fetch_fred_daily("VIXCLS", "example-key", "2024-01-01", "2024-01-02")

    assert result.loc[0, "feature_name"] == "vixcls"
    assert result.loc[0, "available_at_utc"] == pd.Timestamp("2024-01-03T00:00:00Z")
