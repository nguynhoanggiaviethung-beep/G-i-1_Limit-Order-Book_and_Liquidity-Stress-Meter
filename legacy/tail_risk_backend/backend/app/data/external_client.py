"""Point-in-time aware public market and macro data adapters."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EVENT_COLUMNS = [
    "symbol", "feature_name", "observation_time_utc", "available_at_utc",
    "value", "unit", "source", "source_id",
]


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


class ExternalDataClient:
    """Fetch public Binance USDⓈ-M and FRED observations with availability times."""

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 5,
        binance_base_url: str = "https://fapi.binance.com",
        fred_base_url: str = "https://api.stlouisfed.org/fred",
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.binance_base_url = binance_base_url.rstrip("/")
        self.fred_base_url = fred_base_url.rstrip("/")
        self.session = session or requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, url: str, params: dict) -> dict | list:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            # Do not include a request URL in errors; FRED keys are query parameters.
            raise RuntimeError(f"GET {url} failed: {type(exc).__name__}") from None
        if response.status_code == 418:
            raise RuntimeError("Binance returned HTTP 418; stop requests and retry later")
        payload = response.json()
        if response.status_code >= 400:
            if isinstance(payload, dict) and "code" in payload:
                raise RuntimeError(f"Market data API error {payload['code']}: {payload.get('msg', '')}")
            raise RuntimeError(f"GET {url} returned HTTP {response.status_code}")
        if isinstance(payload, dict) and "code" in payload and int(payload["code"]) < 0:
            raise RuntimeError(f"Market data API error {payload['code']}: {payload.get('msg', '')}")
        return payload

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    @staticmethod
    def _event_rows(rows, *, symbol, feature_name, time_key, value_key, unit, source, available_lag):
        if not rows:
            return ExternalDataClient._empty()
        frame = pd.DataFrame(rows)
        observed = pd.to_datetime(pd.to_numeric(frame[time_key]), unit="ms", utc=True)
        values = pd.to_numeric(frame[value_key], errors="coerce")
        return pd.DataFrame({
            "symbol": symbol,
            "feature_name": feature_name,
            "observation_time_utc": observed,
            "available_at_utc": observed + available_lag,
            "value": values,
            "unit": unit,
            "source": source,
            "source_id": frame.get("symbol", frame.get("pair", symbol)).astype(str),
        }).dropna(subset=["value"])

    def fetch_funding_rates(self, symbol: str, start_time, end_time) -> pd.DataFrame:
        """Download paginated funding settlements; the event becomes usable 1m later."""
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        if end <= start:
            raise ValueError("end_time must be later than start_time")
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows = []
        while cursor <= end_ms:
            page = self._get(f"{self.binance_base_url}/fapi/v1/fundingRate", {
                "symbol": symbol.upper(), "startTime": cursor, "endTime": end_ms, "limit": 1000,
            })
            if not page:
                break
            rows.extend(page)
            last_time = int(page[-1]["fundingTime"])
            if len(page) < 1000 or last_time >= end_ms:
                break
            next_cursor = last_time + 1
            if next_cursor <= cursor:
                raise RuntimeError("Binance funding pagination cursor did not advance")
            cursor = next_cursor
        return self._event_rows(
            rows,
            symbol=symbol.upper(),
            feature_name="funding_rate",
            time_key="fundingTime",
            value_key="fundingRate",
            unit="rate_per_settlement",
            source="binance_usdm_funding_rate",
            available_lag=pd.Timedelta(1, unit="min"),
        )

    def fetch_open_interest_history(
        self, symbol: str, start_time, end_time, period: str = "1d"
    ) -> pd.DataFrame:
        """Fetch open interest; Binance exposes only the latest month for this endpoint."""
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        if end <= start:
            raise ValueError("end_time must be later than start_time")
        available_start = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(29, unit="D")
        start = max(start, available_start)
        end = min(end, pd.Timestamp.now(tz="UTC"))
        if end <= start:
            return self._empty()
        page = self._get(f"{self.binance_base_url}/futures/data/openInterestHist", {
            "symbol": symbol.upper(), "period": period,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 500,
        })
        return self._event_rows(
            page,
            symbol=symbol.upper(),
            feature_name="open_interest_value",
            time_key="timestamp",
            value_key="sumOpenInterestValue",
            unit="quote_asset_value",
            source="binance_usdm_open_interest_hist",
            available_lag=pd.Timedelta(1, unit="min"),
        )

    def fetch_basis_history(
        self, symbol: str, start_time, end_time, period: str = "1d"
    ) -> pd.DataFrame:
        """Fetch perpetual futures basis; Binance exposes only the latest 30 days."""
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        if end <= start:
            raise ValueError("end_time must be later than start_time")
        available_start = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(29, unit="D")
        start = max(start, available_start)
        end = min(end, pd.Timestamp.now(tz="UTC"))
        if end <= start:
            return self._empty()
        page = self._get(f"{self.binance_base_url}/futures/data/basis", {
            "pair": symbol.upper(), "contractType": "PERPETUAL", "period": period,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 500,
        })
        if not page:
            return self._empty()
        frame = pd.DataFrame(page)
        observed = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
        period_ms = pd.Timedelta(period).total_seconds() * 1000
        value = pd.to_numeric(frame["basisRate"], errors="coerce")
        return pd.DataFrame({
            "symbol": symbol.upper(),
            "feature_name": "basis_rate",
            "observation_time_utc": observed,
            "available_at_utc": observed + pd.to_timedelta(period_ms + 60_000, unit="ms"),
            "value": value,
            "unit": "futures_basis_rate",
            "source": "binance_usdm_basis",
            "source_id": frame.get("pair", symbol).astype(str),
        }).dropna(subset=["value"])

    def fetch_fred_daily(self, series_id: str, api_key: str, start_date, end_date) -> pd.DataFrame:
        """Fetch initial-release observations, preserving the conservative release date."""
        if not api_key:
            raise ValueError("Set FRED_API_KEY to download FRED/ALFRED series")
        start = pd.Timestamp(start_date).date().isoformat()
        end = pd.Timestamp(end_date).date().isoformat()
        payload = self._get(f"{self.fred_base_url}/series/observations", {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "output_type": 4,
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
            "limit": 100000,
        })
        rows = payload.get("observations", [])
        if not rows:
            return self._empty()
        frame = pd.DataFrame(rows)
        values = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
        observed = pd.to_datetime(frame["date"], utc=True)
        realtime = pd.to_datetime(frame["realtime_start"], utc=True)
        # Never expose a daily macro value before both the observation day closed
        # and its recorded first-release date passed.
        available = pd.concat([
            observed + pd.Timedelta(1, unit="D"),
            realtime + pd.Timedelta(1, unit="D"),
        ], axis=1).max(axis=1)
        return pd.DataFrame({
            "symbol": pd.Series([None] * len(frame), dtype="object"),
            "feature_name": series_id.lower(),
            "observation_time_utc": observed,
            "available_at_utc": pd.to_datetime(available, utc=True),
            "value": values,
            "unit": "fred_series_value",
            "source": "fred_initial_release",
            "source_id": series_id,
        }).dropna(subset=["value"])
