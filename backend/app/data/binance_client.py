"""Public Binance Spot market-data REST client with bounded retries."""

from __future__ import annotations

import time
from typing import Any

import requests

from app.core.config import Settings, load_settings


class BinancePublicClient:
    """Access public Spot market data; no API key or trading permission is used."""

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or load_settings()
        self.session = session or requests.Session()
        self.base_urls = (self.settings.binance_base_url, *self.settings.binance_fallback_urls)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        failures: list[str] = []
        for attempt in range(self.settings.request_retries):
            for base_url in self.base_urls:
                try:
                    response = self.session.get(
                        f"{base_url}{path}", params=params,
                        timeout=self.settings.request_timeout_seconds,
                        headers={"Accept": "application/json"},
                    )
                    if response.status_code in (418, 429):
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 ** attempt)
                        time.sleep(max(1.0, delay))
                        failures.append(f"{base_url}: HTTP {response.status_code}; waited {delay:g}s")
                        continue
                    response.raise_for_status()
                    return response.json()
                except (requests.RequestException, ValueError) as exc:
                    failures.append(f"{base_url}: {type(exc).__name__}: {exc}")
            if attempt + 1 < self.settings.request_retries:
                time.sleep(min(5.0, 0.5 * (2 ** attempt)))
        detail = "; ".join(failures[-len(self.base_urls):])
        raise RuntimeError(f"Binance public REST request failed for {path}: {detail}")

    def depth_snapshot(self, symbol: str, limit: int | None = None) -> dict[str, Any]:
        payload = self.get_json("/api/v3/depth", {
            "symbol": symbol.upper(), "limit": int(limit or self.settings.lob_depth_snapshot_limit),
        })
        if not isinstance(payload, dict) or "lastUpdateId" not in payload:
            raise ValueError("Binance depth endpoint returned an invalid snapshot")
        return payload

    def exchange_symbol_metadata(self, symbol: str) -> dict[str, Any]:
        payload = self.get_json("/api/v3/exchangeInfo", {"symbol": symbol.upper()})
        symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
        match = next((item for item in symbols if str(item.get("symbol", "")).upper() == symbol.upper()), None)
        if match is None:
            raise ValueError(f"Binance exchangeInfo did not return {symbol.upper()}")
        filters = {item.get("filterType"): item for item in match.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        return {
            "symbol": symbol.upper(), "status": match.get("status"),
            "base_asset": match.get("baseAsset"), "quote_asset": match.get("quoteAsset"),
            "tick_size": price_filter.get("tickSize"), "step_size": lot_filter.get("stepSize"),
            "min_quantity": lot_filter.get("minQty"),
        }
