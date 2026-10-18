"""Load research and runtime settings from the repository configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = REPOSITORY_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class Settings:
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    interval: str = "5m"
    start_date: str = "2019-01-01"
    data_dir: Path = REPOSITORY_ROOT / "data"
    binance_base_url: str = "https://api.binance.com"
    request_timeout_seconds: float = 20.0
    request_retries: int = 5
    auto_sync_enabled: bool = True
    auto_sync_interval_seconds: int = 300
    auto_sync_grace_seconds: int = 10
    forecast_horizon_days: int = 1
    test_fraction: float = 0.2
    confidence_levels: tuple[float, ...] = (0.95, 0.975, 0.99)


def load_settings(path: Path | str | None = None) -> Settings:
    """Read settings.yaml, applying safe defaults for omitted values."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    raw: dict[str, Any] = {}
    if settings_path.exists():
        loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {settings_path}")
        raw = loaded

    data = raw.get("data", {})
    binance = raw.get("binance", {})
    auto_sync = raw.get("auto_sync", {})
    forecast = raw.get("forecast", {})
    risk = raw.get("risk", {})
    symbols = tuple(str(symbol).upper() for symbol in raw.get("symbols", ["BTCUSDT", "ETHUSDT"]))
    data_dir = Path(data.get("directory", "data"))
    if not data_dir.is_absolute():
        data_dir = REPOSITORY_ROOT / data_dir
    return Settings(
        symbols=symbols,
        interval=str(raw.get("interval", "5m")),
        start_date=str(raw.get("start_date", "2019-01-01")),
        data_dir=data_dir,
        binance_base_url=str(binance.get("base_url", "https://api.binance.com")).rstrip("/"),
        request_timeout_seconds=float(binance.get("timeout_seconds", 20)),
        request_retries=int(binance.get("retries", 5)),
        auto_sync_enabled=bool(auto_sync.get("enabled", True)),
        auto_sync_interval_seconds=int(auto_sync.get("interval_seconds", 300)),
        auto_sync_grace_seconds=int(auto_sync.get("grace_seconds", 10)),
        forecast_horizon_days=int(forecast.get("horizon_days", 1)),
        test_fraction=float(forecast.get("test_fraction", 0.2)),
        confidence_levels=tuple(float(value) for value in risk.get("confidence_levels", [0.95, 0.975, 0.99])),
    )
