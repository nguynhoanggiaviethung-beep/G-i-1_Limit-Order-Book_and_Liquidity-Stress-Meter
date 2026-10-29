"""Load runtime settings shared by replay API and offline research scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = REPOSITORY_ROOT / "config" / "settings_live.yaml"


@dataclass(frozen=True)
class Settings:
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    data_dir: Path = REPOSITORY_ROOT / "data_processed" / "lob"
    raw_data_dir: Path = REPOSITORY_ROOT / "data_raw" / "binance_spot"
    interim_data_dir: Path = REPOSITORY_ROOT / "data_interim" / "binance_spot"
    binance_base_url: str = "https://data-api.binance.vision"
    binance_fallback_urls: tuple[str, ...] = (
        "https://api.binance.com",
        "https://api-gcp.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://api4.binance.com",
    )
    request_timeout_seconds: float = 4.0
    request_retries: int = 2
    lob_collection_enabled: bool = True
    lob_ws_url: str = "wss://stream.binance.com:443"
    lob_ws_fallback_urls: tuple[str, ...] = (
        "wss://data-stream.binance.vision:443",
        "wss://stream.binance.com:9443",
    )
    lob_depth_snapshot_limit: int = 100
    lob_update_speed: str = "100ms"
    lob_sampling_interval_seconds: int = 1
    lob_stale_after_seconds: int = 5
    lob_raw_file_max_bytes: int = 67108864
    lob_flush_rows: int = 500
    lob_flush_seconds: int = 10
    lob_temporal_history_samples: int = 3600
    lob_target_horizons_seconds: tuple[int, ...] = (60, 300, 3600)
    lob_minimum_train_rows: int = 1000
    lob_test_fraction: float = 0.2
    lob_refit_interval_samples: int = 3600
    lob_online_minimum_train_examples: int = 60
    lob_online_maximum_train_examples: int = 500
    lob_online_refit_every_examples: int = 10
    lob_long_target_horizons_days: tuple[int, ...] = (1, 7, 30)
    lob_long_online_minimum_train_examples: int = 30
    lob_long_online_maximum_train_examples: int = 500
    lob_long_minimum_daily_coverage: float = 0.995
    lob_classification_horizons_seconds: tuple[int, ...] = (1, 5, 10)
    lob_primary_horizon_seconds: int = 5
    lob_direction_delta_quantile: float = 0.5
    lob_stress_definition: str = "composite"
    lob_stress_quantile: float = 0.975
    lob_stress_depth_quantile: float = 0.025
    lob_spread_sensitivity_quantile: float = 0.95
    lob_stress_impact_notional_usdt: float = 100000.0
    lob_stress_impact_notional_by_symbol: dict[str, float] | None = None
    lob_tick_size_by_symbol: dict[str, float] | None = None
    lob_stress_min_positive_rate: float = 0.02
    lob_stress_max_positive_rate: float = 0.10
    lob_max_missing_feature_fraction: float = 0.10
    lob_max_staleness_seconds: int = 2
    lob_first_validation_week_utc: str = "2026-10-05T00:00:00Z"
    lob_development_end_utc: str = "2026-10-11T23:59:59Z"
    lob_untouched_test_start_utc: str = "2026-10-12T00:00:00Z"
    lob_untouched_test_end_utc: str = "2026-10-18T23:59:59Z"
    lob_embargo_seconds: int = 300
    lob_feature_lookback_seconds: int = 300
    lob_deeplob_lookback_seconds: int = 100
    lob_deeplob_max_train_rows: int = 8000
    # Runtime model artifacts and the evaluation summary served by the API.
    lob_artifact_dir: Path = REPOSITORY_ROOT / "backend" / "artifacts" / "meter"
    lob_evaluation_dir: Path = REPOSITORY_ROOT / "outputs" / "tables" / "historical_archive_pilot"

    def impact_notional_for(self, symbol: str) -> float:
        return float((self.lob_stress_impact_notional_by_symbol or {}).get(symbol.upper(), self.lob_stress_impact_notional_usdt))

    def tick_size_for(self, symbol: str) -> float:
        mapping = self.lob_tick_size_by_symbol or {}
        if symbol.upper() not in mapping:
            raise ValueError(f"No price tick size is configured for {symbol.upper()}")
        return float(mapping[symbol.upper()])

    def __post_init__(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be non-empty and unique")
        if any(not symbol.isalnum() or symbol != symbol.upper() for symbol in self.symbols):
            raise ValueError("symbols must be uppercase Binance symbol names")
        if not 1 <= self.lob_depth_snapshot_limit <= 5000:
            raise ValueError("lob.depth_snapshot_limit must be between 1 and 5000")
        if self.lob_update_speed not in {"", "100ms"}:
            raise ValueError("lob.update_speed must be '100ms' or empty for the default stream speed")
        if self.lob_sampling_interval_seconds < 1 or self.lob_stale_after_seconds < self.lob_sampling_interval_seconds:
            raise ValueError("lob.stale_after_seconds must be at least the positive sampling interval")
        if self.lob_raw_file_max_bytes < 1024 * 1024:
            raise ValueError("lob.raw_file_max_bytes must be at least 1 MiB")
        if self.lob_flush_rows < 1 or self.lob_flush_seconds < 1 or self.lob_temporal_history_samples < 1:
            raise ValueError("LOB flush and temporal-history settings must be positive")
        if not self.lob_target_horizons_seconds or any(value < 1 for value in self.lob_target_horizons_seconds):
            raise ValueError("lob.target_horizons_seconds must contain positive values")
        if any(value % self.lob_sampling_interval_seconds for value in self.lob_target_horizons_seconds):
            raise ValueError("Each LOB horizon must be divisible by lob.sampling_interval_seconds")
        if not 0 < self.lob_test_fraction < 1 or self.lob_minimum_train_rows < 30 or self.lob_refit_interval_samples < 1:
            raise ValueError("LOB evaluation settings are invalid")
        if self.lob_online_minimum_train_examples < 30:
            raise ValueError("lob.online_minimum_train_examples must be at least 30")
        if self.lob_online_maximum_train_examples < self.lob_online_minimum_train_examples:
            raise ValueError("lob.online_maximum_train_examples must be at least the online minimum")
        if self.lob_online_refit_every_examples < 1:
            raise ValueError("lob.online_refit_every_examples must be positive")
        if not self.lob_long_target_horizons_days or any(value not in (1, 7, 30) for value in self.lob_long_target_horizons_days):
            raise ValueError("lob.target_horizons_days must contain values from 1, 7, and 30")
        if self.lob_long_online_minimum_train_examples < 30:
            raise ValueError("lob.long_online_minimum_train_examples must be at least 30")
        if self.lob_long_online_maximum_train_examples < self.lob_long_online_minimum_train_examples:
            raise ValueError("lob.long_online_maximum_train_examples must be at least the long minimum")
        if not 0.9 <= self.lob_long_minimum_daily_coverage <= 1:
            raise ValueError("lob.minimum_daily_coverage must be in [0.9, 1]")
        if not self.lob_classification_horizons_seconds or any(value not in (1, 5, 10) for value in self.lob_classification_horizons_seconds):
            raise ValueError("lob.classification_horizons_seconds must use the configured 1, 5, and 10 second horizons")
        if self.lob_primary_horizon_seconds not in self.lob_classification_horizons_seconds:
            raise ValueError("lob.primary_horizon_seconds must be one of the configured classification horizons")
        if self.lob_stress_definition not in {"spread_only", "composite"}:
            raise ValueError("lob.stress_definition must be 'spread_only' or 'composite'")
        if self.lob_deeplob_max_train_rows < 60:
            raise ValueError("lob.deeplob_max_train_rows must be at least 60")
        if not 0 < self.lob_direction_delta_quantile < 1 or not 0 < self.lob_stress_depth_quantile < self.lob_stress_quantile < 1:
            raise ValueError("Direction and stress quantiles must be ordered inside (0, 1)")
        if not 0 < self.lob_spread_sensitivity_quantile < 1:
            raise ValueError("lob.spread_sensitivity_quantile must be in (0, 1)")
        if self.lob_stress_impact_notional_usdt <= 0 or min(self.lob_embargo_seconds, self.lob_feature_lookback_seconds, self.lob_deeplob_lookback_seconds) < 1:
            raise ValueError("Impact notional, feature lookback, deep-model lookback, and embargo must be positive")
        if not 0 < self.lob_stress_min_positive_rate < self.lob_stress_max_positive_rate < 1:
            raise ValueError("Stress prevalence bounds must be ordered inside (0, 1)")
        if not 0 < self.lob_max_missing_feature_fraction < 1 or self.lob_max_staleness_seconds < 0:
            raise ValueError("Missing-feature fraction and replay staleness limit are invalid")
        impact_map = self.lob_stress_impact_notional_by_symbol or {}
        tick_map = self.lob_tick_size_by_symbol or {}
        if any(float(value) <= 0 for value in impact_map.values()) or any(float(value) <= 0 for value in tick_map.values()):
            raise ValueError("Symbol-specific impact notionals and tick sizes must be positive")
        minimum_embargo = max(max(self.lob_classification_horizons_seconds), self.lob_feature_lookback_seconds, self.lob_deeplob_lookback_seconds)
        if self.lob_embargo_seconds < minimum_embargo:
            raise ValueError("lob.embargo_seconds must cover the horizon, longest feature lookback, and DeepLOB sequence")
        first_validation_week = pd.Timestamp(self.lob_first_validation_week_utc)
        development_end = pd.Timestamp(self.lob_development_end_utc)
        test_start = pd.Timestamp(self.lob_untouched_test_start_utc)
        test_end = pd.Timestamp(self.lob_untouched_test_end_utc)
        if any(value.tzinfo is None for value in (first_validation_week, development_end, test_start, test_end)):
            raise ValueError("LOB evaluation timestamps must include an explicit UTC offset")
        if first_validation_week.tz_convert("UTC").weekday() != 0 or first_validation_week.tz_convert("UTC").time().isoformat() != "00:00:00":
            raise ValueError("lob.first_validation_week_utc must be a Monday at 00:00 UTC")
        if test_start.tz_convert("UTC").weekday() != 0 or test_start.tz_convert("UTC").time().isoformat() != "00:00:00":
            raise ValueError("lob.untouched_test_start_utc must be a Monday at 00:00 UTC")
        test_span = test_end.tz_convert("UTC") - test_start.tz_convert("UTC") + timedelta(seconds=1)
        if test_span < timedelta(days=7) or test_span % timedelta(days=7):
            raise ValueError("The untouched test window must cover one or more full UTC weeks")
        if development_end.tz_convert("UTC") != test_start.tz_convert("UTC") - timedelta(seconds=1):
            raise ValueError("lob.development_end_utc must end immediately before the untouched test week")
        if first_validation_week.tz_convert("UTC") >= development_end.tz_convert("UTC"):
            raise ValueError("The first validation week must precede development_end_utc")
        if self.request_timeout_seconds <= 0 or self.request_retries < 1:
            raise ValueError("Binance timeout and retry settings must be positive")


def load_settings(path: Path | str | None = None) -> Settings:
    """Read the selected settings file, applying safe defaults for omitted values."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    raw: dict[str, Any] = {}
    if settings_path.exists():
        loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {settings_path}")
        raw = loaded

    data = raw.get("data", {})
    binance = raw.get("binance", {})
    lob = raw.get("lob", {})
    symbols_raw = raw.get("symbols")
    if symbols_raw is None:
        symbols_path = settings_path.with_name("symbols.yaml")
        symbols_data = yaml.safe_load(symbols_path.read_text(encoding="utf-8")) if symbols_path.exists() else None
        symbols_raw = (symbols_data or {}).get("symbols", ["BTCUSDT", "ETHUSDT"])
    symbols = tuple(str(symbol).upper() for symbol in symbols_raw)
    data_dir = Path(data.get("directory", "data_processed/lob"))
    if not data_dir.is_absolute():
        data_dir = REPOSITORY_ROOT / data_dir
    raw_data_dir = Path(data.get("raw_directory", "data_raw/binance_spot"))
    if not raw_data_dir.is_absolute():
        raw_data_dir = REPOSITORY_ROOT / raw_data_dir
    interim_data_dir = Path(data.get("interim_directory", "data_interim/binance_spot"))
    if not interim_data_dir.is_absolute():
        interim_data_dir = REPOSITORY_ROOT / interim_data_dir
    artifact_dir = Path(lob.get("artifact_directory", "backend/artifacts/meter"))
    if not artifact_dir.is_absolute():
        artifact_dir = REPOSITORY_ROOT / artifact_dir
    evaluation_dir = Path(lob.get("evaluation_directory", "outputs/tables/historical_archive_pilot"))
    if not evaluation_dir.is_absolute():
        evaluation_dir = REPOSITORY_ROOT / evaluation_dir
    return Settings(
        symbols=symbols,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        interim_data_dir=interim_data_dir,
        binance_base_url=str(binance.get("base_url", "https://data-api.binance.vision")).rstrip("/"),
        binance_fallback_urls=tuple(
            str(value).rstrip("/")
            for value in binance.get("fallback_base_urls", [
                "https://api.binance.com",
                "https://api-gcp.binance.com",
                "https://api1.binance.com",
                "https://api2.binance.com",
                "https://api3.binance.com",
                "https://api4.binance.com",
            ])
        ),
        request_timeout_seconds=float(binance.get("timeout_seconds", 4)),
        request_retries=int(binance.get("retries", 2)),
        lob_collection_enabled=bool(lob.get("collection_enabled", True)),
        lob_ws_url=str(lob.get("websocket_base_url", "wss://stream.binance.com:443")).rstrip("/"),
        lob_ws_fallback_urls=tuple(
            str(value).rstrip("/") for value in lob.get("websocket_fallback_urls", [
                "wss://data-stream.binance.vision:443",
                "wss://stream.binance.com:9443",
            ])
        ),
        lob_depth_snapshot_limit=int(lob.get("depth_snapshot_limit", 100)),
        lob_update_speed=str(lob.get("update_speed", "100ms")),
        lob_sampling_interval_seconds=int(lob.get("sampling_interval_seconds", 1)),
        lob_stale_after_seconds=int(lob.get("stale_after_seconds", 5)),
        lob_raw_file_max_bytes=int(lob.get("raw_file_max_bytes", 67108864)),
        lob_flush_rows=int(lob.get("flush_rows", 500)),
        lob_flush_seconds=int(lob.get("flush_seconds", 10)),
        lob_temporal_history_samples=int(lob.get("temporal_history_samples", 3600)),
        lob_target_horizons_seconds=tuple(int(value) for value in lob.get("target_horizons_seconds", [60, 300, 3600])),
        lob_minimum_train_rows=int(lob.get("minimum_train_rows", 1000)),
        lob_test_fraction=float(lob.get("test_fraction", 0.2)),
        lob_refit_interval_samples=int(lob.get("refit_interval_samples", 3600)),
        lob_online_minimum_train_examples=int(lob.get("online_minimum_train_examples", 60)),
        lob_online_maximum_train_examples=int(lob.get("online_maximum_train_examples", 500)),
        lob_online_refit_every_examples=int(lob.get("online_refit_every_examples", 10)),
        lob_long_target_horizons_days=tuple(int(value) for value in lob.get("target_horizons_days", [1, 7, 30])),
        lob_long_online_minimum_train_examples=int(lob.get("long_online_minimum_train_examples", 30)),
        lob_long_online_maximum_train_examples=int(lob.get("long_online_maximum_train_examples", 500)),
        lob_long_minimum_daily_coverage=float(lob.get("minimum_daily_coverage", 0.995)),
        lob_classification_horizons_seconds=tuple(int(value) for value in lob.get("classification_horizons_seconds", [1, 5, 10])),
        lob_primary_horizon_seconds=int(lob.get("primary_horizon_seconds", 5)),
        lob_direction_delta_quantile=float(lob.get("direction_delta_quantile", 0.5)),
        lob_stress_definition=str(lob.get("stress_definition", "composite")),
        lob_stress_quantile=float(lob.get("stress_quantile", 0.975)),
        lob_stress_depth_quantile=float(lob.get("stress_depth_quantile", 0.025)),
        lob_spread_sensitivity_quantile=float(lob.get("spread_sensitivity_quantile", 0.95)),
        lob_stress_impact_notional_usdt=float(lob.get("stress_impact_notional_usdt", 100000)),
        lob_stress_impact_notional_by_symbol={
            str(key).upper(): float(value) for key, value in lob.get("stress_impact_notional_by_symbol", {}).items()
        },
        lob_tick_size_by_symbol={
            str(key).upper(): float(value) for key, value in lob.get("tick_size_by_symbol", {}).items()
        },
        lob_stress_min_positive_rate=float(lob.get("stress_min_positive_rate", 0.02)),
        lob_stress_max_positive_rate=float(lob.get("stress_max_positive_rate", 0.10)),
        lob_max_missing_feature_fraction=float(lob.get("max_missing_feature_fraction", 0.10)),
        lob_max_staleness_seconds=int(lob.get("max_staleness_seconds", 2)),
        lob_first_validation_week_utc=str(lob.get("first_validation_week_utc", "2026-10-05T00:00:00Z")),
        lob_development_end_utc=str(lob.get("development_end_utc", "2026-10-11T23:59:59Z")),
        lob_untouched_test_start_utc=str(lob.get("untouched_test_start_utc", "2026-10-12T00:00:00Z")),
        lob_untouched_test_end_utc=str(lob.get("untouched_test_end_utc", "2026-10-18T23:59:59Z")),
        lob_embargo_seconds=int(lob.get("embargo_seconds", 300)),
        lob_feature_lookback_seconds=int(lob.get("feature_lookback_seconds", 300)),
        lob_deeplob_lookback_seconds=int(lob.get("deeplob_lookback_seconds", 100)),
        lob_deeplob_max_train_rows=int(lob.get("deeplob_max_train_rows", 8000)),
        lob_artifact_dir=artifact_dir,
        lob_evaluation_dir=evaluation_dir,
    )
