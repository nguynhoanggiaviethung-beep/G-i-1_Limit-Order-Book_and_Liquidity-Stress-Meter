"""Read the latest precomputed forecasts produced by train_models.py."""

import json
from pathlib import Path

from app.core.config import load_settings


MODEL_ALIASES = {"benchmark": "rolling_mean_22"}
SUPPORTED_MODELS = ("rolling_mean_22", "har_rv", "random_forest", "garch")


def _read_bundle() -> dict:
    path = Path(__file__).resolve().parents[3] / "results" / "forecasts" / "latest_forecasts.json"
    if not path.exists():
        raise FileNotFoundError("No saved forecast; run backend/scripts/train_models.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_forecast(bundle: dict, symbol: str, model: str, confidence_levels: list[float]) -> dict:
    settings = load_settings()
    symbol = symbol.upper()
    model = MODEL_ALIASES.get(model.lower(), model.lower())
    if symbol not in settings.symbols:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}; choose one of {', '.join(SUPPORTED_MODELS)}")
    try:
        result = bundle["symbols"][symbol]
        forecast = result["forecasts"][model]
    except KeyError as exc:
        raise FileNotFoundError(f"No saved {model} forecast for {symbol}") from exc
    var_es = forecast.get("var_es", {})
    requested = {f"{level:.2f}" for level in confidence_levels}
    var_es = {level: values for level, values in var_es.items() if level in requested}
    target_date = result["target_time"][:10]
    return {
        "forecast_id": f"{symbol.lower()}-{model}-{target_date.replace('-', '')}",
        "symbol": symbol,
        "model": model,
        "horizon_days": result["forecast_horizon_days"],
        "model_steps_ahead": result["garch_steps_ahead"] if model == "garch" else result["forecast_horizon_days"],
        "origin_time": result["garch_origin_time"] if model == "garch" else result["origin_time"],
        "data_as_of_utc": result["data_as_of_utc"],
        "target_time": result["target_time"],
        "realized_variance": forecast["realized_variance"],
        "realized_volatility": forecast["realized_volatility"],
        "risk_unit": result["risk_unit"],
        "var_es": var_es or None,
    }


def create_forecast(symbol: str, model: str, horizon_minutes: int, confidence_levels: list[float]) -> dict:
    settings = load_settings()
    expected_minutes = settings.forecast_horizon_days * 1440
    if horizon_minutes != expected_minutes:
        raise ValueError(f"Only the configured {expected_minutes}-minute forecast is available")
    for level in confidence_levels:
        if level not in settings.confidence_levels:
            raise ValueError(f"Unsupported confidence level {level}; configured levels: {settings.confidence_levels}")
    return _format_forecast(_read_bundle(), symbol, model, confidence_levels)


def get_forecast(forecast_id: str) -> dict:
    bundle = _read_bundle()
    settings = load_settings()
    for symbol in settings.symbols:
        for model in SUPPORTED_MODELS:
            candidate = _format_forecast(bundle, symbol, model, list(settings.confidence_levels))
            if candidate["forecast_id"] == forecast_id:
                return candidate
    raise FileNotFoundError(f"Forecast not found: {forecast_id}")
