"""Read the latest available saved walk-forward forecast for the requested series."""

from pathlib import Path

import pandas as pd

from app.core.config import REPOSITORY_ROOT, load_settings


def _forecast_path(symbol: str, horizon_days: int) -> Path:
    return REPOSITORY_ROOT / "results" / "forecasts" / f"{symbol.lower()}_h{horizon_days}_walkforward.parquet"


def _risk_for_forecast(symbol: str, model: str, horizon_days: int, origin: pd.Timestamp, confidence_levels: list[float]) -> list[dict] | None:
    path = REPOSITORY_ROOT / "results" / "forecasts" / "risk_forecasts.parquet"
    if not path.exists():
        return None
    risks = pd.read_parquet(path)
    selected = risks.loc[
        risks["symbol"].eq(symbol)
        & risks["model"].eq(model)
        & risks["horizon_days"].eq(horizon_days)
        & (pd.to_datetime(risks["origin_time"], utc=True) == origin)
        & risks["confidence"].isin(confidence_levels)
    ]
    if selected.empty:
        return None
    columns = ["confidence", "method", "var_loss_percent", "expected_shortfall_loss_percent", "reliability"]
    return selected[[column for column in columns if column in selected]].to_dict("records")


def create_forecast(symbol: str, model: str, horizon_days: int, confidence_levels: list[float]) -> dict:
    symbol = symbol.upper()
    settings = load_settings()
    if symbol not in settings.symbols:
        raise ValueError(f"Unsupported symbol: {symbol}")
    invalid = set(confidence_levels) - set(settings.confidence_levels)
    if invalid:
        raise ValueError(f"Unsupported confidence level(s): {sorted(invalid)}")
    path = _forecast_path(symbol, horizon_days)
    if not path.exists():
        raise FileNotFoundError(f"No saved {horizon_days}-day forecast for {symbol}; run backend/scripts/run_walkforward.py")
    forecasts = pd.read_parquet(path)
    selected = forecasts.loc[(forecasts["model"] == model) & (forecasts["status"] == "ok")].copy()
    if selected.empty:
        available = sorted(forecasts.loc[forecasts["status"] == "ok", "model"].unique().tolist())
        raise FileNotFoundError(f"No successful forecast for model {model}; available models: {available}")
    row = selected.sort_values("origin_time").iloc[-1]
    origin = pd.Timestamp(row["origin_time"])
    return {
        "forecast_id": f"{symbol.lower()}-h{horizon_days}-{model}-{origin.strftime('%Y%m%d')}",
        "symbol": symbol,
        "model": model,
        "horizon_days": horizon_days,
        "origin_time": origin.isoformat(),
        "target_start": pd.Timestamp(row["target_start"]).isoformat(),
        "target_end": pd.Timestamp(row["target_end"]).isoformat(),
        "data_version": row.get("data_version"),
        "config_hash": row.get("config_hash"),
        "actual_rv_mean": None if pd.isna(row["actual_rv_mean"]) else float(row["actual_rv_mean"]),
        "rv_forecast": float(row["rv_forecast"]),
        "sigma_h_approx": float(row["sigma_h_approx"]),
        "status": row["status"],
        "var_es": _risk_for_forecast(symbol, model, horizon_days, origin, confidence_levels),
        "note": "Forecast loaded from immutable research snapshot; actual value is blank when unavailable at the frozen as-of date.",
    }


def get_forecast(forecast_id: str) -> dict:
    for symbol in load_settings().symbols:
        for horizon in (1, 7, 30):
            path = _forecast_path(symbol, horizon)
            if not path.exists():
                continue
            data = pd.read_parquet(path)
            rows = data.loc[
                (data["symbol"] == symbol)
                & (data["horizon_days"] == horizon)
                & data["status"].eq("ok")
            ]
            for _, row in rows.iterrows():
                candidate = f"{symbol.lower()}-h{horizon}-{row['model']}-{pd.Timestamp(row['origin_time']).strftime('%Y%m%d')}"
                if candidate == forecast_id:
                    origin = pd.Timestamp(row["origin_time"])
                    return {
                        "forecast_id": candidate,
                        "symbol": symbol,
                        "model": row["model"],
                        "horizon_days": horizon,
                        "origin_time": origin.isoformat(),
                        "target_start": pd.Timestamp(row["target_start"]).isoformat(),
                        "target_end": pd.Timestamp(row["target_end"]).isoformat(),
                        "data_version": row.get("data_version"),
                        "config_hash": row.get("config_hash"),
                        "actual_rv_mean": None if pd.isna(row["actual_rv_mean"]) else float(row["actual_rv_mean"]),
                        "rv_forecast": float(row["rv_forecast"]),
                        "sigma_h_approx": float(row["sigma_h_approx"]),
                        "status": row["status"],
                        "var_es": _risk_for_forecast(
                            symbol, row["model"], horizon,
                            pd.Timestamp(row["origin_time"]), list(load_settings().confidence_levels),
                        ),
                    }
    raise FileNotFoundError(f"Forecast not found: {forecast_id}")
