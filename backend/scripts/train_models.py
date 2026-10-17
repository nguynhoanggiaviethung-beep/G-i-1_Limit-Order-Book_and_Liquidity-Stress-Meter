"""Fit one-day realized-volatility forecasts and save out-of-sample results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.evaluation.volatility_metrics import volatility_metrics  # noqa: E402
from app.models.garch import fit_garch  # noqa: E402
from app.models.har import HAR_FEATURES, make_har_model  # noqa: E402
from app.models.machine_learning import make_random_forest  # noqa: E402


LOGGER = logging.getLogger(__name__)


def make_daily_dataset(frame: pd.DataFrame, candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """Use same-time intraday RV snapshots at day t to predict full RV on day t+1."""
    daily = frame.copy()
    daily["utc_date"] = pd.to_datetime(daily["utc_date"], utc=True).dt.floor("D")
    daily = daily.set_index("utc_date").sort_index()

    candle_data = candles.copy()
    candle_data["open_time"] = pd.to_datetime(candle_data["open_time"], utc=True)
    candle_data = candle_data.sort_values("open_time").reset_index(drop=True)
    candle_data["log_return"] = 100 * np.log(candle_data["close"] / candle_data["close"].shift(1))
    candle_data["utc_date"] = candle_data["open_time"].dt.floor("D")
    candle_data["minute_of_day"] = candle_data["open_time"].dt.hour * 60 + candle_data["open_time"].dt.minute
    latest_time = candle_data["open_time"].max()
    cutoff_minute = int(latest_time.hour * 60 + latest_time.minute)
    expected_bars = cutoff_minute // 5 + 1
    snapshots = candle_data[candle_data["minute_of_day"] <= cutoff_minute].copy()
    snapshots["squared_return"] = snapshots["log_return"].pow(2)
    partial = snapshots.groupby("utc_date").agg(
        partial_variance=("squared_return", "sum"),
        observations=("log_return", "count"),
        partial_return=("log_return", "sum"),
    )
    partial.loc[partial["observations"] != expected_bars, ["partial_variance", "partial_return"]] = np.nan

    calendar = pd.date_range(daily.index.min(), candle_data["utc_date"].max(), freq="D", tz="UTC", name="utc_date")
    daily = daily.reindex(calendar)
    target_is_complete = daily["observations"].eq(288)
    target_rv = daily["realized_variance"].where(target_is_complete)
    partial = partial.reindex(calendar)
    rv_snapshot = partial["partial_variance"]
    features = pd.DataFrame(index=calendar)
    features["rv_lag_1"] = rv_snapshot
    features["rv_lag_5"] = rv_snapshot.shift(4)
    features["rv_lag_22"] = rv_snapshot.shift(21)
    features["rv_mean_5"] = rv_snapshot.rolling(5, min_periods=5).mean()
    features["rv_mean_22"] = rv_snapshot.rolling(22, min_periods=22).mean()
    features["partial_return"] = partial["partial_return"]
    target = target_rv.shift(-1).rename("actual_rv_next_day")
    return features, target, cutoff_minute


def fit_and_evaluate(frame: pd.DataFrame, candles: pd.DataFrame, test_fraction: float) -> tuple[dict, pd.DataFrame, dict]:
    features, target, cutoff_minute = make_daily_dataset(frame, candles)
    valid = features.notna().all(axis=1) & target.notna()
    x, y = features.loc[valid], target.loc[valid]
    if len(x) < 300:
        raise ValueError(f"Need at least 300 valid daily samples; got {len(x)}")
    target_dates = x.index + pd.offsets.Day(1)
    split_index = min(max(int(len(x) * (1 - test_fraction)), 1), len(x) - 1)
    split_date = target_dates[split_index]
    train_mask = target_dates < split_date
    test_mask = ~train_mask
    if train_mask.sum() < 100 or test_mask.sum() < 30:
        raise ValueError("The configured test fraction leaves too few train or test samples")

    x_train, y_train = x.loc[train_mask], y.loc[train_mask]
    x_test, y_test = x.loc[test_mask], y.loc[test_mask]
    model_predictions: dict[str, np.ndarray] = {
        "rolling_mean_22": x_test["rv_mean_22"].to_numpy(dtype=float),
    }

    har = make_har_model()
    har.fit(x_train[HAR_FEATURES], y_train)
    model_predictions["har_rv"] = np.maximum(har.predict(x_test[HAR_FEATURES]), 0)

    forest = make_random_forest()
    forest.fit(x_train, y_train)
    model_predictions["random_forest"] = np.maximum(forest.predict(x_test), 0)

    forecast_origin_offset = timedelta(minutes=cutoff_minute + 5, milliseconds=-1)
    predictions = pd.DataFrame({
        "symbol": frame["symbol"].iloc[0],
        "origin_time": x_test.index + forecast_origin_offset,
        "target_time": x_test.index + pd.offsets.Day(1),
        "actual_realized_variance": y_test.to_numpy(dtype=float),
        "actual_realized_volatility": np.sqrt(y_test.to_numpy(dtype=float)),
    })
    metrics: dict[str, dict[str, float]] = {}
    for name, values in model_predictions.items():
        predictions[f"{name}_variance"] = values
        predictions[f"{name}_volatility"] = np.sqrt(values)
        metrics[name] = volatility_metrics(y_test, values)

    all_supervised = features.loc[features.notna().all(axis=1) & target.notna()]
    all_target = target.loc[all_supervised.index]
    latest_origin = features.dropna().index.max()
    latest_target = latest_origin + pd.offsets.Day(1)
    latest_row = features.loc[[latest_origin]]
    final_har = make_har_model().fit(all_supervised[HAR_FEATURES], all_target)
    final_forest = make_random_forest().fit(all_supervised, all_target)
    latest_forecasts = {
        "rolling_mean_22": float(latest_row["rv_mean_22"].iloc[0]),
        "har_rv": max(float(final_har.predict(latest_row[HAR_FEATURES])[0]), 0.0),
        "random_forest": max(float(final_forest.predict(latest_row)[0]), 0.0),
    }

    return metrics, predictions, {
        "split_date": split_date.isoformat(),
        "test_start": predictions["target_time"].min().isoformat(),
        "test_end": predictions["target_time"].max().isoformat(),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "origin_time": (latest_origin + forecast_origin_offset).isoformat(),
        "target_time": latest_target.isoformat(),
        "feature_cutoff_utc": f"{cutoff_minute // 60:02d}:{cutoff_minute % 60:02d}",
        "cutoff_minute": cutoff_minute,
        "forecasts": latest_forecasts,
    }


def garch_forecast_and_risk(frame: pd.DataFrame, confidence_levels: tuple[float, ...], horizon_steps: int) -> tuple[float, dict[str, dict[str, float]], str]:
    """Fit daily GARCH(1,1); return next-day variance and Student-t VaR/ES."""
    from scipy.stats import t as student_t

    returns = frame.loc[frame["observations"].eq(288), "daily_return"].dropna()
    if len(returns) < 250:
        raise ValueError("Need at least 250 complete daily returns for GARCH")
    result = fit_garch(returns)
    forecast_variance = float(result.forecast(horizon=horizon_steps, reindex=False).variance.iloc[-1, horizon_steps - 1])
    sigma = float(np.sqrt(max(forecast_variance, 0)))
    degrees = float(result.params.get("nu", 8.0))
    scale = float(np.sqrt((degrees - 2) / degrees)) if degrees > 2 else 1.0
    risk: dict[str, dict[str, float]] = {}
    for confidence in confidence_levels:
        tail_probability = 1.0 - confidence
        quantile = float(student_t.ppf(tail_probability, df=degrees))
        var = max(-sigma * scale * quantile, 0.0)
        es = sigma * scale * (degrees + quantile**2) / (degrees - 1) * float(student_t.pdf(quantile, df=degrees)) / tail_probability
        risk[f"{confidence:.2f}"] = {"var_loss_percent": var, "expected_shortfall_loss_percent": float(es)}
    last_return_date = pd.to_datetime(frame.loc[frame["observations"].eq(288), "utc_date"], utc=True).max()
    return forecast_variance, risk, last_return_date.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="Symbol to train; may be repeated")
    args = parser.parse_args()
    configure_logging()
    settings = load_settings()
    if settings.forecast_horizon_days != 1:
        raise ValueError("This MVP trainer currently supports a 1-day horizon")
    if not 0.05 <= settings.test_fraction <= 0.5:
        raise ValueError("forecast.test_fraction must be between 0.05 and 0.5")

    metrics_by_symbol: dict[str, dict] = {}
    forecasts_by_symbol: dict[str, dict] = {}
    output_dir = ROOT / "results" / "forecasts"
    metrics_dir = ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for symbol in [value.upper() for value in (args.symbol or settings.symbols)]:
        source = settings.data_dir / "features" / f"symbol={symbol}" / "daily_realized_volatility.parquet"
        if not source.exists():
            raise FileNotFoundError(f"Run build_features.py first; missing {source}")
        daily = pd.read_parquet(source)
        daily["symbol"] = symbol
        candle_path = settings.data_dir / "processed" / f"symbol={symbol}" / f"interval={settings.interval}" / "klines.parquet"
        candles = pd.read_parquet(candle_path)
        metrics, predictions, summary = fit_and_evaluate(daily, candles, settings.test_fraction)
        target_date = pd.Timestamp(summary["target_time"])
        last_full_day = pd.to_datetime(daily.loc[daily["observations"].eq(288), "utc_date"], utc=True).max()
        garch_steps = max(int((target_date.normalize() - last_full_day.normalize()).days), 1)
        garch_variance, risk, garch_origin = garch_forecast_and_risk(daily, settings.confidence_levels, garch_steps)
        summary["forecasts"]["garch"] = garch_variance
        summary["forecasts"] = {
            name: {"realized_variance": value, "realized_volatility": float(np.sqrt(max(value, 0)))}
            for name, value in summary["forecasts"].items()
        }
        summary["forecasts"]["garch"]["var_es"] = risk
        summary["garch_origin_time"] = garch_origin
        summary["garch_steps_ahead"] = garch_steps
        summary["data_as_of_utc"] = pd.to_datetime(candles["close_time"], utc=True).max().isoformat()
        summary["forecast_horizon_days"] = settings.forecast_horizon_days
        summary["target_definition"] = "next UTC day's realized variance from complete 5-minute returns"
        summary["risk_unit"] = "loss-positive daily return percent"
        summary["risk_method"] = "GARCH(1,1) conditional volatility with fitted Student-t innovations"

        predictions_path = output_dir / f"{symbol.lower()}_daily_rv_oos.parquet"
        temporary = predictions_path.with_suffix(".parquet.tmp")
        predictions.to_parquet(temporary, index=False)
        temporary.replace(predictions_path)
        metrics_by_symbol[symbol] = {
            "out_of_sample": summary | {"forecasts": None},
            "metrics": metrics,
            "prediction_file": str(predictions_path),
        }
        forecasts_by_symbol[symbol] = summary
        LOGGER.info("%s one-day forecasts saved; OOS test rows=%s", symbol, summary["test_rows"])

    generated_at = datetime.now(timezone.utc).isoformat()
    _write_json(metrics_dir / "volatility_backtest.json", {
        "generated_at_utc": generated_at,
        "horizon_days": settings.forecast_horizon_days,
        "target": "daily realized variance",
        "symbols": metrics_by_symbol,
    })
    _write_json(output_dir / "latest_forecasts.json", {
        "generated_at_utc": generated_at,
        "source": "Binance Spot 5-minute candles",
        "symbols": forecasts_by_symbol,
    })
    print(f"Wrote latest forecasts to {output_dir / 'latest_forecasts.json'}")
    print(f"Wrote out-of-sample metrics to {metrics_dir / 'volatility_backtest.json'}")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
