"""Daily-origin, multi-day volatility forecasts from synchronized LOB data only."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from math import sqrt
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from app.models.lob_ml import make_lob_rv_regressors


LOB_DAILY_FEATURES = (
    "relative_spread_bps",
    "microprice_minus_mid_bps",
    "obi_qty_5",
    "obi_qty_10",
    "obi_qty_20",
    "bid_depth_qty_5",
    "ask_depth_qty_5",
    "bid_depth_qty_10",
    "ask_depth_qty_10",
    "bid_depth_qty_20",
    "ask_depth_qty_20",
)
MODEL_FEATURES = ("past_rv_1d", "past_rv_7d", "past_rv_30d", *LOB_DAILY_FEATURES)
HAR_FEATURES = ("past_rv_1d", "past_rv_7d", "past_rv_30d")
HAR_INDICES = tuple(MODEL_FEATURES.index(name) for name in HAR_FEATURES)


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _rv_to_vol_percent(rv: float | None) -> float | None:
    return sqrt(max(rv, 0.0)) * 100.0 if rv is not None and np.isfinite(rv) else None


class LOBLongHorizonForecaster:
    """Forecast 1/7/30-day future realized variance at completed UTC-day origins.

    A UTC day is admitted only when its LOB samples meet the coverage threshold
    and there are no material clock gaps. Forecast labels mature only after each
    future UTC day is complete. OOS scores use daily origins and are therefore
    descriptive for the overlapping 7/30-day windows.
    """

    def __init__(
        self,
        symbol: str,
        horizons_days: tuple[int, ...] = (1, 7, 30),
        sampling_interval_seconds: int = 1,
        minimum_train_examples: int = 30,
        maximum_train_examples: int = 500,
        minimum_daily_coverage: float = 0.995,
    ) -> None:
        if not horizons_days or any(value not in (1, 7, 30) for value in horizons_days):
            raise ValueError("Long LOB horizons must be selected from 1, 7, and 30 days")
        if sampling_interval_seconds < 1 or minimum_train_examples < 30 or maximum_train_examples < minimum_train_examples:
            raise ValueError("Invalid long-horizon forecasting settings")
        if not 0.9 <= minimum_daily_coverage <= 1:
            raise ValueError("minimum_daily_coverage must be in [0.9, 1]")
        self.symbol = symbol.upper()
        self.horizons = tuple(sorted(set(horizons_days)))
        self.interval = sampling_interval_seconds
        self.minimum_train_examples = minimum_train_examples
        self.maximum_train_examples = maximum_train_examples
        self.minimum_daily_coverage = minimum_daily_coverage
        self._daily_segment: deque[dict] = deque(maxlen=1000)
        self._training = {h: deque(maxlen=maximum_train_examples) for h in self.horizons}
        self._pending: list[dict] = []
        self._scored = {h: defaultdict(lambda: deque(maxlen=500)) for h in self.horizons}
        self._models: dict[int, dict[str, object]] = {h: {} for h in self.horizons}
        self._fit_failures: dict[int, dict[str, str]] = {h: {} for h in self.horizons}
        self._last_fit_count = {h: 0 for h in self.horizons}
        self._last_complete_date: date | None = None
        self._day: date | None = None
        self._day_invalid = False
        self._day_accumulator: dict | None = None
        self._summary_file: Path | None = None
        self._latest: dict | None = None

    def load_history(self, data_dir: Path) -> int:
        """Replay this collector's compact daily LOB summaries after restart."""
        self._summary_file = Path(data_dir) / "features" / "lob_daily" / f"symbol={self.symbol}" / "daily_summary.jsonl"
        if not self._summary_file.is_file():
            self._refresh_latest()
            return 0
        loaded = 0
        today = datetime.now(timezone.utc).date()
        try:
            records = []
            with self._summary_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Ignore a truncated final write; prior daily rows remain usable.
                        continue
            for row in sorted(records, key=lambda item: item.get("date_utc", "")):
                day = date.fromisoformat(str(row["date_utc"]))
                if day >= today or day <= (self._last_complete_date or date.min):
                    continue
                if not self._valid_summary(row):
                    self._break_segment()
                    self._last_complete_date = day
                    continue
                self._process_complete_day(row, persist=False)
                loaded += 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._break_segment()
        self._refresh_latest()
        return loaded

    def mark_gap(self) -> None:
        """Discard a partial day and pending targets so forecasts never cross feed gaps."""
        if self._day is not None:
            self._day_invalid = True
            self._day_accumulator = None
        else:
            self._day = datetime.now(timezone.utc).date()
            self._day_invalid = True
            self._day_accumulator = None
        self._pending.clear()

    def update(self, feature: dict) -> dict:
        stamp = feature.get("sample_time_utc")
        if not stamp:
            return self.get_latest()
        try:
            timestamp = _utc_datetime(str(stamp))
        except (TypeError, ValueError):
            self.mark_gap()
            return self.get_latest()
        sample_day = timestamp.date()
        if self._day is None:
            self._start_day(sample_day)
        elif sample_day < self._day:
            self.mark_gap()
            return self.get_latest()
        elif sample_day > self._day:
            self._finalize_day()
            self._start_day(sample_day)
        if self._day_invalid:
            return self.get_latest()
        accumulator = self._day_accumulator
        if accumulator is None:
            return self.get_latest()
        previous = accumulator["last_sample"]
        if previous is not None:
            gap = (timestamp - previous).total_seconds()
            if gap <= 0 or gap > max(5 * self.interval, 5):
                self._day_invalid = True
                self._day_accumulator = None
                self._break_segment()
                return self.get_latest()
        accumulator["last_sample"] = timestamp
        accumulator["sample_count"] += 1
        rv_return = _finite(feature.get("mid_return"))
        if rv_return is not None:
            accumulator["return_count"] += 1
            accumulator["daily_rv"] += rv_return * rv_return
            accumulator["daily_return"] += rv_return
        for name in LOB_DAILY_FEATURES:
            value = _finite(feature.get(name))
            if value is not None:
                accumulator["feature_sum"][name] += value
                accumulator["feature_count"][name] += 1
        return self.get_latest()

    def get_latest(self) -> dict:
        self._refresh_latest()
        return self._latest or {
            "symbol": self.symbol,
            "status": "warming_up",
            "source": "Daily aggregates of Binance Spot LOB mid-price and book features only",
            "data_as_of_utc": None,
            "horizons": [self._empty_horizon(h) for h in self.horizons],
            "note": "Long-horizon forecasts appear after complete, quality-checked UTC LOB days accumulate.",
        }

    def _start_day(self, sample_day: date) -> None:
        if self._last_complete_date is not None and sample_day != self._last_complete_date + timedelta(days=1):
            self._break_segment()
        self._day = sample_day
        self._day_invalid = False
        self._day_accumulator = {
            "sample_count": 0,
            "return_count": 0,
            "daily_rv": 0.0,
            "daily_return": 0.0,
            "feature_sum": defaultdict(float),
            "feature_count": defaultdict(int),
            "last_sample": None,
        }

    def _finalize_day(self) -> None:
        if self._day is None:
            return
        accumulator = self._day_accumulator
        if not self._day_invalid and accumulator is not None:
            expected = 86400 // self.interval
            coverage = accumulator["sample_count"] / max(expected, 1)
            minimum_returns = max(1, int(expected * self.minimum_daily_coverage) - 1)
            if coverage >= self.minimum_daily_coverage and accumulator["return_count"] >= minimum_returns:
                row = {
                    "date_utc": self._day.isoformat(),
                    "sample_count": int(accumulator["sample_count"]),
                    "coverage": float(min(1.0, coverage)),
                    "daily_rv": float(max(accumulator["daily_rv"], np.finfo(float).tiny)),
                    "daily_return": float(accumulator["daily_return"]),
                    "lob_features": {
                        name: (
                            float(accumulator["feature_sum"][name] / accumulator["feature_count"][name])
                            if accumulator["feature_count"][name]
                            else None
                        )
                        for name in LOB_DAILY_FEATURES
                    },
                }
                self._process_complete_day(row, persist=True)
            else:
                self._break_segment()
        else:
            self._break_segment()
        self._day = None
        self._day_accumulator = None
        self._day_invalid = False

    def _process_complete_day(self, row: dict, persist: bool) -> None:
        day = date.fromisoformat(str(row["date_utc"]))
        if self._last_complete_date is not None and day != self._last_complete_date + timedelta(days=1):
            self._break_segment()
        daily = {
            "date_utc": day.isoformat(),
            "daily_rv": max(float(row["daily_rv"]), np.finfo(float).tiny),
            "daily_return": float(row.get("daily_return", 0.0)),
            "sample_count": int(row.get("sample_count", 0)),
            "coverage": float(row.get("coverage", 1.0)),
            "lob_features": {
                name: _finite((row.get("lob_features") or {}).get(name))
                for name in LOB_DAILY_FEATURES
            },
        }
        self._daily_segment.append(daily)
        self._last_complete_date = day
        self._mature_pending(daily["daily_rv"])
        self._issue_forecasts(daily)
        if persist:
            self._append_summary(daily)
        self._refresh_latest()

    def _issue_forecasts(self, current: dict) -> None:
        history = list(self._daily_segment)
        rv = np.asarray([item["daily_rv"] for item in history], dtype=float)
        feature_vector = {
            "past_rv_1d": float(rv[-1]),
            "past_rv_7d": float(np.mean(rv[-7:])),
            "past_rv_30d": float(np.mean(rv[-30:])),
        }
        feature_vector.update(current["lob_features"])
        x = np.asarray([[feature_vector.get(name, np.nan) for name in MODEL_FEATURES]], dtype=float)
        for horizon in self.horizons:
            if len(history) < horizon:
                continue
            baseline = max(float(np.mean(rv[-min(len(rv), 30):]) * horizon), np.finfo(float).tiny)
            forecasts: dict[str, float | None] = {"rolling_mean": baseline}
            for key in ("har_rv", "ridge", "hist_gradient_boosting"):
                estimator = self._models[horizon].get(key)
                prediction = None
                if estimator is not None:
                    try:
                        model_x = x[:, HAR_INDICES] if key == "har_rv" else x
                        prediction = max(float(np.exp(np.clip(estimator.predict(model_x)[0], -700, 700))), np.finfo(float).tiny)
                    except Exception as exc:
                        self._fit_failures[horizon][key] = f"{type(exc).__name__}: {exc}"
                forecasts[key] = prediction
            self._pending.append({
                "horizon": horizon,
                "remaining_days": horizon,
                "actual_rv": 0.0,
                "features": feature_vector,
                "forecasts": forecasts,
                "origin_date_utc": current["date_utc"],
            })

    def _mature_pending(self, current_daily_rv: float) -> None:
        still_pending = []
        for pending in self._pending:
            pending["actual_rv"] += current_daily_rv
            pending["remaining_days"] -= 1
            if pending["remaining_days"] > 0:
                still_pending.append(pending)
                continue
            horizon = pending["horizon"]
            actual = max(float(pending["actual_rv"]), np.finfo(float).tiny)
            for name, forecast in pending["forecasts"].items():
                if forecast is not None and np.isfinite(forecast) and forecast > 0:
                    self._scored[horizon][name].append((actual, float(forecast)))
            sample = {"features": pending["features"], "target_rv": actual}
            self._training[horizon].append(sample)
            n = len(self._training[horizon])
            if n >= self.minimum_train_examples and (
                not self._models[horizon] or n - self._last_fit_count[horizon] >= 5
            ):
                self._fit(horizon)
        self._pending = still_pending

    def _fit(self, horizon: int) -> None:
        rows = list(self._training[horizon])
        x = np.asarray([[row["features"].get(name, np.nan) for name in MODEL_FEATURES] for row in rows], dtype=float)
        y = np.asarray([row["target_rv"] for row in rows], dtype=float)
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < self.minimum_train_examples:
            return
        fitted: dict[str, object] = {}
        try:
            har = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
            har.fit(x[valid][:, HAR_INDICES], np.log(y[valid]))
            fitted["har_rv"] = har
            self._fit_failures[horizon].pop("har_rv", None)
        except Exception as exc:
            self._fit_failures[horizon]["har_rv"] = f"{type(exc).__name__}: {exc}"
        for name, estimator in make_lob_rv_regressors().items():
            try:
                estimator.fit(x[valid], np.log(y[valid]))
                fitted[name] = estimator
                self._fit_failures[horizon].pop(name, None)
            except Exception as exc:
                self._fit_failures[horizon][name] = f"{type(exc).__name__}: {exc}"
        if fitted:
            self._models[horizon] = fitted
            self._last_fit_count[horizon] = len(rows)

    def _append_summary(self, row: dict) -> None:
        if self._summary_file is None:
            return
        self._summary_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            **row,
        }
        with self._summary_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _valid_summary(self, row: dict) -> bool:
        expected = 86400 // self.interval
        return (
            row.get("schema_version") == 1
            and float(row.get("coverage", 0)) >= self.minimum_daily_coverage
            and int(row.get("sample_count", 0)) >= int(expected * self.minimum_daily_coverage)
            and _finite(row.get("daily_rv")) is not None
        )

    def _break_segment(self) -> None:
        self._daily_segment.clear()
        self._pending.clear()

    def _refresh_latest(self) -> None:
        daily = list(self._daily_segment)
        outputs = [self._horizon_output(h, daily) for h in self.horizons]
        self._latest = {
            "symbol": self.symbol,
            "status": "ready" if any(item["status"] == "ready" for item in outputs) else "warming_up",
            "source": "Daily aggregates of Binance Spot LOB mid-price and book features only",
            "data_as_of_utc": f"{self._last_complete_date.isoformat()}T23:59:59+00:00" if self._last_complete_date else None,
            "complete_consecutive_days": len(daily),
            "horizons": outputs,
            "note": "Dự báo tính từ ngày UTC LOB hoàn chỉnh gần nhất; cửa sổ OOS 7/30 ngày chồng lấn nên QLIKE chỉ mang tính mô tả.",
        }

    def _horizon_output(self, horizon: int, daily: list[dict]) -> dict:
        required_days = horizon
        eligible = len(daily) >= required_days
        training = list(self._training[horizon])
        models = []
        baseline = None
        if eligible:
            values = np.asarray([row["daily_rv"] for row in daily[-30:]], dtype=float)
            baseline = max(float(np.mean(values) * horizon), np.finfo(float).tiny)
        models.append(self._model_row(horizon, "rolling_mean", "RV rolling mean · truyền thống", "traditional", baseline, len(training), 0))
        if daily:
            history = daily[-1]
            rv = np.asarray([row["daily_rv"] for row in daily], dtype=float)
            vector = {
                "past_rv_1d": float(rv[-1]),
                "past_rv_7d": float(np.mean(rv[-7:])),
                "past_rv_30d": float(np.mean(rv[-30:])),
                **history["lob_features"],
            }
            x = np.asarray([[vector.get(name, np.nan) for name in MODEL_FEATURES]], dtype=float)
        else:
            x = None
        for key, label, family in (
            ("har_rv", "HAR-RV ngày · truyền thống", "traditional"),
            ("ridge", "Ridge · LOB", "lob_model"),
            ("hist_gradient_boosting", "Gradient Boosting · LOB", "lob_model"),
        ):
            prediction = None
            estimator = self._models[horizon].get(key)
            if eligible and estimator is not None and x is not None:
                try:
                    model_x = x[:, HAR_INDICES] if key == "har_rv" else x
                    prediction = max(float(np.exp(np.clip(estimator.predict(model_x)[0], -700, 700))), np.finfo(float).tiny)
                except Exception as exc:
                    self._fit_failures[horizon][key] = f"{type(exc).__name__}: {exc}"
            models.append(self._model_row(horizon, key, label, family, prediction, len(training), self.minimum_train_examples if family == "lob_model" else self.minimum_train_examples))
        return {
            "horizon_days": horizon,
            "label": f"{horizon} ngày",
            "status": "ready" if eligible else "warming_up",
            "target": "Tổng realized variance tương lai của mid-price suy ra từ LOB trong các ngày UTC hoàn chỉnh",
            "complete_days": len(daily),
            "required_days": required_days,
            "training_examples": len(training),
            "minimum_train_examples": self.minimum_train_examples,
            "overlapping_daily_origins": True,
            "models": models,
            "out_of_sample": {name: self._oos_for_model(horizon, name) for name in ("rolling_mean", "har_rv", "ridge", "hist_gradient_boosting")},
            "fit_failures": dict(self._fit_failures[horizon]),
        }

    def _model_row(self, horizon: int, key: str, name: str, family: str, forecast_rv: float | None, train_n: int, minimum: int) -> dict:
        score = self._oos_for_model(horizon, key)
        return {
            "model_key": key,
            "name": name,
            "family": family,
            "status": "ready" if forecast_rv is not None else "warming_up",
            "forecast_rv": forecast_rv,
            "forecast_volatility_percent": _rv_to_vol_percent(forecast_rv),
            "training_examples": train_n,
            "minimum_train_examples": minimum,
            **score,
        }

    def _oos_for_model(self, horizon: int, model: str) -> dict:
        pairs = list(self._scored[horizon].get(model, ()))
        if not pairs:
            return {"oos_windows": 0, "oos_qlike": None}
        actual = np.asarray([pair[0] for pair in pairs], dtype=float)
        forecast = np.asarray([pair[1] for pair in pairs], dtype=float)
        valid = np.isfinite(actual) & np.isfinite(forecast) & (forecast > 0)
        values = np.log(forecast[valid]) + actual[valid] / forecast[valid]
        return {"oos_windows": int(valid.sum()), "oos_qlike": float(np.mean(values)) if valid.any() else None}

    def _empty_horizon(self, horizon: int) -> dict:
        return self._horizon_output(horizon, [])
