"""Prequential multi-horizon forecasts using only synchronized LOB samples."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from math import sqrt

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from app.models.lob_benchmarks import empirical_var_es
from app.models.lob_ml import make_lob_rv_regressors


LOB_FEATURES = (
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
    "past_rv_60",
    "past_rv_300",
    "past_rv_3600",
    "relative_spread_bps_std_60",
    "relative_spread_bps_std_300",
    "obi_qty_10_std_60",
    "obi_qty_10_std_300",
)
HAR_FEATURES = ("past_rv_60", "past_rv_300", "past_rv_3600")
HAR_FEATURE_INDICES = tuple(LOB_FEATURES.index(name) for name in HAR_FEATURES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _volatility_percent(rv: float | None) -> float | None:
    return sqrt(max(rv, 0.0)) * 100.0 if rv is not None and np.isfinite(rv) else None


class LOBOnlineForecaster:
    """Produce RV persistence and supervised LOB forecasts as labels mature.

    Training labels are formed only from complete, contiguous, non-overlapping
    windows. A reconnect clears the unfinished window but keeps mature history.
    """

    def __init__(
        self,
        symbol: str,
        horizons_seconds: tuple[int, ...] = (60, 300, 3600),
        sampling_interval_seconds: int = 1,
        minimum_train_examples: int = 60,
        maximum_train_examples: int = 500,
        refit_every_examples: int = 10,
    ) -> None:
        if not horizons_seconds or any(h < 1 or h % sampling_interval_seconds for h in horizons_seconds):
            raise ValueError("Forecast horizons must be positive multiples of the sampling interval")
        if sampling_interval_seconds < 1 or minimum_train_examples < 30 or maximum_train_examples < minimum_train_examples:
            raise ValueError("Invalid online forecasting settings")
        self.symbol = symbol.upper()
        self.horizons = tuple(sorted(set(horizons_seconds)))
        self.interval = sampling_interval_seconds
        self.minimum_train_examples = minimum_train_examples
        self.maximum_train_examples = maximum_train_examples
        self.refit_every_examples = max(1, refit_every_examples)
        self._segment: deque[dict] = deque(maxlen=max(self.horizons) // self.interval + 1)
        self._segment_index = -1
        self._training: dict[int, deque[dict]] = {
            horizon: deque(maxlen=maximum_train_examples) for horizon in self.horizons
        }
        self._pending: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
        self._scored: dict[int, dict[str, deque[tuple[float, float]]]] = {
            horizon: defaultdict(lambda: deque(maxlen=500)) for horizon in self.horizons
        }
        self._models: dict[int, dict[str, object]] = {horizon: {} for horizon in self.horizons}
        self._last_fit_count: dict[int, int] = {horizon: 0 for horizon in self.horizons}
        self._seen_examples: dict[int, int] = {horizon: 0 for horizon in self.horizons}
        self._fit_failures: dict[int, dict[str, str]] = {horizon: {} for horizon in self.horizons}
        self._latest: dict | None = None

    def mark_gap(self) -> None:
        """Prevent labels and rolling forecasts from crossing a feed gap."""
        self._segment.clear()
        self._segment_index = -1
        for pending in self._pending.values():
            pending.clear()
        self._latest = None

    def get_latest(self) -> dict:
        return self._latest or {
            "symbol": self.symbol,
            "status": "warming_up",
            "source": "Synchronized Binance Spot LOB mid-price and book features",
            "as_of_utc": None,
            "horizons": [self._empty_horizon(horizon) for horizon in self.horizons],
        }

    def update(self, feature: dict) -> dict:
        row = dict(feature)
        row["_feature_vector"] = {
            name: _number(feature.get(name)) for name in LOB_FEATURES
        }
        row["_mid_return"] = _number(feature.get("mid_return"))
        if row["_mid_return"] is None:
            # A first sample after startup/reconnect has no valid one-step return.
            self.mark_gap()
            self._segment.append(row)
            self._segment_index = 0
            return self.get_latest()

        self._segment.append(row)
        self._segment_index += 1
        outputs = []
        for horizon in self.horizons:
            count = horizon // self.interval
            if self._segment_index >= count:
                window = list(self._segment)[-count - 1 :]
                returns = [item.get("_mid_return") for item in window[1:]]
                if len(returns) == count and all(value is not None for value in returns):
                    actual_rv = float(np.sum(np.square(returns)))
                    actual_return = float(np.sum(returns))
                    origin = window[0]
                    if self._segment_index > 0 and self._segment_index % count == 0:
                        self._mature_label(
                            horizon, origin, actual_rv, actual_return,
                            row.get("sample_time_utc"),
                        )

            outputs.append(self._forecast_horizon(horizon, row))

        self._latest = {
            "symbol": self.symbol,
            "status": "ready" if any(item["status"] == "ready" for item in outputs) else "warming_up",
            "source": "Synchronized Binance Spot LOB mid-price and book features",
            "as_of_utc": row.get("sample_time_utc") or _utc_now(),
            "sampling_interval_seconds": self.interval,
            "horizons": outputs,
        }
        return self._latest

    def _mature_label(
        self,
        horizon: int,
        origin: dict,
        actual_rv: float,
        actual_return: float,
        target_end_utc: str | None,
    ) -> None:
        origin_index = self._segment_index - horizon // self.interval
        pending = self._pending[horizon].pop(origin_index, {})
        for model_name, forecast in pending.items():
            if forecast is not None and np.isfinite(forecast) and forecast > 0:
                self._scored[horizon][model_name].append((actual_rv, forecast))

        sample = {
            "features": origin["_feature_vector"],
            "target_rv": actual_rv,
            "target_return": actual_return,
            "target_end_utc": target_end_utc,
        }
        self._training[horizon].append(sample)
        self._seen_examples[horizon] += 1
        n = len(self._training[horizon])
        if n >= self.minimum_train_examples and (
            not self._models[horizon]
            or self._seen_examples[horizon] - self._last_fit_count[horizon] >= self.refit_every_examples
        ):
            self._fit(horizon)

    def _fit(self, horizon: int) -> None:
        rows = list(self._training[horizon])
        x = np.asarray([
            [row["features"].get(name, np.nan) for name in LOB_FEATURES]
            for row in rows
        ], dtype=float)
        y = np.asarray([row["target_rv"] for row in rows], dtype=float)
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < self.minimum_train_examples:
            return
        fitted = {}
        har = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        try:
            har.fit(x[valid][:, HAR_FEATURE_INDICES], np.log(y[valid]))
            fitted["har_rv"] = har
            self._fit_failures[horizon].pop("har_rv", None)
        except Exception as exc:
            self._fit_failures[horizon]["har_rv"] = f"{type(exc).__name__}: {exc}"
        for name, estimator in make_lob_rv_regressors().items():
            try:
                estimator.fit(x[valid], np.log(y[valid]))
                fitted[name] = estimator
                self._fit_failures[horizon].pop(name, None)
            except Exception as exc:  # Keep a working model if its peer cannot fit.
                self._fit_failures[horizon][name] = f"{type(exc).__name__}: {exc}"
        if fitted:
            self._models[horizon] = fitted
            self._last_fit_count[horizon] = self._seen_examples[horizon]

    def _forecast_horizon(self, horizon: int, current: dict) -> dict:
        count = horizon // self.interval
        window = list(self._segment)[-count - 1 :]
        returns = [item.get("_mid_return") for item in window[1:]] if len(window) > count else []
        complete = len(returns) == count and all(value is not None for value in returns)
        persistence = float(np.sum(np.square(returns))) if complete else None
        models = []

        if persistence is not None:
            models.append(self._model_row(
                "RV persistence", "traditional", persistence, horizon, len(self._training[horizon])
            ))
        else:
            models.append(self._model_row(
                "RV persistence", "traditional", None, horizon, len(self._training[horizon]),
                status="warming_up",
            ))

        current_x = np.asarray([[current["_feature_vector"].get(name, np.nan) for name in LOB_FEATURES]], dtype=float)
        for key, label, family in (
            ("har_rv", "HAR-RV · truyền thống", "traditional"),
            ("ridge", "Ridge · LOB", "lob_model"),
            ("hist_gradient_boosting", "Gradient Boosting · LOB", "lob_model"),
        ):
            estimator = self._models[horizon].get(key)
            predicted = None
            if estimator is not None:
                try:
                    model_x = current_x[:, HAR_FEATURE_INDICES] if key == "har_rv" else current_x
                    predicted = max(float(np.exp(np.clip(estimator.predict(model_x)[0], -700, 700))), np.finfo(float).tiny)
                except Exception as exc:
                    self._fit_failures[horizon][key] = f"{type(exc).__name__}: {exc}"
            models.append(self._model_row(
                label,
                family,
                predicted,
                horizon,
                len(self._training[horizon]),
                status="warming_up" if predicted is None else "ready",
                model_key=key,
            ))

        if persistence is not None and self._segment_index >= 0 and self._segment_index % max(count, 1) == 0:
            self._pending[horizon][self._segment_index] = {
                "persistence": persistence,
                "har_rv": next((row["forecast_rv"] for row in models if row["model_key"] == "har_rv"), None),
                "ridge": next((row["forecast_rv"] for row in models if row["model_key"] == "ridge"), None),
                "hist_gradient_boosting": next((row["forecast_rv"] for row in models if row["model_key"] == "hist_gradient_boosting"), None),
            }

        matured = list(self._training[horizon])
        returns_history = [row["target_return"] for row in matured if np.isfinite(row["target_return"])]
        tail = {}
        for confidence in (0.95, 0.99):
            minimum = 200 if confidence == 0.95 else 1000
            if len(returns_history) >= minimum:
                var, es = empirical_var_es(returns_history, confidence)
                if np.isfinite(var) and np.isfinite(es):
                    tail[str(confidence)] = {
                        "var_loss_percent": float(var * 100),
                        "es_loss_percent": float(es * 100),
                        "observations": len(returns_history),
                        "method": "historical non-overlapping LOB mid-price returns",
                    }

        return {
            "horizon_seconds": horizon,
            "label": self._horizon_label(horizon),
            "status": "ready" if persistence is not None else "warming_up",
            "target": "Future realized variance of 1-second mid-price returns",
            "forecast_volatility_unit": "standard deviation of cumulative mid-price return, percent",
            "valid_samples_in_window": len(returns) if complete else len(self._segment),
            "required_samples": count + 1,
            "training_examples": len(matured),
            "minimum_train_examples": self.minimum_train_examples,
            "models": models,
            "tail_risk": tail,
            "out_of_sample": self._oos_summary(horizon),
            "fit_failures": dict(self._fit_failures[horizon]),
        }

    def _model_row(
        self,
        name: str,
        family: str,
        forecast_rv: float | None,
        horizon: int,
        training_examples: int,
        status: str | None = None,
        model_key: str | None = None,
    ) -> dict:
        key = model_key or ("persistence" if family == "traditional" else name)
        return {
            "name": name,
            "family": family,
            "model_key": key,
            "status": status or ("ready" if forecast_rv is not None else "warming_up"),
            "forecast_rv": forecast_rv,
            "forecast_volatility_percent": _volatility_percent(forecast_rv),
            "training_examples": training_examples,
            "minimum_train_examples": self.minimum_train_examples if family == "lob_model" else 0,
            **self._oos_for_model(horizon, key),
        }

    def _oos_for_model(self, horizon: int, model: str) -> dict:
        pairs = list(self._scored[horizon].get(model, ()))
        if not pairs:
            return {"oos_windows": 0, "oos_qlike": None}
        actual = np.asarray([item[0] for item in pairs], dtype=float)
        forecast = np.asarray([item[1] for item in pairs], dtype=float)
        valid = np.isfinite(actual) & np.isfinite(forecast) & (forecast > 0)
        qlike = np.log(forecast[valid]) + actual[valid] / forecast[valid]
        return {"oos_windows": int(valid.sum()), "oos_qlike": float(np.mean(qlike)) if valid.any() else None}

    def _oos_summary(self, horizon: int) -> dict:
        return {
            name: self._oos_for_model(horizon, name)
            for name in ("persistence", "har_rv", "ridge", "hist_gradient_boosting")
        }

    @staticmethod
    def _horizon_label(horizon: int) -> str:
        if horizon < 3600:
            return f"{horizon // 60} phút"
        hours, remainder = divmod(horizon, 3600)
        return f"{hours} giờ" if remainder == 0 else f"{hours} giờ {remainder // 60} phút"

    def _empty_horizon(self, horizon: int) -> dict:
        models = [
            self._model_row("RV persistence", "traditional", None, horizon, 0, status="warming_up"),
            self._model_row("HAR-RV · truyền thống", "traditional", None, horizon, 0, status="warming_up", model_key="har_rv"),
            self._model_row("Ridge · LOB", "lob_model", None, horizon, 0, status="warming_up", model_key="ridge"),
            self._model_row(
                "Gradient Boosting · LOB", "lob_model", None, horizon, 0,
                status="warming_up", model_key="hist_gradient_boosting",
            ),
        ]
        return {
            "horizon_seconds": horizon,
            "label": self._horizon_label(horizon),
            "status": "warming_up",
            "target": "Future realized variance of 1-second mid-price returns",
            "forecast_volatility_unit": "standard deviation of cumulative mid-price return, percent",
            "valid_samples_in_window": 0,
            "required_samples": horizon // self.interval + 1,
            "training_examples": 0,
            "minimum_train_examples": self.minimum_train_examples,
            "models": models,
            "tail_risk": {},
            "out_of_sample": self._oos_summary(horizon),
            "fit_failures": dict(self._fit_failures[horizon]),
        }
