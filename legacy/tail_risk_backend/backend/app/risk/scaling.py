"""Conversions from daily variance forecasts to risk and position scales."""

from __future__ import annotations

import numpy as np


def rv_to_sigma(rv_forecast: float, kappa: float = 1.0) -> float:
    if rv_forecast < 0 or kappa <= 0:
        raise ValueError("RV forecast must be non-negative and kappa positive")
    return float(kappa * np.sqrt(rv_forecast))


def volatility_target_weight(sigma_forecast: float, annual_target: float = 0.10, max_leverage: float = 2.0) -> float:
    if sigma_forecast <= 0 or annual_target <= 0 or max_leverage <= 0:
        raise ValueError("volatility and leverage limits must be positive")
    daily_target = annual_target / np.sqrt(365)
    return float(min(max_leverage, daily_target / sigma_forecast))


def risk_budget_weight(risk_forecast: float, budget_percent: float, max_leverage: float = 2.0) -> float:
    if risk_forecast <= 0 or budget_percent < 0 or max_leverage <= 0:
        raise ValueError("risk must be positive; budget non-negative; max leverage positive")
    return float(min(max_leverage, budget_percent / risk_forecast))


def dynamic_margin(es_forecast: float, buffer: float = 1.3, floor: float = 0.0) -> float:
    if es_forecast < 0 or buffer <= 0 or floor < 0:
        raise ValueError("ES and floor must be non-negative; buffer must be positive")
    return float(max(buffer * es_forecast, floor))
