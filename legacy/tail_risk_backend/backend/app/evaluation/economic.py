"""Simple, cost-aware economic evaluations for research forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _maximum_drawdown(returns_percent: np.ndarray) -> float:
    wealth = np.cumprod(1 + returns_percent / 100)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / peak - 1
    return float(np.min(drawdown)) if drawdown.size else float("nan")


def _summary(returns_percent: np.ndarray, weights: np.ndarray, gamma: float) -> dict:
    returns_percent = returns_percent[np.isfinite(returns_percent)]
    if not len(returns_percent):
        return {}
    std = float(np.std(returns_percent, ddof=1)) if len(returns_percent) > 1 else 0.0
    sharpe = float(np.mean(returns_percent) / std * np.sqrt(365)) if std > 0 else float("nan")
    certainty_equivalent = float(np.mean(returns_percent) - 0.5 * gamma * np.var(returns_percent, ddof=1))
    return {
        "n_obs": int(len(returns_percent)),
        "mean_daily_return_percent": float(np.mean(returns_percent)),
        "annualized_sharpe": sharpe,
        "realized_volatility_annualized_percent": float(std * np.sqrt(365)),
        "maximum_drawdown": _maximum_drawdown(returns_percent),
        "certainty_equivalent_daily_percent": certainty_equivalent,
        "mean_absolute_turnover": float(np.mean(np.abs(np.diff(weights)))) if len(weights) > 1 else 0.0,
    }


def volatility_targeting(
    daily_returns,
    sigma_forecast,
    annual_target: float = 0.10,
    max_leverage: float = 2.0,
    fee_bps: float = 10.0,
    risk_aversion: float = 3.0,
) -> dict:
    """Backtest a daily volatility-target strategy after proportional fees."""
    returns = np.asarray(daily_returns, dtype=float)
    sigma = np.asarray(sigma_forecast, dtype=float)
    if returns.shape != sigma.shape:
        raise ValueError("daily returns and sigma forecasts must align")
    valid = np.isfinite(returns) & np.isfinite(sigma) & (sigma > 0)
    returns, sigma = returns[valid], sigma[valid]
    if not len(returns):
        raise ValueError("No valid aligned returns and forecasts")
    daily_target_percent = annual_target * 100 / np.sqrt(365)
    weights = np.minimum(max_leverage, daily_target_percent / sigma)
    previous = np.r_[0.0, weights[:-1]]
    turnover = np.abs(weights - previous)
    costs_percent = turnover * fee_bps / 100
    strategy_returns = weights * returns - costs_percent
    buy_hold_cost = np.zeros_like(returns)
    buy_hold_cost[0] = fee_bps / 100
    buy_hold = returns - buy_hold_cost
    strategy_stats = _summary(strategy_returns, weights, risk_aversion)
    hold_stats = _summary(buy_hold, np.ones_like(buy_hold), risk_aversion)
    return {
        "strategy": strategy_stats,
        "buy_and_hold": hold_stats,
        "utility_fee_vs_buy_and_hold_daily_percent": (
            strategy_stats["certainty_equivalent_daily_percent"]
            - hold_stats["certainty_equivalent_daily_percent"]
        ),
        "assumptions": {
            "annual_volatility_target": annual_target,
            "max_leverage": max_leverage,
            "fee_basis_points_per_turnover": fee_bps,
            "risk_aversion": risk_aversion,
        },
    }


def risk_budget_backtest(losses, var_forecasts, budget_percent: float, max_leverage: float = 2.0) -> dict:
    losses = np.asarray(losses, dtype=float)
    var = np.asarray(var_forecasts, dtype=float)
    if losses.shape != var.shape:
        raise ValueError("losses and VaR forecasts must align")
    valid = np.isfinite(losses) & np.isfinite(var) & (var > 0)
    losses, var = losses[valid], var[valid]
    weights = np.minimum(max_leverage, budget_percent / var)
    breaches = weights * losses > budget_percent
    return {
        "n_obs": int(len(losses)),
        "budget_percent": float(budget_percent),
        "breaches": int(breaches.sum()),
        "breach_rate": float(breaches.mean()) if len(breaches) else float("nan"),
        "mean_position_weight": float(weights.mean()) if len(weights) else float("nan"),
    }


def dynamic_margin_backtest(
    losses,
    es_forecasts,
    buffer: float = 1.3,
    fixed_margin_percent: float = 5.0,
    annual_capital_cost: float = 0.05,
    floor: float = 0.0,
) -> dict:
    """Compare ES-linked collateral with a fixed notional margin.

    Losses, expected shortfall, and margins use positive percentage points of
    notional. The annual capital charge is a financing-cost assumption, not a
    trading fee or exchange liquidation rule.
    """
    actual = np.asarray(losses, dtype=float)
    es = np.asarray(es_forecasts, dtype=float)
    if actual.shape != es.shape:
        raise ValueError("losses and ES forecasts must align")
    if buffer <= 0 or fixed_margin_percent < 0 or annual_capital_cost < 0 or floor < 0:
        raise ValueError("buffer must be positive; margin, annual cost, and floor non-negative")
    valid = np.isfinite(actual) & np.isfinite(es) & (es >= 0)
    actual, es = actual[valid], es[valid]
    if not len(actual):
        raise ValueError("No valid aligned loss and ES observations")

    dynamic = np.maximum(buffer * es, floor)
    fixed = np.full(len(actual), fixed_margin_percent, dtype=float)

    def score(margin: np.ndarray) -> dict:
        shortfall = np.maximum(actual - margin, 0.0)
        changes = np.abs(np.diff(margin))
        mean_margin = float(np.mean(margin))
        return {
            "n_obs": int(len(actual)),
            "mean_margin_percent": mean_margin,
            "max_margin_percent": float(np.max(margin)),
            "margin_breaches": int(np.sum(actual > margin)),
            "breach_rate": float(np.mean(actual > margin)),
            "mean_shortfall_percent": float(np.mean(shortfall)),
            "max_shortfall_percent": float(np.max(shortfall)),
            "mean_daily_capital_cost_percent": float(mean_margin * annual_capital_cost / 365.0),
            "mean_absolute_margin_change_percent": float(np.mean(changes)) if changes.size else 0.0,
            "procyclicality_ratio": float(np.mean(changes) / max(mean_margin, np.finfo(float).tiny)) if changes.size else 0.0,
        }

    return {
        "dynamic_es_margin": score(dynamic),
        "fixed_margin": score(fixed),
        "assumptions": {
            "es_buffer": float(buffer),
            "fixed_margin_percent_of_notional": float(fixed_margin_percent),
            "annual_capital_cost": float(annual_capital_cost),
            "margin_floor_percent": float(floor),
        },
    }
