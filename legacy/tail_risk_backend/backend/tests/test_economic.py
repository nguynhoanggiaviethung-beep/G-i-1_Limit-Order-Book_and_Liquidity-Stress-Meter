import numpy as np

from app.evaluation.economic import dynamic_margin_backtest


def test_dynamic_margin_reports_breaches_shortfalls_and_capital_cost():
    result = dynamic_margin_backtest(
        losses=np.array([1.0, 4.5, 5.5]),
        es_forecasts=np.array([1.0, 2.0, 4.0]),
        buffer=1.3,
        fixed_margin_percent=5.0,
        annual_capital_cost=0.05,
    )
    dynamic = result["dynamic_es_margin"]
    fixed = result["fixed_margin"]
    assert dynamic["margin_breaches"] == 2
    assert dynamic["max_shortfall_percent"] > 0
    assert dynamic["mean_daily_capital_cost_percent"] > 0
    assert fixed["margin_breaches"] == 1
    assert np.isfinite(dynamic["procyclicality_ratio"])
