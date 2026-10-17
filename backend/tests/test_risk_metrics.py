import pandas as pd

from app.features.risk_metrics import historical_var_es


def test_historical_var_and_expected_shortfall_use_loss_positive_convention():
    result = historical_var_es(pd.Series([0.01, 0.0, -0.02, -0.04]), confidence=0.75)
    assert result["var"] > 0
    assert result["expected_shortfall"] >= result["var"]
