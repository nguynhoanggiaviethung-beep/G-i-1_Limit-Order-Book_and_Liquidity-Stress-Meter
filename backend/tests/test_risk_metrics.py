import pandas as pd
import numpy as np

from app.features.risk_metrics import historical_var_es
from app.risk.var_es import fit_skewt_params, param_var_es


def test_historical_var_and_expected_shortfall_use_loss_positive_convention():
    result = historical_var_es(pd.Series([0.01, 0.0, -0.02, -0.04]), confidence=0.75)
    assert result["var"] > 0
    assert result["expected_shortfall"] >= result["var"]


def test_skewt_fit_standardizes_sample_and_calibrates_known_95pct_tail():
    from arch.univariate import SkewStudent

    sample = SkewStudent(seed=42).simulate([8.0, 0.35])(50_000)
    params = fit_skewt_params(sample * 2.5 + 1.7)
    var, es = param_var_es(1.0, "skewt", 0.95, params)
    standardized = (sample - sample.mean()) / sample.std(ddof=0)
    exception_rate = float(np.mean(-standardized > var))

    assert 2.0 < params["eta"] < 200.0
    assert -1.0 < params["lambda"] < 1.0
    assert abs(exception_rate - 0.05) <= 0.01
    assert es > var > 0
