import pandas as pd
import numpy as np

from app.features.risk_metrics import historical_var_es
from app.risk.var_es import fit_skewt_params, param_var_es


def test_historical_var_and_expected_shortfall_use_loss_positive_convention():
    result = historical_var_es(pd.Series([0.01, 0.0, -0.02, -0.04]), confidence=0.75)
    assert result["var"] > 0
    assert result["expected_shortfall"] >= result["var"]


def test_skewt_fit_retains_location_and_scale_and_calibrates_known_95pct_tail():
    from arch.univariate import SkewStudent

    sample = SkewStudent(seed=42).simulate([8.0, 0.35])(10_000)
    transformed_sample = sample * 2.5 + 1.7
    params = fit_skewt_params(transformed_sample)
    var, es = param_var_es(1.0, "skewt", 0.95, params)
    exception_rate = float(np.mean(-transformed_sample > var))

    assert 2.0 < params["eta"] < 200.0
    assert -1.0 < params["lambda"] < 1.0
    assert abs(params["location"] - 1.7) < 0.03
    assert abs(params["scale"] - 2.5) < 0.03
    assert abs(exception_rate - 0.05) <= 0.01
    assert es > var > 0


def test_skewt_fit_retries_with_fallback_start_after_optimizer_failure(monkeypatch):
    from types import SimpleNamespace
    from arch.univariate import SkewStudent
    from app.risk import var_es

    sample = SkewStudent(seed=7).simulate([8.0, -0.25])(1_000)
    calls = []

    def fake_minimize(objective, start, **kwargs):
        calls.append(np.asarray(start, dtype=float))
        if len(calls) == 1:
            return SimpleNamespace(success=False, fun=np.inf, x=np.asarray(start), message="ABNORMAL")
        return SimpleNamespace(success=True, fun=1.0, x=np.asarray([7.0, -0.2]), message="ok")

    monkeypatch.setattr(var_es, "minimize", fake_minimize)
    params = var_es.fit_skewt_params(sample)

    assert len(calls) == 2
    assert params["eta"] == 7.0
    assert params["lambda"] == -0.2
