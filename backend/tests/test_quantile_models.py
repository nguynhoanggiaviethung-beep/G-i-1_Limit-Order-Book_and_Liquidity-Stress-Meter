import numpy as np

from app.models.quantile import (
    empirical_expected_shortfall,
    fit_conditional_quantiles,
    predict_conditional_quantiles,
    rearrange_quantiles,
)


def test_quantile_rearrangement_removes_crossing():
    result = rearrange_quantiles({0.95: 2.0, 0.975: 1.0, 0.99: 3.0})
    assert result == {0.95: 2.0, 0.975: 2.0, 0.99: 3.0}


def test_linear_quantile_smoke_has_positive_es_above_var():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(400, 4))
    losses = 0.5 + x[:, 0] * 0.3 + rng.standard_t(df=6, size=len(x))
    levels = (0.95, 0.975, 0.99)
    model = fit_conditional_quantiles(
        "linear_qr", x, losses, levels, random_state=18, config={"alpha": 0.01, "min_train_rows": 100}
    )
    predicted = predict_conditional_quantiles("linear_qr", model, x[-1], levels)
    es = empirical_expected_shortfall(losses, predicted[0.95])
    assert all(np.isfinite(list(predicted.values())))
    assert predicted[0.95] <= predicted[0.975] <= predicted[0.99]
    assert es > predicted[0.95]


def test_qrf_quantile_smoke_has_monotone_predictions():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(250, 3))
    losses = 1.0 + x[:, 0] + rng.standard_t(df=5, size=len(x))
    levels = (0.95, 0.975, 0.99)
    model = fit_conditional_quantiles(
        "qrf", x, losses, levels, random_state=19,
        config={"n_estimators": 30, "min_samples_leaf": 5, "min_train_rows": 100},
    )
    predicted = predict_conditional_quantiles("qrf", model, x[-1], levels)
    assert predicted[0.95] <= predicted[0.975] <= predicted[0.99]
    assert all(np.isfinite(list(predicted.values())))
