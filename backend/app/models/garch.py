"""GARCH model adapter; the optional arch package is loaded only when used."""


def fit_garch(returns):
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("Install the optional 'arch' package to run GARCH models") from exc
    model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    return model.fit(disp="off")
