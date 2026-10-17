"""Model Confidence Set adapter for aligned out-of-sample loss columns."""

from __future__ import annotations

import pandas as pd


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    bootstrap: int = 5000,
    block: int = 5,
    seed: int = 20260930,
) -> pd.DataFrame:
    """Return MCS inclusion and p-values using arch's stationary bootstrap."""
    data = losses.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if data.shape[0] < 20 or data.shape[1] < 2:
        raise ValueError("MCS needs >=20 aligned rows and >=2 model loss columns")
    try:
        from arch.bootstrap import MCS
    except ImportError as exc:
        raise RuntimeError("Install arch to run Model Confidence Set inference") from exc
    procedure = MCS(
        data,
        size=alpha,
        reps=bootstrap,
        block_size=block,
        bootstrap="stationary",
        seed=seed,
    )
    procedure.compute()
    included = set(procedure.included.index.astype(str))
    pvalues = procedure.pvalues
    if isinstance(pvalues, pd.DataFrame):
        pvalue_column = next((column for column in pvalues.columns if "Pvalue" in str(column)), pvalues.columns[-1])
        pvalue_map = pvalues[pvalue_column].to_dict()
    else:
        pvalue_map = dict(pvalues)
    return pd.DataFrame([
        {"model": str(name), "included": str(name) in included, "pvalue": float(pvalue_map.get(name, float("nan")))}
        for name in data.columns
    ]).sort_values(["included", "pvalue"], ascending=[False, False]).reset_index(drop=True)
