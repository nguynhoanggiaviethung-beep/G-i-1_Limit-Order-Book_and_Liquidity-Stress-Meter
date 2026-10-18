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
    original = losses.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if original.shape[0] < 20 or original.shape[1] < 2:
        raise ValueError("MCS needs >=20 aligned rows and >=2 model loss columns")
    unique_columns: list[str] = []
    duplicate_map: dict[str, str] = {}
    for column in original.columns:
        duplicate = next((name for name in unique_columns if original[column].equals(original[name])), None)
        if duplicate is None:
            unique_columns.append(column)
        else:
            duplicate_map[str(column)] = str(duplicate)
    data = original[unique_columns]
    if data.shape[1] == 1:
        return pd.DataFrame([
            {"model": str(column), "included": True, "pvalue": 1.0, "status": "identical_loss_series"}
            for column in original.columns
        ])
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
    included_values = procedure.included
    if isinstance(included_values, (pd.Series, pd.DataFrame)):
        included_values = included_values.index
    included = set(map(str, included_values))
    pvalues = procedure.pvalues
    if isinstance(pvalues, pd.DataFrame):
        pvalue_column = next((column for column in pvalues.columns if "Pvalue" in str(column)), pvalues.columns[-1])
        pvalue_map = pvalues[pvalue_column].to_dict()
    else:
        pvalue_map = dict(pvalues)
    rows = []
    for name in original.columns:
        source = duplicate_map.get(str(name), str(name))
        value = pvalue_map.get(source, pvalue_map.get(name, float("nan")))
        rows.append({
            "model": str(name),
            "included": source in included,
            "pvalue": float(value),
            "status": "identical_loss_series" if str(name) in duplicate_map else "ok",
        })
    return pd.DataFrame(rows).sort_values(["included", "pvalue"], ascending=[False, False]).reset_index(drop=True)
