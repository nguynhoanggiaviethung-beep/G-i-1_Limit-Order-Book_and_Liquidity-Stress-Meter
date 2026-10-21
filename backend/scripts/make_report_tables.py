"""Create comparable OOS metric, DM, and optional MCS tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.research_config import load_research_config  # noqa: E402
from app.evaluation.dm_test import dm_test  # noqa: E402
from app.evaluation.mcs import model_confidence_set  # noqa: E402
from app.evaluation.volatility_metrics import qlike_loss, volatility_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--skip-mcs", action="store_true")
    args = parser.parse_args()
    research = load_research_config()
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    metrics_rows = []
    dm_rows = []
    mcs_rows = []
    report: dict = {"as_of_utc": research["as_of_utc"], "symbols": {}}

    for symbol in symbols:
        report["symbols"][symbol] = {}
        for horizon in horizons:
            path = ROOT / "results" / "forecasts" / f"{symbol.lower()}_h{horizon}_walkforward.parquet"
            if not path.exists():
                continue
            rows = pd.read_parquet(path)
            rows = rows.loc[
                rows["status"].eq("ok")
                & rows["fold"].eq("oos")
                & rows["actual_rv_mean"].notna()
            ].copy()
            if rows.empty:
                continue
            wide_y = rows.pivot(index="origin_time", columns="model", values="actual_rv_mean")
            wide_h = rows.pivot(index="origin_time", columns="model", values="rv_forecast")
            wide_floor = rows.pivot(index="origin_time", columns="model", values="train_forecast_floor")
            common = wide_y.notna().all(axis=1) & wide_h.notna().all(axis=1)
            actual = wide_y.loc[common].iloc[:, 0]
            forecasts = wide_h.loc[common]
            floors = wide_floor.loc[common]
            if forecasts.empty:
                continue
            benchmark_name = "random_walk" if "random_walk" in forecasts else None
            symbol_result = {"horizon_days": int(horizon), "n_common_origins": int(len(actual)), "models": {}, "dm_vs_random_walk": {}}
            loss_table = pd.DataFrame(index=actual.index)
            for model in forecasts.columns:
                predicted = forecasts[model].to_numpy(dtype=float)
                actual_values = actual.to_numpy(dtype=float)
                floor_values = floors[model].to_numpy(dtype=float)
                benchmark = forecasts[benchmark_name].to_numpy(dtype=float) if benchmark_name else None
                stats = volatility_metrics(
                    actual_values,
                    predicted,
                    benchmark=benchmark,
                    train_floor=floor_values,
                    hac_lags=int(research["evaluation"]["hac_lags"][int(horizon)]),
                )
                loss_table[model] = qlike_loss(actual_values, predicted, floor_values)
                symbol_result["models"][model] = stats
                metrics_rows.append({"symbol": symbol, "horizon_days": int(horizon), "model": model, **stats})
                if benchmark_name and model != benchmark_name:
                    result = dm_test(
                        loss_table[model],
                        qlike_loss(actual_values, benchmark, floors[benchmark_name].to_numpy(dtype=float)),
                        horizon=int(horizon),
                        hac_lags=int(research["evaluation"]["hac_lags"][int(horizon)]),
                        hln=True,
                    )
                    symbol_result["dm_vs_random_walk"][model] = result
                    dm_rows.append({
                        "symbol": symbol,
                        "horizon_days": int(horizon),
                        "model": model,
                        "benchmark": benchmark_name,
                        **result,
                    })
            if not args.skip_mcs and loss_table.shape[1] >= 2:
                mcs_config = research["evaluation"]["mcs"]
                try:
                    mcs = model_confidence_set(
                        loss_table,
                        alpha=float(mcs_config["alpha"]),
                        bootstrap=int(mcs_config["bootstrap"]),
                        block=int(mcs_config["block"][int(horizon)]),
                        seed=int(research["seed"]),
                    )
                    mcs_rows.extend({
                        "symbol": symbol,
                        "horizon_days": int(horizon),
                        **row,
                    } for row in mcs.to_dict("records"))
                    symbol_result["mcs"] = mcs.to_dict("records")
                except Exception as exc:
                    symbol_result["mcs_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"WARNING: MCS failed for {symbol} h={horizon}: {type(exc).__name__}: {exc}")
            report["symbols"][symbol][str(horizon)] = symbol_result

    tables = ROOT / "results" / "tables"
    metrics_dir = ROOT / "results" / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(tables / "rv_oos_metrics.csv", index=False)
    dm_frame = pd.DataFrame(dm_rows)
    if not dm_frame.empty:
        pvalues = pd.to_numeric(dm_frame["pvalue"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pvalues)
        adjusted = np.full(len(dm_frame), np.nan)
        indices = np.flatnonzero(valid)
        order = np.argsort(pvalues[valid])
        ordered = pvalues[valid][order]
        corrected = np.minimum(np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered), 1.0)
        adjusted[indices[order]] = corrected
        dm_frame["pvalue_holm_global"] = adjusted
        dm_frame["holm_reject_5pct"] = adjusted < 0.05
        dm_frame["multiplicity_family"] = "all DM/HLN model comparisons in this report"
    dm_frame.to_csv(tables / "dm_hln_tests.csv", index=False)
    if mcs_rows:
        pd.DataFrame(mcs_rows).to_csv(tables / "mcs_membership.csv", index=False)
    destination = metrics_dir / "rv_model_comparison.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote model metrics, DM/HLN and MCS outputs under {tables}")


if __name__ == "__main__":
    from _legacy_ohlcv import refuse_legacy_ohlcv
    refuse_legacy_ohlcv("make_report_tables.py")
