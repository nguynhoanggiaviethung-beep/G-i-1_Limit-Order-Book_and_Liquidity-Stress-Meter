"""Build the named research tables from pipeline metrics and risk outputs."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write(frame: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / name, index=False)


def main() -> None:
    research = load_research_config()
    settings = load_settings()
    label = pd.Timestamp(research["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = settings.data_dir / "snapshots" / label
    feature_dir = settings.data_dir / "features" / "snapshots" / label
    tables = ROOT / "results" / "tables"
    metrics = ROOT / "results" / "metrics"
    forecasts = ROOT / "results" / "forecasts"
    tables.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(snapshot_dir / "manifest.json")
    quality = _read_json(metrics / "snapshot_quality.json")
    description_rows = []
    summary_rows = []
    for symbol in research["symbols"]:
        source = manifest.get("symbols", {}).get(symbol, {})
        path = feature_dir / f"symbol={symbol}" / "daily_rv.parquet"
        if not path.exists():
            continue
        daily = pd.read_parquet(path)
        complete = daily.loc[daily["is_complete_day"].fillna(False)].copy()
        returns = pd.to_numeric(complete.get("close_ret"), errors="coerce").dropna()
        rv = pd.to_numeric(complete.get("rv"), errors="coerce").dropna()
        description_rows.append({
            "symbol": symbol,
            "interval": manifest.get("interval"),
            "candle_rows": source.get("rows"),
            "first_candle_utc": source.get("first_open_time_utc"),
            "last_candle_close_utc": source.get("last_close_time_utc"),
            "missing_5m_intervals": source.get("missing_intervals"),
            "complete_utc_days": int(len(complete)),
            "snapshot_data_version": manifest.get("data_version"),
            "quality_status": quality.get("symbols", {}).get(symbol, {}).get("status"),
        })
        summary_rows.append({
            "symbol": symbol,
            "n_complete_days": int(len(complete)),
            "return_mean_percent": float(returns.mean()) if len(returns) else np.nan,
            "return_std_percent": float(returns.std(ddof=1)) if len(returns) > 1 else np.nan,
            "return_skewness": float(returns.skew()) if len(returns) > 2 else np.nan,
            "return_excess_kurtosis": float(returns.kurt()) if len(returns) > 3 else np.nan,
            "rv_mean_percent_squared": float(rv.mean()) if len(rv) else np.nan,
            "rv_median_percent_squared": float(rv.median()) if len(rv) else np.nan,
            "rv_95pct_percent_squared": float(rv.quantile(0.95)) if len(rv) else np.nan,
            "rv_max_percent_squared": float(rv.max()) if len(rv) else np.nan,
        })
    _write(pd.DataFrame(description_rows), tables, "T1_data_description.csv")
    _write(pd.DataFrame(summary_rows), tables, "T2_return_rv_statistics.csv")

    rv_metrics = _read_csv(tables / "rv_oos_metrics.csv")
    _write(rv_metrics, tables, "T3_rv_oos_metrics.csv")

    dm = _read_csv(tables / "dm_hln_tests.csv")
    if not dm.empty and "pvalue" in dm:
        pvalues = pd.to_numeric(dm["pvalue"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pvalues)
        adjusted = np.full(len(dm), np.nan)
        ordered = np.argsort(pvalues[valid])
        finite = pvalues[valid][ordered]
        m = len(finite)
        corrected = np.maximum.accumulate((m - np.arange(m)) * finite)
        corrected = np.minimum(corrected, 1.0)
        valid_indices = np.flatnonzero(valid)
        adjusted[valid_indices[ordered]] = corrected
        dm["pvalue_holm_global"] = adjusted
        dm["holm_reject_5pct"] = adjusted < 0.05
    _write(dm, tables, "T4_dm_hln_holm.csv")
    _write(_read_csv(tables / "mcs_membership.csv"), tables, "T5_mcs_membership.csv")

    calibration_rows = []
    if not rv_metrics.empty and "mincer_zarnowitz" in rv_metrics:
        for row in rv_metrics.to_dict("records"):
            try:
                values = ast.literal_eval(row.get("mincer_zarnowitz", "{}"))
            except (ValueError, SyntaxError):
                values = {}
            calibration_rows.append({
                "symbol": row.get("symbol"),
                "horizon_days": row.get("horizon_days"),
                "model": row.get("model"),
                **(values if isinstance(values, dict) else {}),
            })
    _write(pd.DataFrame(calibration_rows), tables, "T6_mincer_zarnowitz.csv")

    risk_report = _read_json(metrics / "var_es_backtests.json")
    var_rows = []
    es_rows = []
    for result in risk_report.get("backtests", []):
        base = {
            key: result.get(key)
            for key in ("symbol", "model", "horizon_days", "method", "confidence", "n_obs", "n_independent_approx", "expected_exceedances", "reliability", "warning")
        }
        offsets = result.get("offset_backtests") or [result]
        for offset_result in offsets:
            kupiec = offset_result.get("kupiec", {})
            conditional = offset_result.get("conditional_coverage", {})
            dq = offset_result.get("dq", {})
            var_rows.append({
                **base,
                "offset": offset_result.get("offset"),
                "n_obs_offset": offset_result.get("n_obs"),
                "exceptions": offset_result.get("exceptions", result.get("exceptions")),
                "exception_rate": offset_result.get("exception_rate", result.get("exception_rate")),
                "quantile_loss": offset_result.get("quantile_loss"),
                "kupiec_pvalue": kupiec.get("pvalue"),
                "christoffersen_cc_pvalue": conditional.get("pvalue_cc"),
                "dq_pvalue": dq.get("pvalue"),
                "status": offset_result.get("status", "ok"),
                "error": offset_result.get("error"),
            })
        es = result.get("es_evaluation", {})
        z2 = es.get("acerbi_szekely", {})
        e_test = es.get("e_backtest", {})
        es_rows.append({
            **base,
            "fz0_mean": es.get("fz0_mean"),
            "mean_es": es.get("mean_es"),
            "acerbi_szekely_z2": z2.get("z2"),
            "acerbi_szekely_pvalue": z2.get("pvalue"),
            "acerbi_szekely_interpretation": z2.get("interpretation"),
            "max_e_value": e_test.get("max_e_value"),
            "e_backtest_reject_5pct": e_test.get("reject_at_5pct"),
            "es_evaluation_error": result.get("es_evaluation_error"),
        })
    _write(pd.DataFrame(var_rows), tables, "T7_var_coverage.csv")
    _write(pd.DataFrame(es_rows), tables, "T8_es_backtests.csv")
    _write(_read_csv(tables / "regime_rv_metrics.csv"), tables, "T9_regime_metrics.csv")

    economic = _read_csv(tables / "economic_strategy.csv")
    budget = _read_csv(tables / "risk_budget_backtest.csv")
    economic_parts = []
    if not economic.empty:
        economic = economic.copy()
        economic["table_source"] = "economic_strategy"
        economic_parts.append(economic)
    if not budget.empty:
        budget = budget.copy()
        budget["table_source"] = "risk_budget"
        economic_parts.append(budget)
    _write(pd.concat(economic_parts, ignore_index=True, sort=False) if economic_parts else pd.DataFrame(), tables, "T10_economic_risk_strategies.csv")

    robustness = _read_csv(tables / "robustness_summary.csv")
    _write(robustness, tables, "T11_robustness.csv")
    print(f"Wrote T1-T11 paper tables under {tables}")


if __name__ == "__main__":
    main()
