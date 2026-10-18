"""Create ex-ante regime-conditioned metrics and a cost-aware OOS strategy table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.core.research_config import load_research_config  # noqa: E402
from app.evaluation.economic import dynamic_margin_backtest, risk_budget_backtest, volatility_targeting  # noqa: E402
from app.evaluation.regimes import label_regimes  # noqa: E402
from app.evaluation.volatility_metrics import volatility_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--horizon", action="append", type=int)
    args = parser.parse_args()

    settings = load_settings()
    research = load_research_config()
    symbols = [value.upper() for value in (args.symbol or research["symbols"])]
    horizons = args.horizon or research["horizons_days"]
    cutoff = pd.Timestamp(research["as_of_utc"])
    label = cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    feature_root = settings.data_dir / "features" / "snapshots" / label
    forecast_root = ROOT / "results" / "forecasts"
    tables = ROOT / "results" / "tables"
    metrics = ROOT / "results" / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    regime_rows: list[dict] = []
    economic_rows: list[dict] = []
    risk_budget_rows: list[dict] = []
    dynamic_margin_rows: list[dict] = []
    report: dict = {"as_of_utc": research["as_of_utc"], "symbols": {}}

    for symbol in symbols:
        daily_path = feature_root / f"symbol={symbol}" / "daily_rv.parquet"
        forecast_base = forecast_root / f"{symbol.lower()}_h"
        if not daily_path.exists():
            print(f"Skipping {symbol}: frozen features are missing at {daily_path}")
            continue
        daily = pd.read_parquet(daily_path)
        daily["utc_date"] = pd.to_datetime(daily["utc_date"], utc=True)
        labels = label_regimes(daily, research.get("regimes", {}))
        regime_frame = pd.DataFrame({
            "origin_time": daily["utc_date"] + pd.Timedelta(days=1),
            "regime": labels.to_numpy(),
        })
        report["symbols"][symbol] = {"regime_counts": labels.value_counts().to_dict(), "horizons": {}}

        for horizon in horizons:
            path = Path(f"{forecast_base}{horizon}_walkforward.parquet")
            if not path.exists():
                continue
            forecasts = pd.read_parquet(path)
            forecasts = forecasts.loc[
                forecasts["status"].eq("ok")
                & forecasts["fold"].eq("oos")
                & forecasts["actual_rv_mean"].notna()
            ].copy()
            if forecasts.empty:
                continue
            forecasts["origin_time"] = pd.to_datetime(forecasts["origin_time"], utc=True)
            enriched = forecasts.merge(regime_frame, on="origin_time", how="left", validate="many_to_one")
            benchmark = enriched.loc[enriched["model"].eq("random_walk")].drop_duplicates("origin_time")
            benchmark_by_origin = benchmark.set_index("origin_time")["rv_forecast"] if not benchmark.empty else None
            horizon_report = {"n_oos_rows": int(len(enriched)), "regimes": {}}

            for model, model_rows in enriched.groupby("model", sort=True):
                for regime, group in model_rows.groupby("regime", dropna=False, sort=True):
                    comparison = None
                    if benchmark_by_origin is not None:
                        comparison = group["origin_time"].map(benchmark_by_origin).to_numpy(dtype=float)
                    try:
                        result = volatility_metrics(
                            group["actual_rv_mean"].to_numpy(dtype=float),
                            group["rv_forecast"].to_numpy(dtype=float),
                            benchmark=comparison,
                            train_floor=group["train_forecast_floor"].to_numpy(dtype=float),
                            hac_lags=int(research["evaluation"]["hac_lags"][int(horizon)]),
                        )
                    except ValueError:
                        continue
                    regime_name = str(regime) if pd.notna(regime) else "unknown"
                    regime_rows.append({
                        "symbol": symbol,
                        "horizon_days": int(horizon),
                        "model": model,
                        "regime": regime_name,
                        **result,
                    })
                    horizon_report["regimes"].setdefault(regime_name, {})[model] = result

            if int(horizon) == 1:
                for model, group in enriched.groupby("model", sort=True):
                    valid = group.dropna(subset=["y_cum_return", "sigma_h_approx"])
                    if valid.empty:
                        continue
                    result = volatility_targeting(
                        valid["y_cum_return"].to_numpy(dtype=float),
                        valid["sigma_h_approx"].to_numpy(dtype=float),
                    )
                    economic_rows.append({
                        "symbol": symbol,
                        "horizon_days": 1,
                        "model": model,
                        "strategy": "volatility_targeting",
                        **result,
                    })
                risk_path = forecast_root / "risk_forecasts.parquet"
                if risk_path.exists():
                    risks = pd.read_parquet(risk_path)
                    risks = risks.loc[
                        risks["symbol"].eq(symbol)
                        & risks["horizon_days"].eq(1)
                        & risks["confidence"].eq(0.95)
                    ].copy()
                    risks["origin_time"] = pd.to_datetime(risks["origin_time"], utc=True)
                    forecast_versions = forecasts.copy()
                    forecast_versions["origin_time"] = pd.to_datetime(
                        forecast_versions["origin_time"], utc=True
                    )
                    # All models at one origin use the same frozen candle snapshot;
                    # use that to validate quantile forecasts too, even though they
                    # have a separate quantile-config hash and no RV model row.
                    if "data_version" in risks and "data_version" in forecast_versions:
                        origin_versions = (
                            forecast_versions.groupby("origin_time", as_index=False)["data_version"]
                            .first()
                        )
                        risks = risks.merge(
                            origin_versions.rename(columns={"data_version": "_expected_data_version"}),
                            on="origin_time",
                            how="left",
                            validate="many_to_one",
                        )
                        version_matches = (
                            risks["_expected_data_version"].isna()
                            | risks["data_version"].eq(risks["_expected_data_version"])
                        )
                        risks = risks.loc[version_matches].drop(columns="_expected_data_version")
                    # A config hash belongs to a model's own forecast run. Do not
                    # compare a quantile model's hash with an unrelated HAR/RV hash.
                    if "config_hash" in risks and "config_hash" in forecast_versions:
                        expected_configs = forecast_versions[
                            ["model", "origin_time", "config_hash"]
                        ].drop_duplicates(["model", "origin_time"])
                        risks = risks.merge(
                            expected_configs.rename(columns={"config_hash": "_expected_config_hash"}),
                            on=["model", "origin_time"],
                            how="left",
                            validate="many_to_one",
                        )
                        config_matches = (
                            risks["_expected_config_hash"].isna()
                            | risks["config_hash"].eq(risks["_expected_config_hash"])
                        )
                        risks = risks.loc[config_matches].drop(columns="_expected_config_hash")
                    for (model, method), group in risks.groupby(["model", "method"], sort=True):
                        score = risk_budget_backtest(
                            group["actual_loss_percent"],
                            group["var_loss_percent"],
                            budget_percent=1.0,
                        )
                        risk_budget_rows.append({
                            "symbol": symbol,
                            "horizon_days": 1,
                            "model": model,
                            "method": method,
                            "confidence": 0.95,
                            "evaluation_scope": (
                                str(group["evaluation_scope"].iloc[0])
                                if "evaluation_scope" in group else "unknown"
                            ),
                            "budget_percent": 1.0,
                            **score,
                        })
                        valid_margin = group.dropna(
                            subset=["actual_loss_percent", "expected_shortfall_loss_percent"]
                        ).sort_values("origin_time")
                        if not valid_margin.empty:
                            margin_result = dynamic_margin_backtest(
                                valid_margin["actual_loss_percent"],
                                valid_margin["expected_shortfall_loss_percent"],
                            )
                            summary = {
                                "symbol": symbol,
                                "horizon_days": 1,
                                "model": model,
                                "risk_method": method,
                                "confidence": 0.95,
                                "evaluation_scope": (
                                    str(valid_margin["evaluation_scope"].iloc[0])
                                    if "evaluation_scope" in valid_margin else "unknown"
                                ),
                                **margin_result["dynamic_es_margin"],
                            }
                            dynamic_margin_rows.append({**summary, "comparison": margin_result["fixed_margin"], "assumptions": margin_result["assumptions"]})
                            economic_rows.append({
                                **summary,
                                "strategy": "dynamic_margin",
                                **{f"fixed_{key}": value for key, value in margin_result["fixed_margin"].items() if key != "n_obs"},
                                **{f"assumption_{key}": value for key, value in margin_result["assumptions"].items()},
                            })

            report["symbols"][symbol]["horizons"][str(horizon)] = horizon_report

    pd.DataFrame(regime_rows).to_csv(tables / "regime_rv_metrics.csv", index=False)
    pd.DataFrame(economic_rows).to_csv(tables / "economic_strategy.csv", index=False)
    pd.DataFrame(risk_budget_rows).to_csv(tables / "risk_budget_backtest.csv", index=False)
    pd.DataFrame(dynamic_margin_rows).to_csv(tables / "dynamic_margin_backtest.csv", index=False)
    destination = metrics / "regime_economic_evaluation.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote regime/economic results under {tables}")


if __name__ == "__main__":
    main()
