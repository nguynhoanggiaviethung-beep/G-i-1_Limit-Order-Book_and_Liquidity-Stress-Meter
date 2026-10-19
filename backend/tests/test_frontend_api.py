import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from app.api.routes import forecasts, research_data
from app.main import app, frontend_fallback, home
from app.services import forecast_service


def test_frontend_endpoints_are_in_openapi():
    paths = app.openapi()["paths"]
    assert "/api/v1/meta" in paths
    assert "/api/v1/results/tables/{table_id}" in paths
    assert "/api/v1/results/figures/{figure_id}" in paths
    assert "get" in paths["/api/v1/forecasts/history"]


def test_meta_returns_snapshot_provenance_and_models(tmp_path, monkeypatch):
    root = tmp_path
    label = "2026-10-01T00-00-00Z"
    snapshot = root / "data" / "snapshots" / label
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps({"data_version": "abc123"}), encoding="utf-8"
    )
    config = root / "config"
    config.mkdir()
    (config / "research.yaml").write_text("as_of_utc: 2026-10-01", encoding="utf-8")
    table_dir = root / "results" / "tables"
    table_dir.mkdir(parents=True)
    (table_dir / "rv_oos_metrics.csv").write_text("model\nhar\nxgboost\n", encoding="utf-8")
    monkeypatch.setattr(research_data, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        research_data,
        "load_research_config",
        lambda: {"as_of_utc": "2026-10-01T00:00:00Z", "horizons_days": [1, 7, 30]},
    )
    monkeypatch.setattr(
        research_data,
        "load_settings",
        lambda: SimpleNamespace(
            data_dir=root / "data", symbols=("BTCUSDT", "ETHUSDT"), confidence_levels=(0.95, 0.975, 0.99)
        ),
    )

    result = research_data.get_meta()

    assert result["data_version"] == "abc123"
    assert result["models"] == ["har", "xgboost"]
    assert result["mode"] == "research"
    assert len(result["config_hash"]) == 64
    assert research_data.get_meta(mode="production")["data_version"] is None


def test_result_table_and_figure_are_allowlisted(tmp_path, monkeypatch):
    tables = tmp_path / "results" / "tables"
    figures = tmp_path / "results" / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    table_path = tables / "T3_rv_oos_metrics.csv"
    figure_path = figures / "F4_dm_holm_heatmap.svg"
    table_path.write_text("model,qlike\nhar,0.2\n", encoding="utf-8")
    figure_path.write_text("<svg></svg>", encoding="utf-8")
    monkeypatch.setattr(research_data, "REPOSITORY_ROOT", tmp_path)

    assert Path(research_data.get_result_table("T3").path) == table_path
    assert Path(research_data.get_result_figure("F4").path) == figure_path
    with pytest.raises(HTTPException) as error:
        research_data.get_result_table("../secrets")
    assert error.value.status_code == 404


def test_history_returns_oos_points_and_matching_saved_risk(tmp_path, monkeypatch):
    root = tmp_path
    forecast_dir = root / "results" / "forecasts"
    forecast_dir.mkdir(parents=True)
    forecasts_frame = pd.DataFrame(
        [{
            "symbol": "BTCUSDT",
            "model": "har",
            "horizon_days": 1,
            "status": "ok",
            "fold": "oos",
            "origin_time": pd.Timestamp("2026-09-30T00:00:00Z"),
            "target_end": pd.Timestamp("2026-10-01T00:00:00Z"),
            "actual_rv_mean": 1.25,
            "rv_forecast": 1.5,
            "sigma_h_approx": 1.2,
            "y_cum_return": -0.8,
            "data_version": "data-v",
            "config_hash": "config-v",
        }]
    )
    forecasts_frame.to_parquet(forecast_dir / "btcusdt_h1_walkforward.parquet", index=False)
    risk_frame = pd.DataFrame(
        [{
            "symbol": "BTCUSDT",
            "model": "har",
            "horizon_days": 1,
            "origin_time": pd.Timestamp("2026-09-30T00:00:00Z"),
            "method": "filtered_historical_simulation",
            "confidence": 0.95,
            "evaluation_scope": "full_oos",
            "actual_loss_percent": 0.8,
            "var_loss_percent": 1.7,
            "expected_shortfall_loss_percent": 2.3,
            "status": "ok",
        }]
    )
    risk_frame.to_parquet(forecast_dir / "risk_forecasts.parquet", index=False)
    monkeypatch.setattr(forecast_service, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        forecast_service,
        "load_settings",
        lambda: SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT")),
    )

    result = forecast_service.get_forecast_history("BTCUSDT", "har", 1)

    assert result["points"][0]["actual_rv_mean"] == 1.25
    assert result["points"][0]["var_loss_percent"] == 1.7
    assert result["points"][0]["expected_shortfall_loss_percent"] == 2.3


def test_fastapi_serves_built_spa_and_deep_links(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.REPOSITORY_ROOT", tmp_path)
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    index = dist / "index.html"
    index.write_text("<main>frontend</main>", encoding="utf-8")
    (dist / "assets").mkdir()
    asset = dist / "assets" / "app.js"
    asset.write_text("console.log('app')", encoding="utf-8")

    assert Path(home().path) == index
    assert Path(frontend_fallback("forecast").path) == index
    assert Path(frontend_fallback("assets/app.js").path) == asset

