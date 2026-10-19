"""Read-only metadata and downloadable artifacts for the frontend."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import REPOSITORY_ROOT, load_settings
from app.core.research_config import load_research_config
from app.core.versioning import sha256_file

router = APIRouter(prefix="/api/v1", tags=["research artifacts"])


def _snapshot_manifest() -> tuple[dict, Path]:
    research = load_research_config()
    settings = load_settings()
    label = pd.Timestamp(research["as_of_utc"]).strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest_path = settings.data_dir / "snapshots" / label / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=503, detail="Frozen snapshot manifest is unavailable")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Frozen snapshot manifest is unreadable") from exc


@router.get("/meta")
def get_meta(mode: Literal["research", "production"] = "research") -> dict:
    """Expose provenance and the filters supported by the current workspace."""
    research = load_research_config()
    settings = load_settings()
    manifest, _ = _snapshot_manifest()
    models_path = REPOSITORY_ROOT / "results" / "tables" / "rv_oos_metrics.csv"
    models: list[str] = []
    if models_path.exists():
        try:
            model_frame = pd.read_csv(models_path, usecols=["model"])
            models = sorted(model_frame["model"].dropna().astype(str).unique().tolist())
        except (OSError, ValueError, pd.errors.ParserError):
            models = []
    if not models:
        model_config = research.get("models", {})
        models = sorted({str(model) for group in model_config.values() for model in group})
    config_path = REPOSITORY_ROOT / "config" / "research.yaml"
    return {
        "mode": mode,
        "as_of_utc": research["as_of_utc"],
        # Production forecasts return their own rolling candle fingerprint. Do
        # not label the frozen research version as the current production feed.
        "data_version": manifest.get("data_version") if mode == "research" else None,
        "config_hash": sha256_file(config_path) if config_path.exists() else None,
        "symbols": list(settings.symbols),
        "horizons_days": [int(value) for value in research["horizons_days"]],
        "models": models,
        "confidence_levels": list(settings.confidence_levels),
        "source": "Binance Spot 5m; frozen snapshot" if mode == "research" else "Rolling Binance Spot 5m data",
    }


@router.get("/results/tables/{table_id}")
def get_result_table(table_id: str) -> FileResponse:
    """Serve an allowlisted T1-T11 CSV artifact without evaluating it in-browser."""
    if not re.fullmatch(r"T(?:[1-9]|1[01])", table_id):
        raise HTTPException(status_code=404, detail="Unknown research table")
    tables_dir = REPOSITORY_ROOT / "results" / "tables"
    candidates = sorted(tables_dir.glob(f"{table_id}_*.csv"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Research table {table_id} is not available")
    return FileResponse(
        candidates[0],
        media_type="text/csv",
        filename=candidates[0].name,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/results/figures/{figure_id}")
def get_result_figure(figure_id: str) -> FileResponse:
    """Serve an allowlisted F1-F9 SVG artifact."""
    if not re.fullmatch(r"F[1-9]", figure_id):
        raise HTTPException(status_code=404, detail="Unknown research figure")
    path = REPOSITORY_ROOT / "results" / "figures" / f"{figure_id}_*.svg"
    candidates = sorted(path.parent.glob(path.name))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Research figure {figure_id} is not available")
    return FileResponse(
        candidates[0],
        media_type="image/svg+xml",
        filename=candidates[0].name,
        headers={"X-Content-Type-Options": "nosniff"},
    )
