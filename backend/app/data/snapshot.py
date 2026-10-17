"""Freeze an immutable, as-of copy of validated candles for research runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import timezone
from pathlib import Path

import pandas as pd

from app.core.config import REPOSITORY_ROOT, load_settings
from app.core.research_config import load_research_config
from app.core.versioning import sha256_bytes, sha256_file


def _timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize(timezone.utc)
    return result.tz_convert("UTC")


def _gap_report(frame: pd.DataFrame, interval: str) -> tuple[int, list[dict]]:
    delta = pd.Timedelta(interval)
    times = pd.to_datetime(frame["open_time"], utc=True).sort_values().reset_index(drop=True)
    differences = times.diff()
    positions = (differences > delta).to_numpy().nonzero()[0]
    edges = []
    missing_total = 0
    for position in positions:
        absent = int(differences.iloc[position] / delta) - 1
        missing_total += absent
        edges.append({
            "after": times.iloc[position - 1].isoformat(),
            "before": times.iloc[position].isoformat(),
            "missing_intervals": absent,
        })
    return missing_total, edges


def freeze_snapshot(as_of: str | pd.Timestamp | None = None, force: bool = False) -> Path:
    """Copy only candles with close_time <= as_of and write hashes to a manifest."""
    settings = load_settings()
    research = load_research_config()
    cutoff = _timestamp(as_of or research["as_of_utc"])
    label = cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshots_root = settings.data_dir / "snapshots"
    destination = snapshots_root / label
    if destination.exists() and not force:
        if (destination / "manifest.json").exists():
            return destination
        raise FileExistsError(f"Snapshot exists without a manifest: {destination}")

    snapshots_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=snapshots_root))
    manifest: dict = {
        "as_of_utc": cutoff.isoformat(),
        "interval": settings.interval,
        "symbols": {},
        "config": {},
    }
    try:
        for symbol in research["symbols"]:
            source = settings.data_dir / "processed" / f"symbol={symbol}" / f"interval={settings.interval}" / "klines.parquet"
            if not source.exists():
                raise FileNotFoundError(f"Missing processed candles: {source}")
            frame = pd.read_parquet(source)
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
            frame = frame.loc[frame["close_time"] <= cutoff].sort_values("open_time").reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"No candles at or before {cutoff.isoformat()} for {symbol}")
            relative = Path(f"symbol={symbol}") / "klines.parquet"
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False)
            missing, edges = _gap_report(frame, settings.interval)
            manifest["symbols"][symbol] = {
                "file": relative.as_posix(),
                "sha256": sha256_file(output),
                "rows": int(len(frame)),
                "first_open_time_utc": frame["open_time"].min().isoformat(),
                "last_open_time_utc": frame["open_time"].max().isoformat(),
                "last_close_time_utc": frame["close_time"].max().isoformat(),
                "missing_intervals": missing,
                "gap_edges": edges,
            }
        for config_name in ("research.yaml", "search_space.yaml"):
            config_path = REPOSITORY_ROOT / "config" / config_name
            manifest["config"][config_name] = sha256_file(config_path)
        version_material = "|".join(
            manifest["symbols"][symbol]["sha256"] for symbol in sorted(manifest["symbols"])
        )
        version_material += "|" + "|".join(
            f"{name}:{manifest['config'][name]}" for name in sorted(manifest["config"])
        )
        manifest["data_version"] = sha256_bytes(version_material.encode("utf-8"))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if destination.exists():
            if force:
                shutil.rmtree(destination)
            else:
                raise FileExistsError(f"Snapshot already exists: {destination}")
        os.replace(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_snapshot(as_of: str | pd.Timestamp | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load a frozen snapshot and verify each Parquet file against its manifest."""
    settings = load_settings()
    research = load_research_config()
    cutoff = _timestamp(as_of or research["as_of_utc"])
    path = settings.data_dir / "snapshots" / cutoff.strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for symbol, metadata in manifest["symbols"].items():
        source = path / metadata["file"]
        if sha256_file(source) != metadata["sha256"]:
            raise ValueError(f"Snapshot SHA-256 mismatch for {symbol}: {source}")
        frames[symbol] = pd.read_parquet(source)
    return frames, manifest
