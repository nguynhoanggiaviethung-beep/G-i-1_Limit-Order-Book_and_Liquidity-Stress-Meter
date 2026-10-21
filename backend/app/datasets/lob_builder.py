"""Build and persist forward-target datasets from captured LOB features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.datasets.leakage_checks import assert_no_target_leakage
from app.features.lob_targets import build_lob_targets
from app.quality.validation import validate_lob_samples
from app.storage.manifest import append_manifest, sha256_file


def load_lob_features(data_dir: Path, symbol: str) -> pd.DataFrame:
    data_root = Path(data_dir)
    manifest = data_root / "state" / "lob" / symbol.lower() / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"No LOB data manifest found for {symbol}: {manifest}")
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = (data_root / "features" / "lob" / f"symbol={symbol.upper()}").resolve()
    files = []
    for entry in entries:
        if entry.get("artifact_type") != "features" or entry.get("symbol") != symbol.upper():
            continue
        path = (data_root / entry["path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"Manifest feature path escapes the configured symbol directory: {path}")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"LOB feature checksum verification failed: {path}")
        files.append(path)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No checksummed LOB feature partitions found for {symbol}: {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def build_dataset(
    samples: pd.DataFrame,
    horizon_samples: int,
    sampling_interval_seconds: int = 1,
) -> pd.DataFrame:
    report = validate_lob_samples(samples, sampling_interval_seconds=sampling_interval_seconds)
    if report["errors"]:
        # Gap reports are informational; invalid schema/book errors are not.
        fatal = [error for error in report["errors"] if "sampling gap" not in error]
        if fatal:
            raise ValueError("Invalid LOB feature samples: " + "; ".join(fatal))
    dataset = build_lob_targets(samples, horizon_samples, sampling_interval_seconds)
    assert_no_target_leakage(dataset)
    return dataset


def save_dataset(dataset: pd.DataFrame, data_dir: Path, symbol: str, horizon_samples: int) -> Path:
    directory = Path(data_dir) / "features" / "lob_datasets" / f"symbol={symbol.upper()}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"horizon_samples={horizon_samples}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    dataset.to_parquet(temporary, index=False)
    temporary.replace(path)
    sample_times = pd.to_datetime(dataset["sample_time_utc"], utc=True)
    append_manifest(Path(data_dir), symbol.upper(), {
        "artifact_type": "dataset_targets",
        "path": str(path.relative_to(Path(data_dir))),
        "symbol": symbol.upper(),
        "horizon_samples": horizon_samples,
        "rows": int(len(dataset)),
        "valid_target_rows": int(dataset["target_valid"].fillna(False).sum()),
        "start_time_utc": sample_times.min().isoformat() if len(sample_times) else None,
        "end_time_utc": sample_times.max().isoformat() if len(sample_times) else None,
        "sha256": sha256_file(path),
        "collector_version": "lob-dataset-builder-1",
    })
    return path


def load_dataset(data_dir: Path, symbol: str, horizon_samples: int) -> pd.DataFrame:
    data_root = Path(data_dir)
    path = data_root / "features" / "lob_datasets" / f"symbol={symbol.upper()}" / f"horizon_samples={horizon_samples}.parquet"
    manifest = data_root / "state" / "lob" / symbol.lower() / "manifest.jsonl"
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"LOB dataset or manifest is missing for {symbol}, H={horizon_samples}")
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    relative = str(path.relative_to(data_root))
    matching = [entry for entry in entries if entry.get("artifact_type") == "dataset_targets" and entry.get("path") == relative]
    if not matching or sha256_file(path) != matching[-1].get("sha256"):
        raise ValueError(f"LOB dataset checksum verification failed: {path}")
    return pd.read_parquet(path)
