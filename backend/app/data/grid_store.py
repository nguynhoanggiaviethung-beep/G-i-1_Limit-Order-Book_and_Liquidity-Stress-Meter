"""Read normalized one-second limit-order-book replay grids from Parquet."""

from __future__ import annotations

import os
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from app.core.config import REPOSITORY_ROOT, load_settings


DEFAULT_GRID_ROOT = load_settings().data_dir / "grid"


def _utc_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


class GridStore:
    """Small read-only catalog for partitioned ``book_1s.parquet`` files."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("LOB_GRID_ROOT")
        self.root = Path(configured).expanduser() if configured else DEFAULT_GRID_ROOT
        if not self.root.is_absolute():
            self.root = REPOSITORY_ROOT / self.root
        self._frame_cache: OrderedDict[Path, tuple[int, int, pd.DataFrame]] = OrderedDict()

    def _read_partition(self, path: Path) -> pd.DataFrame:
        stat = path.stat()
        cached = self._frame_cache.get(path)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            self._frame_cache.move_to_end(path)
            return cached[2].copy()
        frame = pd.read_parquet(path)
        self._frame_cache[path] = (stat.st_mtime_ns, stat.st_size, frame)
        self._frame_cache.move_to_end(path)
        while len(self._frame_cache) > 3:
            self._frame_cache.popitem(last=False)
        return frame.copy()

    def _files(self, symbol: str) -> list[Path]:
        symbol = symbol.upper()
        if not self.root.exists():
            return []
        return sorted(
            path for path in self.root.rglob("*.parquet")
            if f"symbol={symbol}" in path.parts
            and (path.name in {"book_1s.parquet", "features_1s.parquet"} or "book_1s" in path.stem or "features_1s" in path.stem)
        )

    @staticmethod
    def _date_from_path(path: Path) -> str | None:
        for part in path.parts:
            if part.startswith("date="):
                return part.split("=", 1)[1]
        return None

    def catalog(self, symbols: tuple[str, ...] | list[str]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for symbol in symbols:
            partitions = []
            for path in self._files(symbol):
                try:
                    rows = int(pq.ParquetFile(path).metadata.num_rows)
                except Exception:
                    rows = None
                date = self._date_from_path(path)
                qa = self._qa_for_partition(symbol, date, path)
                partitions.append({"date_utc": date, "rows": rows, **qa})
            dates = [item["date_utc"] for item in partitions if item["date_utc"]]
            entries[symbol] = {
                "status": "ready" if partitions else "unavailable",
                "partition_count": len(partitions),
                "row_count": sum(item["rows"] or 0 for item in partitions) if partitions else 0,
                "valid_seconds": self._sum_optional(partitions, "valid_seconds"),
                "expected_seconds": self._sum_optional(partitions, "expected_seconds"),
                "dropped_seconds": self._sum_optional(partitions, "dropped_seconds"),
                "sequence_gap_count": self._sum_optional(partitions, "sequence_gap_count"),
                "snapshot_bootstraps": self._sum_optional(partitions, "snapshot_bootstraps"),
                "snapshot_recoveries": self._sum_optional(partitions, "snapshot_recoveries"),
                "qa_available": bool(partitions) and all(item.get("qa_available", False) for item in partitions),
                "coverage": self._coverage(partitions),
                "first_date_utc": min(dates) if dates else None,
                "last_date_utc": max(dates) if dates else None,
                "partitions": partitions,
                "reason": None if partitions else f"No replay grid found under {self.root}",
            }
        return {
            "schema": "data_status.v1",
            "status": "ready" if any(item["status"] == "ready" for item in entries.values()) else "unavailable",
            "source": "Configured historical or captured L2 order book and trades, normalized to a one-second grid",
            "grid_root": str(self.root),
            "symbols": entries,
        }

    @staticmethod
    def _sum_optional(partitions: list[dict[str, Any]], key: str) -> int | None:
        values = [item.get(key) for item in partitions]
        if not values or any(value is None for value in values):
            return None
        return int(sum(values))

    @classmethod
    def _coverage(cls, partitions: list[dict[str, Any]]) -> float | None:
        valid = cls._sum_optional(partitions, "valid_seconds")
        expected = cls._sum_optional(partitions, "expected_seconds")
        return float(valid / expected) if valid is not None and expected else None

    @staticmethod
    def _qa_for_partition(symbol: str, date: str | None, data_path: Path) -> dict[str, Any]:
        if not date:
            return {"qa_available": False}
        qa_root = REPOSITORY_ROOT / "outputs" / "manifests"
        candidates = (
            data_path.with_suffix(".qa.json"),
            data_path.parent / "qa.json",
            qa_root / f"symbol={symbol}" / f"date={date}" / "qa.json",
            qa_root / symbol / f"{date}.json",
        )
        for path in candidates:
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    return {
                        "qa_available": True,
                        "valid_seconds": payload.get("valid_seconds"),
                        "expected_seconds": payload.get("expected_seconds"),
                        "dropped_seconds": payload.get("dropped_seconds"),
                        "sequence_gap_count": payload.get("sequence_gap_count"),
                        "snapshot_bootstraps": payload.get("snapshot_bootstraps"),
                        "snapshot_recoveries": payload.get("snapshot_recoveries"),
                        "coverage": payload.get("coverage"),
                    }
                except (OSError, json.JSONDecodeError, TypeError):
                    break
        return {"qa_available": False, "valid_seconds": None, "expected_seconds": None, "dropped_seconds": None, "sequence_gap_count": None, "snapshot_bootstraps": None, "snapshot_recoveries": None, "coverage": None}

    def read(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        start_ts, end_ts = _utc_timestamp(start), _utc_timestamp(end)
        files = self._files(symbol)
        if start_ts is not None:
            files = [path for path in files if (self._date_from_path(path) or "") >= start_ts.date().isoformat()]
        if end_ts is not None:
            files = [path for path in files if (self._date_from_path(path) or "") <= end_ts.date().isoformat()]
        if not files:
            return pd.DataFrame()
        if start_ts is None and end_ts is None:
            files = files[-1:]
        frames = [self._read_partition(path) for path in files]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return frame
        timestamp_column = next(
            (name for name in ("timestamp_utc", "sample_time_utc", "timestamp", "time") if name in frame.columns),
            None,
        )
        if timestamp_column is None:
            return pd.DataFrame()
        values = frame[timestamp_column]
        if pd.api.types.is_numeric_dtype(values):
            magnitude = pd.to_numeric(values, errors="coerce").dropna().abs().median()
            unit = "ms" if magnitude and magnitude > 10**11 else "s"
            parsed = pd.to_datetime(values, unit=unit, utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
        frame = frame.assign(timestamp_utc=parsed).dropna(subset=["timestamp_utc"])
        if "symbol" not in frame:
            frame["symbol"] = symbol
        if "book_valid" in frame:
            validity = frame["book_valid"]
            if validity.dtype == object:
                validity = validity.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            else:
                validity = validity.fillna(False).astype(bool)
            frame["book_valid"] = validity
            frame = frame[validity]
        if start_ts is not None:
            frame = frame[frame["timestamp_utc"] >= start_ts]
        if end_ts is not None:
            frame = frame[frame["timestamp_utc"] <= end_ts]
        frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
        if limit is not None:
            frame = frame.tail(max(0, int(limit)))
        return frame.reset_index(drop=True)

    @staticmethod
    def json_row(row: pd.Series) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, pd.Timestamp):
                return value.isoformat().replace("+00:00", "Z")
            if hasattr(value, "tolist"):
                return clean(value.tolist())
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            if hasattr(value, "item"):
                return clean(value.item())
            return value

        result: dict[str, Any] = {}
        for key, value in row.items():
            result[key] = clean(value)
        result["timestamp_utc"] = pd.Timestamp(row["timestamp_utc"]).isoformat().replace("+00:00", "Z")
        return result

