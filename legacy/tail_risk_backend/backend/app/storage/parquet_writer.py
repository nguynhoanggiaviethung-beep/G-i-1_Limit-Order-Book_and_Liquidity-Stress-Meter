"""Partitioned Parquet writer for sampled local books and features."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.orderbook.book_state import LocalOrderBook
from app.storage.manifest import append_manifest, sha256_file


class LOBParquetWriter:
    def __init__(self, data_dir: Path, flush_rows: int = 500, flush_seconds: int = 10) -> None:
        self.data_dir = Path(data_dir)
        self.flush_rows = max(1, flush_rows)
        self.flush_seconds = max(1, flush_seconds)
        self._features: list[dict] = []
        self._snapshots: list[dict] = []
        self._last_flush = datetime.now(timezone.utc)

    def append(self, feature_row: dict, book: LocalOrderBook) -> None:
        self._features.append(dict(feature_row))
        self._snapshots.append({
            "symbol": book.symbol,
            "sample_time_utc": feature_row["sample_time_utc"],
            "event_time_utc": feature_row.get("event_time_utc"),
            "received_at_utc": feature_row.get("received_at_utc"),
            "last_update_id": book.last_update_id,
            "book_valid": bool(book.valid),
            "bid_levels_top_20_json": json.dumps([[str(p), str(q)] for p, q in book.top_levels("bid", 20)]),
            "ask_levels_top_20_json": json.dumps([[str(p), str(q)] for p, q in book.top_levels("ask", 20)]),
        })

    def flush_if_due(self) -> list[Path]:
        age = (datetime.now(timezone.utc) - self._last_flush).total_seconds()
        if len(self._features) >= self.flush_rows or (self._features and age >= self.flush_seconds):
            return self.flush()
        return []

    def flush(self) -> list[Path]:
        if not self._features:
            self._last_flush = datetime.now(timezone.utc)
            return []
        written: list[Path] = []
        for kind, rows in (("features", self._features), ("lob_snapshots", self._snapshots)):
            frame = pd.DataFrame(rows)
            times = pd.to_datetime(frame["sample_time_utc"], utc=True)
            frame["sample_time_utc"] = times
            group = frame.assign(_date=times.dt.strftime("%Y-%m-%d"))
            for date, part in group.groupby("_date", sort=True):
                part = part.drop(columns="_date")
                symbol = str(part["symbol"].iloc[0]).upper()
                base = self.data_dir / ("features/lob" if kind == "features" else "processed/lob_snapshots")
                directory = base / f"symbol={symbol}" / f"date={date}"
                directory.mkdir(parents=True, exist_ok=True)
                file_name = f"part-{datetime.now(timezone.utc):%H%M%S}-{uuid.uuid4().hex[:10]}.parquet"
                path = directory / file_name
                part.to_parquet(path, index=False)
                written.append(path)
                sample_times = pd.to_datetime(part["sample_time_utc"], utc=True)
                append_manifest(self.data_dir, symbol, {
                    "artifact_type": kind,
                    "path": str(path.relative_to(self.data_dir)),
                    "symbol": symbol,
                    "date_utc": date,
                    "rows": int(len(part)),
                    "start_time_utc": sample_times.min().isoformat(),
                    "end_time_utc": sample_times.max().isoformat(),
                    "sha256": sha256_file(path),
                    "collector_version": "lob-collector-1",
                    "book_valid_rows": int(part["book_valid"].fillna(False).sum()),
                })
        self._features.clear()
        self._snapshots.clear()
        self._last_flush = datetime.now(timezone.utc)
        return written
