"""Partitioned, append-only storage helpers for market-data captures."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any


def _day_for(record: dict[str, Any]) -> str:
    timestamp = record.get("event_time_ms", record.get("T", record.get("E")))
    if timestamp is None:
        timestamp = record.get("receive_time_ms")
    if timestamp is None:
        timestamp = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    return datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc).date().isoformat()


class PartitionedGzipJsonlWriter:
    """Write newline-delimited JSON into symbol/date partitions and size-capped parts."""

    def __init__(
        self,
        root: Path,
        symbol: str,
        max_part_bytes: int,
        suffix: str = "jsonl.gz",
        flush_rows: int = 500,
        flush_seconds: int = 10,
    ) -> None:
        self.root = Path(root)
        self.symbol = symbol.upper()
        self.max_part_bytes = int(max_part_bytes)
        self.suffix = suffix
        self.flush_rows = int(flush_rows)
        self.flush_seconds = int(flush_seconds)
        if self.max_part_bytes < 1 or self.flush_rows < 1 or self.flush_seconds < 1:
            raise ValueError("Partition size and flush settings must be positive")
        self._handle = None
        self._day: str | None = None
        self._part = 0
        self._uncompressed_bytes = 0
        self._records_since_flush = 0
        self._last_flush = time.monotonic()
        self._paths: list[Path] = []

    def _open(self, day: str) -> None:
        self._close_part()
        self._day = day
        directory = self.root / f"symbol={self.symbol}" / f"date={day}"
        directory.mkdir(parents=True, exist_ok=True)
        while True:
            path = directory / f"part-{self._part:05d}.{self.suffix}"
            self._part += 1
            if not path.exists():
                break
        self._handle = gzip.open(path, "wt", encoding="utf-8", newline="\n")
        self._paths.append(path)
        self._uncompressed_bytes = 0
        self._records_since_flush = 0
        self._last_flush = time.monotonic()

    def _write_line(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("The JSONL writer is not open")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)
        self._handle.write(line + "\n")
        self._uncompressed_bytes += len((line + "\n").encode("utf-8"))
        self._records_since_flush += 1
        if self._records_since_flush >= self.flush_rows or time.monotonic() - self._last_flush >= self.flush_seconds:
            self._handle.flush()
            self._records_since_flush = 0
            self._last_flush = time.monotonic()

    def write(self, record: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> None:
        day = _day_for(record)
        needs_new_part = (
            self._handle is None
            or day != self._day
            or self._uncompressed_bytes >= self.max_part_bytes
        )
        if needs_new_part:
            self._open(day)
            if checkpoint is not None and record.get("kind") != "snapshot":
                self._write_line(checkpoint)
        self._write_line(record)

    def _close_part(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        self._close_part()

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    def file_metadata(self) -> list[dict[str, Any]]:
        self.close()
        output = []
        for path in self._paths:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            output.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        return output
