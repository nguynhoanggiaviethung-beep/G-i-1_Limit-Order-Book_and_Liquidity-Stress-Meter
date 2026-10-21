"""Append-only JSONL storage for raw LOB events and REST snapshots."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage.manifest import append_manifest, sha256_file


def _date_from_ms(value: int | None) -> str:
    stamp = datetime.fromtimestamp((value or 0) / 1000, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%d")


class RawLOBWriter:
    def __init__(self, data_dir: Path, max_file_bytes: int = 64 * 1024 * 1024) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "raw" / "lob"
        self.max_file_bytes = max_file_bytes
        self._connection_paths: dict[str, set[Path]] = {}
        self._lock = threading.Lock()

    def _append(self, symbol: str, event_date: str, connection_id: str, record: dict[str, Any]) -> Path:
        directory = self.root / symbol.lower() / f"date={event_date}"
        directory.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            path = directory / f"events-{connection_id[:8]}-{index:04d}.jsonl"
            if not path.exists() or path.stat().st_size < self.max_file_bytes:
                break
            index += 1
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
        return path

    def write_depth_event(
        self,
        symbol: str,
        event: dict[str, Any],
        stream: str,
        connection_id: str,
        collector_version: str,
        received_at_utc: str,
    ) -> Path:
        payload = {
            "source": "binance_spot_ws_diff_depth",
            "symbol": symbol.upper(),
            "stream": stream,
            "connection_id": connection_id,
            "collector_version": collector_version,
            "event_time_utc": _iso_ms(event.get("E")),
            "transaction_time_utc": _iso_ms(event.get("T")),
            "received_at_utc": received_at_utc,
            "payload": event,
        }
        path = self._append(symbol.upper(), _date_from_ms(event.get("E")), connection_id, payload)
        with self._lock:
            self._connection_paths.setdefault(connection_id, set()).add(path)
        return path

    def write_snapshot(
        self,
        symbol: str,
        snapshot: dict[str, Any],
        connection_id: str,
        collector_version: str,
        received_at_utc: str,
    ) -> Path:
        payload = {
            "source": "binance_spot_rest_depth_snapshot",
            "symbol": symbol.upper(),
            "stream": "GET /api/v3/depth",
            "connection_id": connection_id,
            "collector_version": collector_version,
            "snapshot_received_at_utc": received_at_utc,
            "payload": snapshot,
        }
        received_date = datetime.fromisoformat(received_at_utc).astimezone(timezone.utc).strftime("%Y-%m-%d")
        path = self._append(symbol.upper(), received_date, connection_id, payload)
        with self._lock:
            self._connection_paths.setdefault(connection_id, set()).add(path)
        return path

    def finish_connection(self, symbol: str, connection_id: str) -> None:
        """Record checksums and time coverage after a connection's raw files close."""
        with self._lock:
            paths = sorted(self._connection_paths.pop(connection_id, set()))
        for path in paths:
            timestamps: list[str] = []
            row_count = 0
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    stamp = record.get("event_time_utc") or record.get("snapshot_received_at_utc")
                    if stamp:
                        timestamps.append(stamp)
                    row_count += 1
            append_manifest(self.data_dir, symbol.upper(), {
                "artifact_type": "raw_events_jsonl",
                "path": str(path.relative_to(self.data_dir)),
                "symbol": symbol.upper(),
                "rows_in_file": row_count,
                "start_time_utc": min(timestamps) if timestamps else None,
                "end_time_utc": max(timestamps) if timestamps else None,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "connection_id": connection_id,
                "collector_version": "lob-collector-1",
            })


def _iso_ms(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()
