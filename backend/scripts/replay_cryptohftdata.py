"""Replay an ordered L2/trade event stream or capture folder into daily book grids."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.cryptohft_replay import OneSecondBookReplay


def _read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def _record_time_ms(record: dict) -> int:
    payload = record.get("data", record)
    for name in ("event_time_ms", "E", "T", "trade_timestamp_ms", "timestamp_ms"):
        value = payload.get(name) if isinstance(payload, dict) else None
        if value is not None:
            return int(value)
    raise ValueError("Every replay event needs an exchange event timestamp")


def _read_event_path(path: Path) -> Iterator[dict]:
    if path.is_file():
        if path.suffix.lower() not in {".gz", ".jsonl", ".ndjson"} and not path.name.endswith(".jsonl.gz"):
            raise ValueError(f"Expected JSONL/JSONL.GZ input, got {path}")
        yield from _read_jsonl(path)
        return
    files = sorted(path.rglob("*.jsonl.gz")) + sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.ndjson"))
    if not files:
        raise ValueError(f"No JSONL event files found under {path}")
    streams = [_read_jsonl(file) for file in files]
    yield from heapq.merge(*streams, key=_record_time_ms)


def _write_day(output_root: Path, symbol: str, day: str, rows: list[dict], qa: dict[str, int]) -> None:
    destination = output_root / f"symbol={symbol}" / f"date={day}"
    destination.mkdir(parents=True, exist_ok=True)
    if rows:
        output = destination / "book_1s.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, output)
    else:
        output = destination / "book_1s.parquet"
        output.unlink(missing_ok=True)
    expected = int(qa.get("expected_seconds", 0))
    valid = int(qa.get("valid_seconds", 0))
    payload = {
        **qa,
        "symbol": symbol,
        "date_utc": day,
        "coverage": float(valid / expected) if expected else None,
        "source_rows_validated": True,
        "event_schema": "binance_l2_trade_normalized.v1",
    }
    manifest_root = REPOSITORY_ROOT / "outputs" / "manifests" / f"symbol={symbol}" / f"date={day}"
    manifest_root.mkdir(parents=True, exist_ok=True)
    qa_path = manifest_root / "qa.json"
    temporary_qa = qa_path.with_suffix(".json.tmp")
    temporary_qa.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_qa, qa_path)
    print(
        f"{symbol} {day}: {valid:,}/{expected:,} seconds valid, "
        f"{qa.get('sequence_gap_count', 0)} sequence gaps, "
        f"{qa.get('invalid_book_event_count', 0)} invalid-book events"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Ordered JSONL/JSONL.GZ file or a capture events directory")
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--tick-size", type=float, help="Override configured price tick size")
    parser.add_argument("--output-root", type=Path, help="Defaults to the configured processed-data grid directory")
    parser.add_argument("--max-event-age-seconds", type=int, default=2)
    args = parser.parse_args()
    if not args.input.exists() or not (args.input.is_file() or args.input.is_dir()):
        raise SystemExit(f"Input event stream or capture directory does not exist: {args.input}")
    settings = load_settings()
    tick_size = args.tick_size if args.tick_size is not None else settings.tick_size_for(args.symbol)
    output_root = args.output_root or (settings.data_dir / "grid")
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    replay = OneSecondBookReplay(args.symbol, tick_size, args.max_event_age_seconds, levels=20)
    buffered_day: str | None = None
    buffered_rows: list[dict] = []
    written_days: set[str] = set()

    def flush_before(day: str) -> None:
        nonlocal buffered_day, buffered_rows
        if buffered_day is not None and buffered_day < day:
            _write_day(output_root, args.symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}))
            written_days.add(buffered_day)
            buffered_day, buffered_rows = None, []
        for completed_day in sorted(key for key in replay.daily_qa if key < day and key not in written_days):
            _write_day(output_root, args.symbol, completed_day, [], replay.daily_qa[completed_day])
            written_days.add(completed_day)

    for record in _read_event_path(args.input):
        for row in replay.feed(record):
            day = pd.Timestamp(row["timestamp_utc"]).date().isoformat()
            if buffered_day is None:
                buffered_day = day
            elif day != buffered_day:
                _write_day(output_root, args.symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}))
                written_days.add(buffered_day)
                buffered_day, buffered_rows = day, []
            buffered_rows.append(row)
        if replay.current_second is not None:
            current_day = datetime.fromtimestamp(replay.current_second, tz=timezone.utc).date().isoformat()
            flush_before(current_day)

    for row in replay.finish():
        day = pd.Timestamp(row["timestamp_utc"]).date().isoformat()
        if buffered_day is None:
            buffered_day = day
        elif day != buffered_day:
            _write_day(output_root, args.symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}))
            written_days.add(buffered_day)
            buffered_day, buffered_rows = day, []
        buffered_rows.append(row)
    if buffered_day is not None:
        _write_day(output_root, args.symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}))
        written_days.add(buffered_day)
    for day in sorted(set(replay.daily_qa) - written_days):
        _write_day(output_root, args.symbol, day, [], replay.daily_qa[day])
    if not replay.qa["expected_seconds"]:
        print("No replayable events found in the input stream")
        return 2
    print(f"Replay complete: {replay.qa['valid_seconds']:,}/{replay.qa['expected_seconds']:,} valid seconds -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
