"""Replay CryptoHFTData hourly Binance Spot Parquet files into the project grid."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq
import zstandard as zstd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPOSITORY_ROOT, load_settings
from app.data.cryptohft_replay import OneSecondBookReplay


ZSTD_MAGIC = bytes.fromhex("28b52ffd")
MAX_PROVIDER_RECEIVE_TIME_REGRESSION_NS = 100_000_000
# Provider messages that cannot be turned into book events are skipped, never silently:
# every skip is counted by reason and printed at the end of each symbol.
SKIPPED_EVENTS: Counter = Counter()
SKIPPED_EXAMPLES: dict[str, str] = {}


def _display_path(path: Path) -> str:
    """Repo-relative path when possible; absolute otherwise (e.g. output on another drive)."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != ZSTD_MAGIC:
        return pq.read_table(path, columns=columns).to_pandas()
    # Stream-decompress to a temporary file. ZstdDecompressor.decompress() fails on
    # frames written by streaming encoders (no content size in the header), and
    # holding compressed + decompressed bytes in RAM is wasteful for large hours.
    with tempfile.TemporaryFile(dir=path.parent) as scratch:
        with path.open("rb") as source:
            zstd.ZstdDecompressor().copy_stream(source, scratch)
        scratch.seek(0)
        return pq.ParquetFile(scratch).read(columns=columns).to_pandas()


def _as_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _event_clock(received_time_ns) -> tuple[int, int]:
    received_ns = int(received_time_ns)
    return received_ns, received_ns // 1_000_000


def _book_events(frame: pd.DataFrame, symbol: str) -> Iterator[tuple[int, dict]]:
    """Group level rows into messages while preserving provider file order."""
    group_columns = (
        "received_time", "event_time", "event_type", "first_update_id",
        "final_update_id", "last_update_id",
    )
    def build(key, bid_levels, ask_levels):
        received_ns, exchange_time, event_type, first_id, final_id, last_id = key
        received_ns, received_ms = _event_clock(received_ns)
        exchange_ms = _as_int(exchange_time)
        if event_type == "snapshot":
            snapshot_id = _as_int(last_id)
            if snapshot_id is None:
                snapshot_id = _as_int(final_id)
            if snapshot_id is None or not bid_levels or not ask_levels:
                raise ValueError(
                    f"Empty or unsequenced provider snapshot for {symbol} at {received_ms}; "
                    "book state cannot be reconstructed safely"
                )
            payload = {
                "kind": "snapshot",
                "symbol": symbol,
                "event_time_ms": received_ms,
                "book_event_time_ms": received_ms,
                "exchange_event_time_ms": exchange_ms,
                "lastUpdateId": snapshot_id,
                "bids": bid_levels,
                "asks": ask_levels,
            }
        elif event_type == "update":
            first_update_id = _as_int(first_id)
            final_update_id = _as_int(final_id)
            if first_update_id is None or final_update_id is None:
                raise ValueError(f"Provider depth update has no sequence IDs for {symbol} at {received_ms}")
            payload = {
                "kind": "depth_update",
                "symbol": symbol,
                "event_time_ms": received_ms,
                "E": received_ms,
                "exchange_event_time_ms": exchange_ms,
                "U": first_update_id,
                "u": final_update_id,
                "b": bid_levels,
                "a": ask_levels,
            }
        else:
            raise ValueError(f"Unsupported provider order-book event_type: {event_type!r}")
        return received_ns, payload

    if frame.empty:
        return

    # A file has one row per price level; vectorize message-boundary detection
    # instead of performing several pandas null checks for every row.
    key_frame = frame.loc[:, group_columns].copy()
    numeric_keys = ("received_time", "event_time", "first_update_id", "final_update_id", "last_update_id")
    for column in numeric_keys:
        key_frame[column] = pd.to_numeric(key_frame[column], errors="coerce").fillna(-1).astype("int64")
    key_frame["event_type"] = key_frame["event_type"].astype("string").fillna("")
    key_values = key_frame.to_numpy(copy=False)
    starts = np.zeros(len(frame), dtype=bool)
    starts[0] = True
    starts[1:] = np.any(key_values[1:] != key_values[:-1], axis=1)
    boundaries = np.flatnonzero(starts)
    ends = np.r_[boundaries[1:], len(frame)]

    sides = frame["side"].astype("string").str.lower().fillna("").to_numpy()
    prices = frame["price"].to_numpy()
    quantities = frame["quantity"].to_numpy()
    previous_received_ns = -1
    for start, end in zip(boundaries, ends):
        key_row = key_values[start]
        normalized_key = []
        for column, value in zip(group_columns, key_row):
            if column in numeric_keys:
                numeric_value = int(value)
                normalized_key.append(None if numeric_value == -1 else numeric_value)
            else:
                normalized_key.append(str(value))
        key = tuple(normalized_key)
        if key[0] is None:
            raise ValueError(f"Provider order-book message has no receive time for {symbol}")
        received_ns = int(key[0])
        regression_ns = 0
        if received_ns < previous_received_ns:
            regression_ns = previous_received_ns - received_ns
            if regression_ns > MAX_PROVIDER_RECEIVE_TIME_REGRESSION_NS:
                raise ValueError(
                    f"Provider book receive time moved backwards by {regression_ns / 1_000_000:.3f} ms "
                    f"for {symbol}, exceeding the {MAX_PROVIDER_RECEIVE_TIME_REGRESSION_NS / 1_000_000:.0f} ms correction limit"
                )
            # Preserve the provider's message/sequence order. Small receive-clock
            # regressions are clamped to the preceding receive time and recorded in QA.
            key = (previous_received_ns, *key[1:])
            received_ns = previous_received_ns
        previous_received_ns = received_ns
        block_sides = sides[start:end]
        bid_mask = block_sides == "bid"
        ask_mask = block_sides == "ask"
        if np.any(~(bid_mask | ask_mask | (block_sides == "noop"))):
            bad_side = str(block_sides[np.flatnonzero(~(bid_mask | ask_mask | (block_sides == "noop")))[0]])
            raise ValueError(f"Unsupported provider order-book side: {bad_side!r}")
        bids = [[str(price), str(quantity)] for price, quantity in zip(prices[start:end][bid_mask], quantities[start:end][bid_mask])]
        asks = [[str(price), str(quantity)] for price, quantity in zip(prices[start:end][ask_mask], quantities[start:end][ask_mask])]
        try:
            event_ns, event = build(key, bids, asks)
        except ValueError as exc:
            reason = re.sub(r"\d+", "#", str(exc))[:140]
            SKIPPED_EVENTS[reason] += 1
            SKIPPED_EXAMPLES.setdefault(reason, str(exc))
            continue
        if regression_ns:
            event["provider_receive_time_regression_ms"] = regression_ns / 1_000_000
        yield event_ns, event


def _trade_events(frame: pd.DataFrame, symbol: str) -> Iterator[tuple[int, dict]]:
    if frame.empty:
        return
    frame = frame.sort_values(["received_time", "trade_id"], kind="stable").copy()
    received_ns = pd.to_numeric(frame["received_time"], errors="coerce")
    buyer_maker = frame["is_buyer_maker"]
    if received_ns.isna().any() or buyer_maker.isna().any():
        bad_time = received_ns[received_ns.isna()].index[0] if received_ns.isna().any() else received_ns.index[0]
        raise ValueError(f"Provider trade has missing receive time or aggressor side for {symbol} near row {bad_time}")
    quantities = pd.to_numeric(frame["quantity"], errors="coerce")
    prices = pd.to_numeric(frame["price"], errors="coerce")
    valid = quantities.notna() & prices.notna() & (quantities > 0) & (prices > 0)
    frame = frame.loc[valid].copy()
    if frame.empty:
        return
    received_ns = pd.to_numeric(frame["received_time"], errors="raise").astype("int64")
    received_ms = received_ns // 1_000_000
    # Match OneSecondBookReplay's (t-1s, t] sampling buckets.
    frame["_bucket"] = ((received_ms + 999) // 1000).astype("int64")
    frame["_quantity"] = pd.to_numeric(frame["quantity"], errors="raise").astype(float)
    frame["_price"] = pd.to_numeric(frame["price"], errors="raise").astype(float)
    frame["_buyer_maker"] = frame["is_buyer_maker"].astype(bool)
    frame["_buy"] = np.where(frame["_buyer_maker"], 0.0, frame["_quantity"])
    frame["_sell"] = np.where(frame["_buyer_maker"], frame["_quantity"], 0.0)
    frame["_received_ns"] = received_ns.to_numpy()
    frame["_received_ms"] = received_ms.to_numpy()
    exchange_time = pd.to_numeric(frame["trade_time"], errors="coerce")
    exchange_time = exchange_time.fillna(pd.to_numeric(frame["event_time"], errors="coerce"))
    frame["_exchange_time_ms"] = exchange_time.fillna(-1).astype("int64")
    grouped = frame.groupby("_bucket", sort=True, observed=True)
    totals = grouped.agg(
        aggressive_buy_volume=("_buy", "sum"),
        aggressive_sell_volume=("_sell", "sum"),
        trade_count=("_price", "size"),
        last_trade_price=("_price", "last"),
        last_trade_time_ms=("_received_ms", "last"),
        last_exchange_trade_time_ms=("_exchange_time_ms", "last"),
        received_ns=("_received_ns", "last"),
    )
    for bucket, row in totals.iterrows():
        received_ms = int(row.last_trade_time_ms)
        yield int(row.received_ns), {
            "kind": "trade_aggregate",
            "symbol": symbol,
            "event_time_ms": received_ms,
            "trade_timestamp_ms": received_ms,
            "aggressive_buy_volume": float(row.aggressive_buy_volume),
            "aggressive_sell_volume": float(row.aggressive_sell_volume),
            "trade_count": int(row.trade_count),
            "last_trade_price": str(row.last_trade_price),
            "last_trade_time_ms": received_ms,
            "last_exchange_trade_time_ms": int(row.last_exchange_trade_time_ms) if int(row.last_exchange_trade_time_ms) >= 0 else None,
            "bucket_second": int(bucket),
        }


def _write_day(
    output_root: Path,
    manifest_root: Path,
    symbol: str,
    day: str,
    rows: list[dict],
    qa: dict,
    source_files: list[str],
) -> None:
    destination = output_root / f"symbol={symbol}" / f"date={day}"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "book_1s.parquet"
    if rows:
        temporary = output.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, output)
    else:
        output.unlink(missing_ok=True)
    expected = int(qa.get("expected_seconds", 0))
    valid = int(qa.get("valid_seconds", 0))
    payload = {
        **qa,
        "symbol": symbol,
        "date_utc": day,
        "coverage": float(valid / expected) if expected else None,
        "source_rows_validated": True,
        "event_schema": "cryptohftdata_binance_spot_l2_trades.v1",
        "source": "CryptoHFTData Binance Spot public archive",
        "timestamp_basis": "provider received_time, converted from nanoseconds to UTC milliseconds",
        "source_files": sorted(set(source_files)),
    }
    day_manifest_root = manifest_root / f"symbol={symbol}" / f"date={day}"
    day_manifest_root.mkdir(parents=True, exist_ok=True)
    qa_path = day_manifest_root / "qa.json"
    temporary_qa = qa_path.with_suffix(".json.tmp")
    temporary_qa.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_qa, qa_path)
    target = _display_path(output) if rows else "KHÔNG ghi file parquet (0 giây hợp lệ)"
    print(
        f"{symbol} {day}: {valid:,}/{expected:,} seconds valid, "
        f"{qa.get('sequence_gap_count', 0)} sequence gaps, "
        f"{qa.get('invalid_book_event_count', 0)} invalid-book events -> {target}",
        flush=True,
    )


def _replay_symbol(
    symbol: str,
    start: date,
    end: date,
    input_root: Path,
    output_root: Path,
    manifest_root: Path,
    tick_size: float,
    max_event_age: int,
) -> int:
    replay = OneSecondBookReplay(symbol, tick_size, max_event_age_seconds=max_event_age, levels=20)
    buffered_day: str | None = None
    buffered_rows: list[dict] = []
    written_days: set[str] = set()
    day_sources: dict[str, list[str]] = {}
    missing_orderbook_hours = 0
    missing_trade_hours = 0
    hours_seen = 0

    first_day, last_day = start.isoformat(), end.isoformat()

    def accept(row: dict) -> None:
        nonlocal buffered_day, buffered_rows
        row_day = pd.Timestamp(row["timestamp_utc"]).date().isoformat()
        if not first_day <= row_day <= last_day:
            # The 00:00:00 sample of the day after --end-date belongs to the next chunk.
            return
        if buffered_day is None:
            buffered_day = row_day
        elif row_day != buffered_day:
            _write_day(output_root, manifest_root, symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}), day_sources.get(buffered_day, []))
            written_days.add(buffered_day)
            buffered_day, buffered_rows = row_day, []
        buffered_rows.append(row)

    for day in _date_range(start, end):
        day_text = day.isoformat()
        day_sources.setdefault(day_text, [])
        for hour in range(24):
            hours_seen += 1
            hour_text = f"{hour:02d}"
            hour_dir = input_root / day_text / hour_text
            orderbook_path = hour_dir / f"{symbol}_orderbook.parquet.zst"
            trades_path = hour_dir / f"{symbol}_trades.parquet.zst"
            if hour % 4 == 0:
                print(f"{symbol} {day_text} {hour_text}:00 UTC: replaying archive hours", flush=True)
            if not orderbook_path.is_file():
                missing_orderbook_hours += 1
                continue

            book = _read_parquet(orderbook_path, [
                "received_time", "event_time", "event_type", "first_update_id",
                "final_update_id", "last_update_id", "side", "price", "quantity",
            ])
            book_events = _book_events(book, symbol)
            day_sources[day_text].append(str(orderbook_path.relative_to(input_root)))

            if trades_path.is_file():
                trades = _read_parquet(trades_path, [
                    "received_time", "event_time", "trade_id", "price", "quantity", "trade_time", "is_buyer_maker",
                ])
                trade_events = _trade_events(trades, symbol)
                day_sources[day_text].append(str(trades_path.relative_to(input_root)))
            else:
                missing_trade_hours += 1
                trade_events = iter(())

            for _, event in heapq.merge(book_events, trade_events, key=lambda item: item[0]):
                for output_row in replay.feed(event):
                    accept(output_row)

    for output_row in replay.finish():
        accept(output_row)
    if buffered_day is not None:
        _write_day(output_root, manifest_root, symbol, buffered_day, buffered_rows, replay.daily_qa.get(buffered_day, {}), day_sources.get(buffered_day, []))
        written_days.add(buffered_day)
    for completed_day in sorted(set(replay.daily_qa) - written_days):
        if not first_day <= completed_day <= last_day:
            continue
        _write_day(output_root, manifest_root, symbol, completed_day, [], replay.daily_qa[completed_day], day_sources.get(completed_day, []))

    print(
        f"{symbol}: {replay.qa['valid_seconds']:,}/{replay.qa['expected_seconds']:,} valid seconds; "
        f"{replay.qa['sequence_gap_count']:,} gaps; {missing_orderbook_hours:,}/{hours_seen:,} missing orderbook hours; "
        f"{missing_trade_hours:,}/{hours_seen:,} missing trade hours; source=CryptoHFTData Spot; "
        f"grid={output_root}",
        flush=True,
    )
    if SKIPPED_EVENTS:
        print(f"{symbol}: BỎ QUA {sum(SKIPPED_EVENTS.values()):,} message của nhà cung cấp, theo lý do:", flush=True)
        for reason, count in SKIPPED_EVENTS.most_common(8):
            print(f"  {count:>10,} x {reason}\n             ví dụ: {SKIPPED_EXAMPLES[reason]}", flush=True)
        SKIPPED_EVENTS.clear()
        SKIPPED_EXAMPLES.clear()
    return replay.qa["valid_seconds"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=REPOSITORY_ROOT / "config" / "settings.yaml", help="Research settings; defaults to the isolated submission archive settings")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--symbol", action="append", choices=("BTCUSDT", "ETHUSDT"), help="Repeat for each symbol; defaults to both")
    parser.add_argument("--input-root", type=Path, help="Defaults to data_raw/cryptohftdata/binance_spot")
    parser.add_argument("--output-root", type=Path, help="Defaults to the project's replay grid directory")
    parser.add_argument("--manifest-root", type=Path, help="Defaults to outputs/manifests")
    parser.add_argument("--max-event-age-seconds", type=int, default=2)
    args = parser.parse_args()
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date")
    settings_path = args.settings if args.settings.is_absolute() else REPOSITORY_ROOT / args.settings
    settings = load_settings(settings_path)
    input_root = args.input_root or (REPOSITORY_ROOT / "data_raw" / "cryptohftdata" / "binance_spot")
    output_root = args.output_root or (settings.data_dir / "grid")
    manifest_root = args.manifest_root or (REPOSITORY_ROOT / "outputs" / "manifests")
    if not input_root.is_absolute():
        input_root = REPOSITORY_ROOT / input_root
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    if not manifest_root.is_absolute():
        manifest_root = REPOSITORY_ROOT / manifest_root
    symbols = tuple(dict.fromkeys(args.symbol or ("BTCUSDT", "ETHUSDT")))
    if args.max_event_age_seconds < 0:
        raise SystemExit("--max-event-age-seconds must be non-negative")
    total_valid = 0
    for symbol in symbols:
        total_valid += _replay_symbol(
            symbol, args.start_date, args.end_date, input_root, output_root,
            manifest_root, settings.tick_size_for(symbol), args.max_event_age_seconds,
        )
    if not total_valid:
        print("No valid L2 seconds were reconstructed. Check archive coverage and checkpoint snapshots.")
        return 2
    print(f"Historical archive replay complete: {total_valid:,} valid symbol-seconds -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())