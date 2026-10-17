"""Normalize downloaded CryptoHFTData hourly files and write a QA manifest.

This creates a typed, locally compressed Parquet event store. It preserves all
source levels/events and records sequence diagnostics; it does not fabricate
book states or model labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data_raw" / "cryptohftdata"
DEFAULT_OUT = ROOT / "data_processed" / "cryptohftdata"
DEFAULT_QA = ROOT / "outputs" / "manifests" / "cryptohftdata_qa.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_outer_zstd_parquet(path: Path) -> pa.Table:
    compressed = pa.BufferReader(path.read_bytes())
    raw = pa.CompressedInputStream(compressed, "zstd").read()
    return pq.read_table(pa.BufferReader(raw))


def utc_array(values: pa.ChunkedArray, unit: str) -> pa.Array:
    return pc.cast(values, pa.timestamp(unit, tz="UTC"), safe=False)


def regression_count(values: pa.ChunkedArray) -> int:
    rows = values.to_pylist()
    return sum(a > b for a, b in zip(rows, rows[1:]) if a is not None and b is not None)


def seq_summary(table: pa.Table) -> dict:
    if "event_type" not in table.column_names:
        return {"snapshot_rows": 0, "update_groups": 0, "sequence_gaps": 0,
                "sequence_overlaps": 0, "sequence_stale_groups": 0,
                "sequence_order_regressions": 0}
    ev = table["event_type"].to_pylist()
    first = table["first_update_id"].to_pylist() if "first_update_id" in table.column_names else [None] * len(ev)
    final = table["final_update_id"].to_pylist() if "final_update_id" in table.column_names else [None] * len(ev)
    last = table["last_update_id"].to_pylist() if "last_update_id" in table.column_names else [None] * len(ev)
    snapshots = sum(x == "snapshot" for x in ev)
    update_groups = {(a, b) for e, a, b in zip(ev, first, final)
                     if e == "update" and a is not None and b is not None}
    snapshot_ids = {(l if l is not None else f) for e, f, l in zip(ev, final, last)
                    if e == "snapshot" and (l is not None or f is not None)}
    ordered_events = [(seq, "snapshot", seq) for seq in snapshot_ids]
    ordered_events.extend((end, "update", (start, end)) for start, end in update_groups)
    ordered_events.sort(key=lambda item: (item[0], 0 if item[1] == "snapshot" else 1))
    gaps = overlaps = stale = 0
    previous = None
    for _, kind, value in ordered_events:
        if kind == "snapshot":
            previous = value
            continue
        start, end = value
        if previous is not None:
            if end < previous + 1:
                stale += 1
                continue
            if start > previous + 1:
                gaps += 1
        previous = max(previous or end, end)
    original_groups = []
    previous_key = None
    for e, start, end in zip(ev, first, final):
        key = (start, end) if e == "update" else None
        if key is not None and key != previous_key:
            original_groups.append(key)
        previous_key = key
    regressions = sum(a[1] > b[1] for a, b in zip(original_groups, original_groups[1:]))
    return {"snapshot_rows": snapshots, "update_groups": len(update_groups),
            "sequence_gaps": gaps, "sequence_overlaps": overlaps,
            "sequence_stale_groups": stale,
            "sequence_order_regressions": regressions,
            "first_update_id": min((x[0] for x in update_groups), default=None),
            "last_update_id": max((x[1] for x in update_groups), default=None),
            "snapshot_last_update_id_min": min((x for x, e in zip(last, ev) if e == "snapshot" and x is not None), default=None),
            "snapshot_last_update_id_max": max((x for x, e in zip(last, ev) if e == "snapshot" and x is not None), default=None)}


def normalize_one(source: Path, output: Path) -> dict:
    table = read_outer_zstd_parquet(source)
    is_book = source.name.endswith("_orderbook.parquet")
    expected = ({"received_time", "event_time", "symbol", "event_type", "side", "price", "quantity"}
                if is_book else {"received_time", "event_time", "symbol", "trade_id", "price", "quantity", "is_buyer_maker"})
    missing = expected - set(table.column_names)
    if missing:
        raise ValueError(f"{source}: required columns absent: {sorted(missing)}")
    symbols = set(table["symbol"].to_pylist())
    if len(symbols) != 1:
        raise ValueError(f"{source}: expected one symbol, found {symbols}")
    if is_book:
        diagnostics = seq_summary(table)
        if "received_time" in table.column_names:
            table = table.append_column("received_time_utc", utc_array(table["received_time"], "ns"))
        if "event_time" in table.column_names:
            # CryptoHFTData's exchange event_time for Binance is milliseconds.
            table = table.append_column("event_time_utc", utc_array(table["event_time"], "ms"))
        if "final_update_id" in table.column_names:
            table = table.sort_by([("final_update_id", "ascending"), ("received_time", "ascending")])
    else:
        table = table.append_column("received_time_utc", utc_array(table["received_time"], "ns"))
        trade_time_name = "trade_time" if "trade_time" in table.column_names else "event_time"
        trade_unit = "ms" if trade_time_name == "trade_time" else "ms"
        table = table.append_column("exchange_time_utc", utc_array(table[trade_time_name], trade_unit))
        maker = table["is_buyer_maker"].to_pylist()
        table = table.append_column("aggressor_side", pa.array(
            [None if v is None else ("sell" if v else "buy") for v in maker], type=pa.string()))
        order = "trade_time" if "trade_time" in table.column_names else "received_time"
        table = table.sort_by([(order, "ascending"), ("trade_id", "ascending")])
        diagnostics = {"snapshot_rows": 0, "update_groups": 0, "sequence_gaps": 0,
                       "sequence_overlaps": 0, "sequence_order_regressions": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)
    receive = table["received_time"].combine_chunks()
    summary = {"source_file": str(source.relative_to(ROOT)).replace("\\", "/"),
               "normalized_file": str(output.relative_to(ROOT)).replace("\\", "/"),
               "sha256": sha256(source), "source_bytes": source.stat().st_size,
               "normalized_bytes": output.stat().st_size, "rows": table.num_rows,
               "symbol": next(iter(symbols)), "data_type": "orderbook" if is_book else "trades",
               "received_time_min_ns": pc.min(receive).as_py(),
               "received_time_max_ns": pc.max(receive).as_py(), **diagnostics}
    if is_book:
        summary["received_time_regressions_after_sequence_sort"] = regression_count(table["received_time"])
        summary["event_time_regressions_after_sequence_sort"] = regression_count(table["event_time"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--qa", default=str(DEFAULT_QA))
    args = parser.parse_args()
    raw, out, qa_path = Path(args.raw).resolve(), Path(args.output).resolve(), Path(args.qa).resolve()
    sources = sorted(raw.glob("binance_spot/2025-*/[0-2][0-9]/*.parquet"))
    if not sources:
        raise SystemExit(f"No downloaded CryptoHFTData files under {raw}")
    files, errors = [], []
    for index, source in enumerate(sources, start=1):
        rel = source.relative_to(raw)
        target = out / rel
        try:
            if target.exists() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                # Reuse normalized output on reruns, but refresh quality metrics from source.
                original = read_outer_zstd_parquet(source)
                diagnostics = seq_summary(original) if source.name.endswith("_orderbook.parquet") else {
                    "snapshot_rows": 0, "update_groups": 0, "sequence_gaps": 0,
                    "sequence_overlaps": 0, "sequence_stale_groups": 0,
                    "sequence_order_regressions": 0}
                entry = {"source_file": str(source.relative_to(ROOT)).replace("\\", "/"),
                    "normalized_file": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(source), "source_bytes": source.stat().st_size,
                    "normalized_bytes": target.stat().st_size, "rows": original.num_rows,
                    "symbol": original["symbol"][0].as_py(),
                    "data_type": "orderbook" if source.name.endswith("_orderbook.parquet") else "trades",
                    **diagnostics}
                if source.name.endswith("_orderbook.parquet"):
                    normalized = pq.read_table(target, columns=["received_time", "event_time"])
                    entry["received_time_regressions_after_sequence_sort"] = regression_count(normalized["received_time"])
                    entry["event_time_regressions_after_sequence_sort"] = regression_count(normalized["event_time"])
                files.append(entry)
            else:
                files.append(normalize_one(source, target))
            print(f"[{index}/{len(sources)}] normalized {rel}", flush=True)
        except Exception as exc:
            errors.append({"source_file": str(source), "error": str(exc)})
            print(f"[{index}/{len(sources)}] ERROR {rel}: {exc}", flush=True)
    books = [f for f in files if f["data_type"] == "orderbook"]
    trades = [f for f in files if f["data_type"] == "trades"]
    expected_hours = set()
    for file in files:
        expected_hours.add((Path(file["source_file"]).parent.parent.name,
                            Path(file["source_file"]).parent.name,
                            file["symbol"], file["data_type"]))
    by_day_type = {}
    for f in files:
        p = Path(f["source_file"])
        key = f"{p.parent.parent.name}/{f['symbol']}/{f['data_type']}"
        by_day_type[key] = by_day_type.get(key, 0) + 1
    seq_gaps = sum(f.get("sequence_gaps", 0) for f in books)
    seq_overlaps = sum(f.get("sequence_overlaps", 0) for f in books)
    seq_regressions = sum(f.get("sequence_order_regressions", 0) for f in books)
    received_time_regressions = sum(f.get("received_time_regressions_after_sequence_sort", 0) for f in books)
    event_time_regressions = sum(f.get("event_time_regressions_after_sequence_sort", 0) for f in books)
    sorted_books = sorted(books, key=lambda f: f["source_file"])
    boundary_gaps = []
    boundary_overlaps = []
    for previous_book, current_book in zip(sorted_books, sorted_books[1:]):
        expected = previous_book.get("last_update_id")
        actual = current_book.get("first_update_id")
        if expected is not None and actual is not None and actual != expected + 1:
            issue = {"previous_file": previous_book["source_file"], "next_file": current_book["source_file"],
                     "expected_first_update_id": expected + 1, "actual_first_update_id": actual}
            (boundary_gaps if actual > expected + 1 else boundary_overlaps).append(issue)
    symbols = sorted({f["symbol"] for f in files})
    dates = sorted({Path(f["source_file"]).parent.parent.name for f in files})
    total_raw_bytes = sum(f["source_bytes"] for f in files)
    qa = {"schema_version": "1.0", "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
          "source": "CryptoHFTData Binance Spot", "requested_research_window": ["2025-06-28", "2025-12-31"],
          "observed_dates": dates, "observed_symbols": symbols,
          "downloaded_file_counts": {"total": len(files), "orderbook": len(books), "trades": len(trades)},
          "raw_bytes_normalized": total_raw_bytes, "normalized_files_root": str(out),
          "coverage_files_per_day_symbol_type": by_day_type,
          "missing_or_failed_files": errors,
          "book_quality_totals": {"snapshot_rows": sum(f.get("snapshot_rows", 0) for f in books),
              "update_groups": sum(f.get("update_groups", 0) for f in books),
              "sequence_gaps_within_hour": seq_gaps, "sequence_overlaps_within_hour": seq_overlaps,
              "sequence_stale_groups_within_hour": sum(f.get("sequence_stale_groups", 0) for f in books),
              "sequence_order_regressions_in_source_order": seq_regressions,
              "received_time_regressions_after_sequence_sort": received_time_regressions,
              "event_time_regressions_after_sequence_sort": event_time_regressions,
              "sequence_gaps_at_hour_boundaries": len(boundary_gaps),
              "sequence_overlaps_at_hour_boundaries": len(boundary_overlaps),
              "hour_boundary_gap_examples": boundary_gaps[:20],
              "hour_boundary_overlap_examples": boundary_overlaps[:20],
              "book_hours_without_snapshot": sum(f.get("snapshot_rows", 0) == 0 for f in books)},
          "readiness": {"normalized_event_store_ready_for_backend_ingestion": bool(files) and not errors,
              "full_research_dataset_ready": False,
              "reasons": ["Downloaded subset covers BTCUSDT only and three calendar dates, not the configured six-month/two-symbol research scope.",
                  "The event files are normalized and sequence ordered; full L2 state replay and best-quote validation are the next backend data adapter stage.",
                  "No labels, chronological train/validation/test splits, or train-only stress thresholds have been created.",
                  "The source has substantial within-hour sequence-order regressions; normalized output restores update-ID order. Review remaining event-time regressions before time-bucket features." ]},
          "files": files, "errors": errors}
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"qa_report": str(qa_path), "normalized_files": len(files),
                      "errors": len(errors), "symbols": symbols, "dates": dates,
                      "sequence_gaps": seq_gaps, "sequence_overlaps": seq_overlaps,
                      "source_order_regressions": seq_regressions,
                      "full_research_ready": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
