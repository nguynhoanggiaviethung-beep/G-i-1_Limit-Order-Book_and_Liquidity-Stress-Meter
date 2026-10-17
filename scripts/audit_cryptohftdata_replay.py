"""Replay normalized Binance Spot L2 events and audit valid 1 Hz top-of-book.

This is a data-quality/backend-ingestion audit, not a model feature pipeline.
It applies snapshots and deltas by Binance update IDs and never fills gaps.
"""
from __future__ import annotations

import argparse
import heapq
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED = ROOT / "data_processed" / "cryptohftdata" / "binance_spot"
DEFAULT_OUT = ROOT / "data_processed" / "cryptohftdata" / "replay"
DEFAULT_QA = ROOT / "outputs" / "manifests" / "cryptohftdata_replay_qa.json"


def iso_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def apply_level(book: dict, heap: list, side: str, price: str, quantity: str) -> None:
    px, qty = float(price), float(quantity)
    if px <= 0 or qty < 0:
        raise ValueError(f"invalid L2 price/quantity: {price} / {quantity}")
    if qty == 0:
        book[side].pop(px, None)
        return
    book[side][px] = qty
    heapq.heappush(heap[side], -px if side == "bid" else px)


def best_price(book: dict, heap: list, side: str):
    queue = heap[side]
    while queue:
        top = -queue[0] if side == "bid" else queue[0]
        if top in book[side]:
            return top
        heapq.heappop(queue)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", default=str(DEFAULT_NORMALIZED))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--qa", default=str(DEFAULT_QA))
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"),
        help="Replay one symbol. Required when normalized files contain multiple symbols.")
    args = parser.parse_args()
    normalized, out, qa_path = Path(args.normalized), Path(args.output), Path(args.qa)
    paths = sorted(normalized.glob("2025-*/[0-2][0-9]/*_orderbook.parquet"))
    available_symbols = sorted({path.name.split("_")[0] for path in paths})
    if args.symbol:
        paths = [path for path in paths if path.name.startswith(f"{args.symbol}_")]
    elif len(available_symbols) > 1:
        raise SystemExit("Multiple symbols found. Run once per symbol with --symbol BTCUSDT or --symbol ETHUSDT.")
    if not paths:
        suffix = f" for {args.symbol}" if args.symbol else ""
        raise SystemExit(f"No normalized order book files found{suffix} under {normalized}")

    book = {"bid": {}, "ask": {}}
    heap = {"bid": [], "ask": []}
    valid = False
    last_update_id = None
    pending_gap = None
    invalid_intervals = []
    samples = {}
    counters = {"files": len(paths), "events": 0, "updates_applied": 0,
        "updates_skipped_while_book_invalid": 0, "stale_updates_skipped": 0,
        "sequence_gaps": 0, "snapshots": 0, "snapshot_recoveries": 0,
        "incomplete_snapshots": 0, "invalid_crossed_or_empty_states": 0,
        "valid_seconds": 0}

    def save_sample(ts: int):
        bid, ask = best_price(book, heap, "bid"), best_price(book, heap, "ask")
        if bid is None or ask is None or bid >= ask:
            counters["invalid_crossed_or_empty_states"] += 1
            return
        second = ts // 1_000_000_000
        samples[second] = {"timestamp_ns": ts, "timestamp_utc": iso_ns(ts),
            "symbol": symbol, "data_status": "valid", "best_bid": format(bid, ".8f"),
            "best_bid_quantity": format(book["bid"][bid], ".8f"),
            "best_ask": format(ask, ".8f"),
            "best_ask_quantity": format(book["ask"][ask], ".8f"),
            "mid_price": format((bid + ask) / 2, ".8f"),
            "spread": format(ask - bid, ".8f"),
            "last_update_id": last_update_id}

    symbol = None
    for path_no, path in enumerate(paths, start=1):
        table = pq.read_table(path, columns=["received_time", "event_type", "first_update_id",
            "final_update_id", "last_update_id", "side", "price", "quantity"])
        columns = {name: table[name].to_pylist() for name in table.column_names}
        symbol = path.name.split("_")[0]
        current_key = None
        levels = []
        event_ts = None
        event_start = event_end = None
        event_kind = None

        def finish_event():
            nonlocal valid, last_update_id, pending_gap
            if current_key is None:
                return
            counters["events"] += 1
            if event_kind == "snapshot":
                new_book = {"bid": {}, "ask": {}}
                new_heap = {"bid": [], "ask": []}
                for side, price, quantity in levels:
                    apply_level(new_book, new_heap, side, price, quantity)
                if not new_book["bid"] or not new_book["ask"]:
                    counters["incomplete_snapshots"] += 1
                    valid = False
                    return
                book.clear(); book.update(new_book)
                heap.clear(); heap.update(new_heap)
                last_update_id = event_end
                valid = True
                counters["snapshots"] += 1
                if pending_gap is not None:
                    invalid_intervals.append({"start_utc": iso_ns(pending_gap),
                        "recovered_utc": iso_ns(event_ts), "recovery_snapshot_update_id": event_end})
                    counters["snapshot_recoveries"] += 1
                    pending_gap = None
                save_sample(event_ts)
                return

            start, end = event_start, event_end
            if not valid or last_update_id is None:
                counters["updates_skipped_while_book_invalid"] += 1
                return
            if end <= last_update_id:
                counters["stale_updates_skipped"] += 1
                return
            if not (start <= last_update_id + 1 <= end):
                counters["sequence_gaps"] += 1
                valid = False
                pending_gap = event_ts
                return
            for side, price, quantity in levels:
                apply_level(book, heap, side, price, quantity)
            last_update_id = end
            counters["updates_applied"] += 1
            save_sample(event_ts)

        for i in range(table.num_rows):
            kind = columns["event_type"][i]
            received = columns["received_time"][i]
            first_id = columns["first_update_id"][i]
            final_id = columns["final_update_id"][i]
            snapshot_id = columns["last_update_id"][i]
            end_id = snapshot_id if kind == "snapshot" and snapshot_id is not None else final_id
            key = (kind, received, first_id if kind == "update" else None, end_id)
            if key != current_key:
                finish_event()
                current_key = key
                levels = []
                event_ts = received
                event_start = first_id
                event_end = end_id
                event_kind = kind
            levels.append((columns["side"][i], columns["price"][i], columns["quantity"][i]))
        finish_event()
        print(f"[{path_no}/{len(paths)}] replayed {path.relative_to(normalized)}", flush=True)

    if pending_gap is not None:
        invalid_intervals.append({"start_utc": iso_ns(pending_gap), "recovered_utc": None,
            "recovery_snapshot_update_id": None, "status": "unresolved_at_end_of_downloaded_data"})
    ordered = [samples[k] for k in sorted(samples)]
    counters["valid_seconds"] = len(ordered)
    out.mkdir(parents=True, exist_ok=True)
    output = out / f"{symbol}_book_top_1s.parquet"
    pq.write_table(pa.Table.from_pylist(ordered), output, compression="zstd")
    timestamps = [row["timestamp_ns"] for row in ordered]
    report = {"schema_version": "1.0", "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "CryptoHFTData Binance Spot L2", "symbol": symbol,
        "files_replayed": len(paths), "normalized_event_files_root": str(normalized),
        "valid_top_of_book_output": str(output), "valid_1s_rows": len(ordered),
        "valid_time_start_utc": ordered[0]["timestamp_utc"] if ordered else None,
        "valid_time_end_utc": ordered[-1]["timestamp_utc"] if ordered else None,
        "valid_seconds_with_missing_intervals_not_filled": len(ordered),
        "output_timestamp_regressions_after_second_sort": sum(a > b for a,b in zip(timestamps,timestamps[1:])),
        "observed_dates_utc": sorted({path.parent.parent.name for path in paths}),
        "counters": counters, "invalid_intervals": invalid_intervals,
        "assessment": {"replayable_until_sequence_gap": True,
            "full_downloaded_span_reconstructable": not any(i.get("status") == "unresolved_at_end_of_downloaded_data" for i in invalid_intervals),
            "backend_can_continue_event_ingestion_and_replay_code": True,
            "backend_can_claim_complete_research_dataset": False,
            "reasons": [f"This replay covers {len({path.parent.parent.name for path in paths})} observed UTC date(s) for {symbol}; compare them with the configured study window.",
                "Any unresolved sequence gap remains invalid through the end of the downloaded data; no observations were fabricated.",
                "This output contains valid 1-second top-of-book rows only; it does not yet include the study's full depth features, labels, or leakage-safe splits."]}}
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(qa_path), "output": str(output),
        "valid_seconds": counters["valid_seconds"], "sequence_gaps": counters["sequence_gaps"],
        "snapshots": counters["snapshots"], "snapshot_recoveries": counters["snapshot_recoveries"],
        "unresolved_at_end": bool(pending_gap), "invalid_intervals": invalid_intervals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
