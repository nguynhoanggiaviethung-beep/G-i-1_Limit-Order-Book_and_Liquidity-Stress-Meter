"""Normalize the available Binance/Tardis pilot data for backend integration.

This is an integration pilot, not a model-training dataset. Raw files are read
only. Outputs are written under data_processed/backend_pilot and outputs/.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
VERIFY = ROOT / "data_verify"
OUT = ROOT / "data_processed" / "backend_pilot"
MANIFESTS = ROOT / "outputs" / "manifests"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
DATE = "2025-01-01"
DAY_START_US = 1735689600000000  # 2025-01-01T00:00:00Z
DURATION_US = 300 * 1_000_000
DEPTH = 20
OFI_WINDOW_US = 5 * 1_000_000


def iso_us(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def dec(value: str | Decimal) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"non-finite decimal: {value}")
    return result


def book_columns() -> list[str]:
    cols = ["symbol", "local_timestamp_us", "local_timestamp_utc", "second_utc", "data_status",
            "best_bid", "best_ask", "mid_price", "microprice", "spread", "bid_levels", "ask_levels",
            "obi_l1", "obi_l5", "obi_l10", "obi_l20", "ofi_5s", "ofi_5s_status",
            "bid_depth_slope", "ask_depth_slope", "realized_volatility_5s",
            "realized_volatility_5s_status",
            "trade_count_5s", "trade_buy_quantity_5s", "trade_sell_quantity_5s",
            "trade_unknown_count_5s", "trade_unknown_quantity_5s", "trade_imbalance_5s",
            "trade_imbalance_5s_status"]
    for side in ("bid", "ask"):
        for i in range(1, DEPTH + 1):
            cols.extend((f"{side}_price_{i}", f"{side}_quantity_{i}", f"{side}_cumulative_quantity_{i}"))
    return cols


def depth_slope(levels: list[tuple[Decimal, Decimal]], mid: Decimal) -> str | None:
    levels = levels[:DEPTH]
    if len(levels) < 5:
        return None
    xs, ys = [], []
    cumulative = Decimal(0)
    for price, qty in levels:
        cumulative += qty
        xs.append(float(abs(price - mid) / mid * Decimal(10_000)))
        ys.append(float(cumulative))
    xbar, ybar = mean(xs), mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        return None
    return format(Decimal(str(sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom)), "f")


def ofi_event(prev: tuple[Decimal, Decimal, Decimal, Decimal] | None,
              cur: tuple[Decimal, Decimal, Decimal, Decimal] | None) -> Decimal | None:
    if prev is None or cur is None:
        return None
    pb, qb, pa, qa = prev
    cb, cq, ca, cqa = cur
    # Cont, Kukanov & Stoikov event OFI: when a best price is unchanged,
    # the contribution is the change in displayed quantity, not the full
    # current/previous quantity alone.
    zero = Decimal(0)
    bid = (cq if cb >= pb else zero) - (qb if cb <= pb else zero)
    ask = (qa if ca >= pa else zero) - (cqa if ca <= pa else zero)
    return bid + ask


def reconstruct(symbol: str) -> dict:
    source = RAW / f"binance_incremental_book_L2_{DATE}_{symbol}.csv.gz"
    if not source.exists():
        raise FileNotFoundError(source)
    bids: dict[Decimal, Decimal] = {}
    asks: dict[Decimal, Decimal] = {}
    seen_in_snapshot_epoch = {"bid": set(), "ask": set()}
    snapshot_zero_levels = {"bid": set(), "ask": set()}
    snapshot_bounds = {"bid": [None, None], "ask": [None, None]}
    state = {"rows_read": 0, "rows_skipped_before_snapshot": 0, "rows_in_window": 0,
             "timestamp_regressions": 0, "snapshot_resets": 0, "snapshot_zero_rows": 0,
             "incremental_removals": 0, "removals_absent": 0,
             "absent_removals_by_side": {"bid": 0, "ask": 0},
             "empty_states": 0, "crossed_or_locked_states": 0, "valid_groups": 0,
             "invalid_groups": 0, "rows_bad": 0}
    anomalies = []
    last_snapshot_ts = None
    samples: dict[int, dict] = {}
    events: deque[tuple[int, Decimal | None, bool]] = deque()
    t0 = None
    cutoff = None
    slot_start = None
    group_ts = None
    group_reset = False
    last_snapshot = False
    seen_snapshot = False
    prev_local = None
    prev_top = None
    group_count = 0

    def current_top():
        if not bids or not asks:
            return None
        bid, ask = max(bids), min(asks)
        if bid >= ask:
            return None
        return bid, bids[bid], ask, asks[ask]

    def finish_group():
        nonlocal prev_top, group_count
        if group_ts is None or t0 is None or group_ts < t0 or group_ts >= cutoff:
            return
        group_count += 1
        top = current_top()
        if top is None:
            state["empty_states" if not bids or not asks else "crossed_or_locked_states"] += 1
            state["invalid_groups"] += 1
            events.append((group_ts, None, True))
            prev_top = None
        else:
            state["valid_groups"] += 1
            value = None if group_reset else ofi_event(prev_top, top)
            events.append((group_ts, value, group_reset or value is None))
            prev_top = top

        bucket = group_ts // 1_000_000
        if slot_start is None or not slot_start <= bucket < slot_start + 300:
            return
        while events and events[0][0] <= group_ts - OFI_WINDOW_US:
            events.popleft()
        window_valid = bool(events) and not any(bad for _, _, bad in events)
        window_values = [v for _, v, _ in events if v is not None]
        ofi = sum(window_values, Decimal(0)) if window_valid else None
        bid_levels = sorted(bids.items(), reverse=True)
        ask_levels = sorted(asks.items())
        mid = (top[0] + top[2]) / Decimal(2)
        total_depth = {}
        for n in (1, 5, 10, 20):
            bq = sum((q for _, q in bid_levels[:n]), Decimal(0))
            aq = sum((q for _, q in ask_levels[:n]), Decimal(0))
            total_depth[n] = (bq - aq) / (bq + aq) if bq + aq else None
        row = {
            "symbol": symbol, "local_timestamp_us": group_ts,
            "local_timestamp_utc": iso_us(group_ts),
            "second_utc": datetime.fromtimestamp(bucket, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "data_status": "valid", "best_bid": format(top[0], "f"),
            "best_ask": format(top[2], "f"), "mid_price": format(mid, "f"),
            "microprice": format((top[2] * top[1] + top[0] * top[3]) / (top[1] + top[3]), "f"),
            "spread": format(top[2] - top[0], "f"),
            "bid_levels": len(bids), "ask_levels": len(asks),
            "obi_l1": None if total_depth[1] is None else format(total_depth[1], "f"),
            "obi_l5": None if total_depth[5] is None else format(total_depth[5], "f"),
            "obi_l10": None if total_depth[10] is None else format(total_depth[10], "f"),
            "obi_l20": None if total_depth[20] is None else format(total_depth[20], "f"),
            "ofi_5s": None if ofi is None else format(ofi, "f"),
            "ofi_5s_status": "available" if ofi is not None else "unavailable_reset_or_invalid_window",
            "bid_depth_slope": depth_slope(bid_levels, mid),
            "ask_depth_slope": depth_slope(ask_levels, mid),
            "realized_volatility_5s": None,
            "realized_volatility_5s_status": "insufficient_contiguous_history",
            "trade_count_5s": 0,
            "trade_buy_quantity_5s": "0",
            "trade_sell_quantity_5s": "0",
            "trade_unknown_count_5s": 0,
            "trade_unknown_quantity_5s": "0",
            "trade_imbalance_5s": None,
            "trade_imbalance_5s_status": "insufficient_trade_coverage",
        }
        for side, levels in (("bid", bid_levels), ("ask", ask_levels)):
            cumulative = Decimal(0)
            for i in range(1, DEPTH + 1):
                if len(levels) >= i:
                    price, qty = levels[i - 1]
                    cumulative += qty
                    row[f"{side}_price_{i}"] = format(price, "f")
                    row[f"{side}_quantity_{i}"] = format(qty, "f")
                    row[f"{side}_cumulative_quantity_{i}"] = format(cumulative, "f")
                else:
                    row[f"{side}_price_{i}"] = None
                    row[f"{side}_quantity_{i}"] = None
                    row[f"{side}_cumulative_quantity_{i}"] = None
        # Last valid grouped state in each UTC second wins.
        samples[bucket] = row

    with gzip.open(source, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "local_timestamp", "is_snapshot", "side", "price", "amount"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source.name}: missing {sorted(required - set(reader.fieldnames or []))}")
        for row_no, row in enumerate(reader, start=2):
            state["rows_read"] += 1
            try:
                local_ts = int(row["local_timestamp"])
                snap = row["is_snapshot"].strip().lower() in ("true", "1")
                if not seen_snapshot and not snap:
                    state["rows_skipped_before_snapshot"] += 1
                    continue
                if not seen_snapshot:
                    seen_snapshot = True
                    t0 = local_ts
                    cutoff = t0 + DURATION_US
                    slot_start = t0 // 1_000_000
                if local_ts >= cutoff:
                    break
                if prev_local is not None and local_ts < prev_local:
                    state["timestamp_regressions"] += 1
                    raise ValueError(f"local timestamp regression at row {row_no}: {local_ts} < {prev_local}")
                if group_ts is not None and local_ts != group_ts:
                    finish_group()
                    group_reset = False
                group_ts = local_ts
                side = row["side"].strip().lower()
                if side not in ("bid", "ask"):
                    raise ValueError(f"invalid side {side!r} at row {row_no}")
                price, amount = dec(row["price"]), dec(row["amount"])
                if price <= 0 or amount < 0:
                    raise ValueError(f"nonpositive price or negative amount at row {row_no}")
                state["rows_in_window"] += 1
                if snap and not last_snapshot:
                    bids.clear(); asks.clear()
                    seen_in_snapshot_epoch = {"bid": set(), "ask": set()}
                    state["snapshot_resets"] += 1
                    group_reset = True
                    prev_top = None
                    last_snapshot_ts = local_ts
                book = bids if side == "bid" else asks
                if amount == 0:
                    if snap:
                        state["snapshot_zero_rows"] += 1
                        snapshot_zero_levels[side].add(price)
                    else:
                        state["incremental_removals"] += 1
                        if price in book:
                            del book[price]
                        else:
                            state["removals_absent"] += 1
                            state["absent_removals_by_side"][side] += 1
                            low, high = snapshot_bounds[side]
                            if price in seen_in_snapshot_epoch[side]:
                                classification = "repeated_remove_after_prior_presence"
                            elif price in snapshot_zero_levels[side]:
                                classification = "price_had_zero_amount_in_snapshot"
                            elif ((side == "bid" and low is not None and price < low)
                                  or (side == "ask" and high is not None and price > high)):
                                classification = "outside_initial_snapshot_price_bound"
                            else:
                                classification = "unseen_price_within_initial_snapshot_bounds"
                            anomalies.append({"symbol": symbol, "row_number": row_no,
                                "local_timestamp_us": local_ts, "local_timestamp_utc": iso_us(local_ts),
                                "side": side, "price": format(price, "f"), "snapshot_row": snap,
                                "snapshot_reset_count": state["snapshot_resets"],
                                "last_snapshot_timestamp_us": last_snapshot_ts,
                                "microseconds_since_snapshot": local_ts - last_snapshot_ts if last_snapshot_ts is not None else None,
                                "initial_snapshot_min_price": format(low, "f") if low is not None else None,
                                "initial_snapshot_max_price": format(high, "f") if high is not None else None,
                                "classification": classification,
                                "group_timestamp_us": local_ts,
                                "assessment": "recorded_and_ignored_for_state; classification is diagnostic and not proof of source correctness"})
                else:
                    book[price] = amount
                    seen_in_snapshot_epoch[side].add(price)
                    if snap:
                        low, high = snapshot_bounds[side]
                        snapshot_bounds[side] = (price if low is None else min(low, price),
                                                 price if high is None else max(high, price))
                last_snapshot = snap
                prev_local = local_ts
            except (ValueError, InvalidOperation, TypeError) as exc:
                state["rows_bad"] += 1
                raise ValueError(f"{source.name} row {row_no}: {exc}") from exc
    finish_group()

    # Fill only sampled seconds that had at least one valid book state; never interpolate.
    output = OUT / f"pilot_{symbol}_book_states_1s.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = [samples[s] for s in sorted(samples)]
    pd.DataFrame(ordered, columns=book_columns()).to_csv(output, index=False)
    state.update({"symbol": symbol, "source_file": source.name,
        "t0_local_timestamp_us": t0, "t0_utc": iso_us(t0) if t0 is not None else None,
        "window_end_exclusive_utc": iso_us(cutoff) if cutoff is not None else None,
        "utc_second_slot_start": slot_start,
        "initial_snapshot_price_bounds": {side: {"min": format(v[0], "f") if v[0] is not None else None,
            "max": format(v[1], "f") if v[1] is not None else None} for side, v in snapshot_bounds.items()},
        "snapshot_zero_price_rows_by_side": {side: len(values) for side, values in snapshot_zero_levels.items()},
        "sample_rows": len(ordered), "sampled_utc_seconds": [
            datetime.fromtimestamp(s, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            for s in sorted(samples)],
        "sample_gaps": max(0, 300 - len(ordered)), "output_file": str(output.relative_to(ROOT)).replace("\\", "/")})
    return state, anomalies


def normalize_trades(symbol: str, window_start: int, cutoff: int) -> dict:
    source = RAW / f"binance_trades_{DATE}_{symbol}.csv.gz"
    output = OUT / f"pilot_{symbol}_trades_normalized.csv"
    columns = ["symbol", "exchange_timestamp_us", "exchange_timestamp_utc", "local_timestamp_us",
               "local_timestamp_utc", "trade_id", "source_side", "price", "quantity"]
    rows = 0
    invalid = 0
    with gzip.open(source, "rt", newline="", encoding="utf-8") as handle, output.open("w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        if not {"timestamp", "local_timestamp", "id", "side", "price", "amount"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{source.name}: required trade columns missing")
        for row in reader:
            local_ts = int(row["local_timestamp"])
            if local_ts < window_start or local_ts >= cutoff:
                continue
            price, qty = dec(row["price"]), dec(row["amount"])
            side = row["side"].strip().lower()
            if price <= 0 or qty <= 0 or side not in ("buy", "sell", "unknown"):
                invalid += 1
                continue
            exchange_ts = int(row["timestamp"])
            writer.writerow({"symbol": symbol, "exchange_timestamp_us": exchange_ts,
                "exchange_timestamp_utc": iso_us(exchange_ts), "local_timestamp_us": local_ts,
                "local_timestamp_utc": iso_us(local_ts), "trade_id": row["id"],
                "source_side": side, "price": format(price, "f"), "quantity": format(qty, "f")})
            rows += 1
    return {"source_file": source.name, "output_file": str(output.relative_to(ROOT)).replace("\\", "/"),
            "coverage_start_us": window_start, "coverage_end_exclusive_us": cutoff,
            "rows_in_window": rows, "invalid_rows_skipped": invalid,
            "side_semantics": "Tardis side is liquidity taker/aggressor side; unknown values are retained and excluded from signed imbalance"}


def enrich_features(symbol: str, trade_coverage_start_us: int) -> dict:
    """Add strictly point-in-time 5-second trade-flow and realized-vol features."""
    book_path = OUT / f"pilot_{symbol}_book_states_1s.csv"
    trades_path = OUT / f"pilot_{symbol}_trades_normalized.csv"
    book = pd.read_csv(book_path, dtype=str, keep_default_na=False)
    trades = pd.read_csv(trades_path, dtype=str, keep_default_na=False)
    trades = trades.sort_values("local_timestamp_us", kind="stable")
    trade_rows = [(int(r.local_timestamp_us), dec(r.quantity), r.source_side)
                  for r in trades.itertuples(index=False)]
    active = deque()
    pointer = 0
    buy_qty, sell_qty, unknown_qty = Decimal(0), Decimal(0), Decimal(0)
    buy_count = sell_count = unknown_count = 0
    previous_second = None
    previous_mid = None
    returns = deque()
    enriched = []
    for row in book.to_dict("records"):
        ts = int(row["local_timestamp_us"])
        while pointer < len(trade_rows) and trade_rows[pointer][0] <= ts:
            trade_ts, qty, side = trade_rows[pointer]
            active.append((trade_ts, qty, side))
            if side == "buy": buy_qty += qty; buy_count += 1
            elif side == "sell": sell_qty += qty; sell_count += 1
            else: unknown_qty += qty; unknown_count += 1
            pointer += 1
        lower = ts - OFI_WINDOW_US
        while active and active[0][0] <= lower:
            trade_ts, qty, side = active.popleft()
            if side == "buy": buy_qty -= qty; buy_count -= 1
            elif side == "sell": sell_qty -= qty; sell_count -= 1
            else: unknown_qty -= qty; unknown_count -= 1
        complete_trade_window = ts - OFI_WINDOW_US >= trade_coverage_start_us
        if complete_trade_window:
            row["trade_count_5s"] = str(buy_count + sell_count + unknown_count)
            row["trade_buy_quantity_5s"] = format(buy_qty, "f")
            row["trade_sell_quantity_5s"] = format(sell_qty, "f")
            row["trade_unknown_count_5s"] = str(unknown_count)
            row["trade_unknown_quantity_5s"] = format(unknown_qty, "f")
            known_qty = buy_qty + sell_qty
            row["trade_imbalance_5s"] = format((buy_qty - sell_qty) / known_qty, "f") if known_qty > 0 else ""
            row["trade_imbalance_5s_status"] = "available" if known_qty > 0 else "no_known_trades_in_window"
        else:
            row["trade_count_5s"] = ""
            row["trade_buy_quantity_5s"] = ""
            row["trade_sell_quantity_5s"] = ""
            row["trade_unknown_count_5s"] = ""
            row["trade_unknown_quantity_5s"] = ""
            row["trade_imbalance_5s"] = ""
            row["trade_imbalance_5s_status"] = "insufficient_trade_coverage"

        second = int(pd.Timestamp(row["second_utc"]).timestamp())
        mid = float(Decimal(row["mid_price"]))
        if previous_second is not None and second == previous_second + 1:
            returns.append(math.log(mid / previous_mid))
            while len(returns) > 5:
                returns.popleft()
        else:
            returns.clear()
        row["realized_volatility_5s"] = format(math.sqrt(sum(r * r for r in returns)), ".15g") if len(returns) == 5 else ""
        row["realized_volatility_5s_status"] = "available" if len(returns) == 5 else "insufficient_contiguous_history"
        previous_second, previous_mid = second, mid
        enriched.append(row)
    pd.DataFrame(enriched, columns=book_columns()).to_csv(book_path, index=False)
    return {"trade_imbalance_window": "(t-5s,t]", "trade_side": "Tardis aggressor side",
            "trade_unknown_rows_preserved": int((trades.source_side == "unknown").sum()) if len(trades) else 0,
            "realized_volatility": "sqrt(sum of squared one-second log mid-price returns over five contiguous prior intervals); null until five intervals are available",
            "features_output_file": str(book_path.relative_to(ROOT)).replace("\\", "/")}


def compare_quotes(symbol: str) -> dict:
    book_path = OUT / f"pilot_{symbol}_book_states_1s.csv"
    quote_path = VERIFY / f"binance_quotes_{DATE}_{symbol}.csv.gz"
    comparison_path = OUT / f"pilot_{symbol}_quotes_comparison.csv"
    book = pd.read_csv(book_path, dtype=str)
    book["sample_ts_us"] = pd.to_numeric(book["local_timestamp_us"], errors="raise")
    quotes = pd.read_csv(quote_path, compression="gzip", dtype=str)
    needed = {"local_timestamp", "bid_price", "ask_price", "symbol", "exchange"}
    if not needed.issubset(quotes.columns):
        raise ValueError(f"{quote_path.name}: missing {sorted(needed - set(quotes.columns))}")
    quotes = quotes[(quotes.exchange == "binance") & (quotes.symbol == symbol)].copy()
    quotes["quote_ts_us"] = pd.to_numeric(quotes.local_timestamp, errors="raise")
    quotes = quotes.sort_values("quote_ts_us").drop_duplicates("quote_ts_us", keep="last")
    quotes = quotes.rename(columns={"bid_price": "quote_bid", "ask_price": "quote_ask"})
    merged = pd.merge_asof(book.sort_values("sample_ts_us"), quotes.sort_values("quote_ts_us"),
                           left_on="sample_ts_us", right_on="quote_ts_us", direction="backward", allow_exact_matches=True)
    matched = merged.quote_ts_us.notna()
    merged["bid_match"] = [dec(a) == dec(b) if pd.notna(a) and pd.notna(b) else False for a,b in zip(merged.best_bid, merged.quote_bid)]
    merged["ask_match"] = [dec(a) == dec(b) if pd.notna(a) and pd.notna(b) else False for a,b in zip(merged.best_ask, merged.quote_ask)]
    merged["both_match"] = merged.bid_match & merged.ask_match
    merged.to_csv(comparison_path, index=False)
    n = int(matched.sum())
    return {"reference_file": quote_path.name, "output_file": str(comparison_path.relative_to(ROOT)).replace("\\", "/"),
        "samples": len(merged), "without_preceding_quote": int((~matched).sum()), "compared": n,
        "bid_matches": int((merged.bid_match & matched).sum()), "ask_matches": int((merged.ask_match & matched).sum()),
        "both_match": int((merged.both_match & matched).sum()),
        "bid_match_rate": float((merged.bid_match & matched).sum()/n) if n else None,
        "ask_match_rate": float((merged.ask_match & matched).sum()/n) if n else None,
        "both_match_rate": float((merged.both_match & matched).sum()/n) if n else None}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    qa = {"dataset_version": "pilot-1.0.0", "schema_version": "1.0",
          "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
          "data_mode": "historical_replay", "source": "Tardis Binance Spot historical files",
          "date_utc": DATE, "symbols": {},
          "limitations": ["One pilot date only; not representative for training or final evaluation.",
            "Pilot covers at most 300 UTC-second slots from the first snapshot; absent slots are not interpolated.",
            "No train/validation/test split or target labels are produced from this pilot.",
            "No full-year 2025 data availability is claimed.",
            "Depth quote validation is not available: references validate best bid/ask only.",
            "Raw trade side is preserved; aggressor interpretation must be confirmed before signed flow features."]}
    absent = []
    source_log = {"schema_version": "1.0", "source": "Tardis.dev historical data",
                  "exchange": "Binance Spot", "date_utc": DATE, "symbols": {},
                  "processing_script": "scripts/prepare_backend_pilot.py"}
    for symbol in SYMBOLS:
        book_qa, anomalies = reconstruct(symbol)
        absent.extend(anomalies)
        coverage_start = max(DAY_START_US, book_qa["t0_local_timestamp_us"] - OFI_WINDOW_US)
        trade_qa = normalize_trades(symbol, coverage_start,
                                    book_qa["t0_local_timestamp_us"] + DURATION_US)
        feature_qa = enrich_features(symbol, coverage_start)
        quote_qa = compare_quotes(symbol)
        qa["symbols"][symbol] = {"book": book_qa, "trades": trade_qa, "features": feature_qa,
                                 "quote_validation": quote_qa}
        source_files = [RAW / book_qa["source_file"], RAW / trade_qa["source_file"],
                        VERIFY / quote_qa["reference_file"]]
        source_log["symbols"][symbol] = [{"file": str(f.relative_to(ROOT)).replace("\\", "/"),
            "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in source_files]
    pd.DataFrame(absent, columns=["symbol", "row_number", "local_timestamp_us", "local_timestamp_utc", "side", "price", "snapshot_row", "snapshot_reset_count", "last_snapshot_timestamp_us", "microseconds_since_snapshot", "initial_snapshot_min_price", "initial_snapshot_max_price", "classification", "group_timestamp_us", "assessment"]).to_csv(OUT / "absent_level_removals.csv", index=False)
    removal_summary = {}
    for symbol in SYMBOLS:
        rows = [r for r in absent if r["symbol"] == symbol]
        per_side = {}
        for side in ("bid", "ask"):
            selected = [r for r in rows if r["side"] == side]
            ages = [r["microseconds_since_snapshot"] for r in selected if r["microseconds_since_snapshot"] is not None]
            per_side[side] = {"count": len(selected),
                "classifications": {c: sum(r.get("classification") == c for r in selected) for c in sorted({r.get("classification") for r in selected})},
                "within_1s_of_snapshot": sum(age <= 1_000_000 for age in ages),
                "within_1_to_5s_of_snapshot": sum(1_000_000 < age <= 5_000_000 for age in ages),
                "after_5s_of_snapshot": sum(age > 5_000_000 for age in ages),
                "first_utc": min((r["local_timestamp_utc"] for r in selected), default=None),
                "last_utc": max((r["local_timestamp_utc"] for r in selected), default=None)}
        removal_summary[symbol] = per_side
    qa["absent_level_removals_analysis"] = {"by_symbol_and_side": removal_summary,
        "interpretation": "All removals for absent levels are retained and ignored for state application per L2 semantics. Outside-bound deletes are consistent with an update beyond the captured snapshot price range; repeated removes follow a prior observed level. Unseen in-bound deletes remain unresolved and need source/sequence review. These are diagnostics, not proof of source correctness."}
    qa["absent_level_removals_file"] = str((OUT / "absent_level_removals.csv").relative_to(ROOT)).replace("\\", "/")
    qa["readiness"] = {"normalized_pilot_for_backend_integration": True,
        "features_available": ["mid_price", "microprice", "spread", "top_20_depth", "OBI_L1_L5_L10_L20", "event_grouped_OFI_5s", "depth_slope", "trade_imbalance_5s", "realized_volatility_5s"],
        "labels_available": False, "model_training_ready": False,
        "blockers": ["Only one pilot date and limited 5-minute window", "No leakage-safe chronological train/validation/test period", "Stress threshold and direction delta must be fitted using a real train period", "Full removal anomaly review and quote QA results must be reviewed before expanding collection"]}
    write_json(MANIFESTS / "pilot_data_qa.json", qa)
    write_json(MANIFESTS / "source_log.json", source_log)
    write_json(MANIFESTS / "split_manifest_pilot.json", {"schema_version": "1.0", "status": "not_created_pilot_insufficient",
        "reason": "A 5-minute integration pilot is not a valid training/evaluation period; do not derive labels/splits or report model metrics from it.",
        "symbols": list(SYMBOLS), "data_date_utc": DATE, "future_policy": "Use common chronological UTC boundaries across symbols and purge at least the maximum label horizon at each split boundary."})
    print(json.dumps({"qa_report": str((MANIFESTS / "pilot_data_qa.json").relative_to(ROOT)),
                      "symbols": {s: {"samples": qa["symbols"][s]["book"]["sample_rows"],
                          "quote_validation": qa["symbols"][s]["quote_validation"],
                          "absent_removals": qa["symbols"][s]["book"]["removals_absent"]} for s in SYMBOLS},
                      "training_ready": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
