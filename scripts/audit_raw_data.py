
from pathlib import Path
import gzip
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_raw"

FILES = [
    "binance_incremental_book_L2_2025-01-01_BTCUSDT.csv.gz",
    "binance_incremental_book_L2_2025-01-01_ETHUSDT.csv.gz",
    "binance_trades_2025-01-01_BTCUSDT.csv.gz",
    "binance_trades_2025-01-01_ETHUSDT.csv.gz",
]

CHUNK_SIZE = 200_000
DAY_START_US = 1735689600000000
DAY_END_US = 1735776000000000  # exclusive

BOOK_COLUMNS = {
    "exchange", "symbol", "timestamp", "local_timestamp",
    "is_snapshot", "side", "price", "amount"
}
TRADE_COLUMNS = {
    "exchange", "symbol", "timestamp", "local_timestamp",
    "id", "side", "price", "amount"
}


def audit_file(path: Path):
    is_book = "incremental_book_L2" in path.name
    required = BOOK_COLUMNS if is_book else TRADE_COLUMNS

    print("\n" + "=" * 76)
    print(f"FILE: {path.name}")

    if not path.exists():
        print("ERROR: FILE NOT FOUND")
        return

    total_rows = 0
    invalid_timestamp = 0
    outside_day = 0
    invalid_local_timestamp = 0
    invalid_price = 0
    invalid_amount = 0
    nonpositive_price = 0
    negative_amount = 0
    zero_amount = 0
    invalid_side = 0
    snapshot_rows = 0
    snapshot_transitions = 0
    rows_before_first_snapshot = 0
    timestamp_regressions = 0
    local_timestamp_regressions = 0

    min_ts = None
    max_ts = None
    min_local_ts = None
    max_local_ts = None
    previous_ts = None
    previous_local_ts = None
    previous_snapshot = False
    seen_snapshot = False
    side_counts = {}

    try:
        reader = pd.read_csv(
            path,
            compression="gzip",
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

        first_chunk = True

        for chunk in reader:
            if first_chunk:
                missing = required - set(chunk.columns)
                if missing:
                    print(f"ERROR: MISSING COLUMNS: {sorted(missing)}")
                    print(f"ACTUAL COLUMNS: {list(chunk.columns)}")
                    return
                first_chunk = False

            n = len(chunk)
            total_rows += n

            ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
            local_ts = pd.to_numeric(
                chunk["local_timestamp"], errors="coerce"
            )
            price = pd.to_numeric(chunk["price"], errors="coerce")
            amount = pd.to_numeric(chunk["amount"], errors="coerce")

            invalid_timestamp += int(ts.isna().sum())
            invalid_local_timestamp += int(local_ts.isna().sum())
            invalid_price += int(price.isna().sum())
            invalid_amount += int(amount.isna().sum())

            valid_ts = ts.dropna()
            valid_local = local_ts.dropna()

            if not valid_ts.empty:
                chunk_min = int(valid_ts.min())
                chunk_max = int(valid_ts.max())
                min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
                max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)
                outside_day += int(
                    ((ts < DAY_START_US) | (ts >= DAY_END_US)).sum()
                )

            if not valid_local.empty:
                chunk_min = int(valid_local.min())
                chunk_max = int(valid_local.max())
                min_local_ts = (
                    chunk_min if min_local_ts is None
                    else min(min_local_ts, chunk_min)
                )
                max_local_ts = (
                    chunk_max if max_local_ts is None
                    else max(max_local_ts, chunk_max)
                )

            nonpositive_price += int((price <= 0).sum())
            negative_amount += int((amount < 0).sum())
            zero_amount += int((amount == 0).sum())

            sides = chunk["side"].astype("string").str.lower()
            for value, count in sides.value_counts(dropna=False).items():
                key = "<NA>" if pd.isna(value) else str(value)
                side_counts[key] = side_counts.get(key, 0) + int(count)

            allowed_sides = {"bid", "ask"} if is_book else {"buy", "sell"}
            invalid_side += int((~sides.isin(allowed_sides)).sum())

            if is_book:
                snapshots = (
                    chunk["is_snapshot"]
                    .astype("string")
                    .str.lower()
                    .isin(["true", "1"])
                )
                snapshot_rows += int(snapshots.sum())

                # Detect snapshot runs and updates before first snapshot.
                for snap in snapshots.tolist():
                    if snap and not seen_snapshot:
                        seen_snapshot = True
                        snapshot_transitions += 1
                    elif snap and not previous_snapshot:
                        snapshot_transitions += 1
                    elif not snap and not seen_snapshot:
                        rows_before_first_snapshot += 1
                    previous_snapshot = snap

            # Preserve original row order; count timestamp regressions.
            for value in ts.tolist():
                if pd.isna(value):
                    continue
                value = int(value)
                if previous_ts is not None and value < previous_ts:
                    timestamp_regressions += 1
                previous_ts = value

            for value in local_ts.tolist():
                if pd.isna(value):
                    continue
                value = int(value)
                if (
                    previous_local_ts is not None
                    and value < previous_local_ts
                ):
                    local_timestamp_regressions += 1
                previous_local_ts = value

        print(f"Rows: {total_rows:,}")
        print(f"Timestamp UTC start: {pd.to_datetime(min_ts, unit='us', utc=True) if min_ts is not None else None}")
        print(f"Timestamp UTC end:   {pd.to_datetime(max_ts, unit='us', utc=True) if max_ts is not None else None}")
        print(f"Local timestamp start: {pd.to_datetime(min_local_ts, unit='us', utc=True) if min_local_ts is not None else None}")
        print(f"Local timestamp end:   {pd.to_datetime(max_local_ts, unit='us', utc=True) if max_local_ts is not None else None}")
        print(f"Invalid exchange timestamps: {invalid_timestamp:,}")
        print(f"Invalid local timestamps: {invalid_local_timestamp:,}")
        print(f"Rows outside target UTC day: {outside_day:,}")
        print(f"Invalid/missing prices: {invalid_price:,}")
        print(f"Invalid/missing amounts: {invalid_amount:,}")
        print(f"Nonpositive prices: {nonpositive_price:,}")
        print(f"Negative amounts: {negative_amount:,}")
        print(f"Zero amounts: {zero_amount:,}")
        print(f"Invalid side rows: {invalid_side:,}")
        print(f"Side counts: {side_counts}")
        print(f"Timestamp regressions in file order: {timestamp_regressions:,}")
        print(f"Local timestamp regressions in file order: {local_timestamp_regressions:,}")

        if is_book:
            print(f"Snapshot rows: {snapshot_rows:,}")
            print(f"Snapshot runs/transitions (approx.): {snapshot_transitions:,}")
            print(f"Rows before first snapshot: {rows_before_first_snapshot:,}")

    except Exception as exc:
        print(f"ERROR WHILE READING FILE: {type(exc).__name__}: {exc}")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {DATA_DIR}")

    for filename in FILES:
        audit_file(DATA_DIR / filename)


if __name__ == "__main__":
    main()
