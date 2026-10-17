
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data_raw")

# Chỉ kiểm tra dữ liệu Binance cần cho nghiên cứu
files = [
    p for p in DATA_DIR.rglob("*.csv.gz")
    if "binance_" in p.name.lower()
    and any(s in p.name.upper() for s in ["BTCUSDT", "ETHUSDT"])
    and any(
        kind in p.name.lower()
        for kind in ["incremental_book_l2", "trades"]
    )
]

if not files:
    raise FileNotFoundError(
        "Không tìm thấy file Binance BTCUSDT/ETHUSDT phù hợp."
    )

for file in sorted(files):
    print("\n" + "=" * 70)
    print("FILE:", file.name)

    total_rows = 0
    snapshot_rows = 0
    timestamp_min = None
    timestamp_max = None
    unique_timestamps = set()
    sides = set()
    invalid_timestamps = 0

    for chunk in pd.read_csv(
        file,
        compression="gzip",
        chunksize=200_000
    ):
        total_rows += len(chunk)

        if "timestamp" not in chunk.columns:
            raise ValueError(f"Thiếu cột timestamp: {file.name}")

        # Tardis timestamps được lưu theo microsecond
        raw_ts = pd.to_numeric(
            chunk["timestamp"], errors="coerce"
        )
        invalid_timestamps += raw_ts.isna().sum()
        ts = pd.to_datetime(
            raw_ts, unit="us", utc=True, errors="coerce"
        ).dropna()

        if not ts.empty:
            chunk_min = ts.min()
            chunk_max = ts.max()

            timestamp_min = (
                chunk_min if timestamp_min is None
                else min(timestamp_min, chunk_min)
            )
            timestamp_max = (
                chunk_max if timestamp_max is None
                else max(timestamp_max, chunk_max)
            )

        unique_timestamps.update(raw_ts.dropna().unique())

        if "is_snapshot" in chunk.columns:
            snapshot_rows += (
                chunk["is_snapshot"].astype(str).str.lower()
                .isin(["true", "1", "yes"]).sum()
            )

        if "side" in chunk.columns:
            sides.update(chunk["side"].dropna().astype(str).unique())

    print("Rows:", f"{total_rows:,}")
    print("Invalid timestamps:", f"{invalid_timestamps:,}")
    print("Start UTC:", timestamp_min)
    print("End UTC:", timestamp_max)
    print("Distinct timestamps:", f"{len(unique_timestamps):,}")
    print("Snapshot rows:", f"{snapshot_rows:,}")
    print("Side values:", sorted(sides))

    if timestamp_min and timestamp_max:
        duration = timestamp_max - timestamp_min
        print("Observed duration:", duration)
