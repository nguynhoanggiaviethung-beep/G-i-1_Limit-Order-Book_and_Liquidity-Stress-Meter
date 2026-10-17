# Limit Order Book and Liquidity Stress Meter

## Research data source

The selected primary source for the L2 order book is **CryptoHFTData, Binance Spot**. It provides hourly Zstandard-compressed Parquet files containing L2 snapshots and updates, with matching hourly trade files. The documented Binance coverage begins on **2025-06-28**. The project's initial research window is **2025-06-28 through 2025-12-31**, for BTCUSDT and ETHUSDT.

This is not a full 2025 dataset: January through 27 June is unavailable from this source. Binance's public archive has long historical trades and candles, but not the L2 event history needed to reconstruct this project's order book. Crypto Lake lists older order-book coverage, but its free page offers trial data and directs users to a subscription for all data. Do not mix those sources silently.

## Download the hourly source data

Install dependencies in the existing project environment, then download the required files:

```powershell
& "$env:USERPROFILE\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python scripts/download_cryptohftdata.py
```

The downloader is resumable, logs each hourly request in `data_raw/cryptohftdata/download_manifest.jsonl`, and keeps raw files under `data_raw/cryptohftdata/`. Anonymous access is documented as rate-limited to 60 requests per minute; the script defaults to 55. A free API key is optional and, if you have one, should be stored in the local environment variable `CRYPTOHFTDATA_API_KEY`, never pasted into chat. Missing hourly objects are recorded and skipped.

Normalize downloaded files and write the QA report with:

```powershell
python scripts/normalize_cryptohftdata.py
```

Typed normalized event Parquet files are written under `data_processed/cryptohftdata/`; the report is `outputs/manifests/cryptohftdata_qa.json`. The converter retains the full source event/level rows, sorts book rows by sequence ID, maps timestamps to UTC, and maps `is_buyer_maker` to aggressor side. It does not build 1 Hz book-state features or hide sequence gaps.

Replay the normalized BTC book and audit sequence recovery with:

```powershell
python scripts/audit_cryptohftdata_replay.py
```

This writes valid, observed 1 Hz top-of-book samples to `data_processed/cryptohftdata/replay/BTCUSDT_book_top_1s.parquet` and its replay/gap report to `outputs/manifests/cryptohftdata_replay_qa.json`. Invalid seconds are omitted, never interpolated. This is an ingestion/replay checkpoint, not the final full-depth research feature table.

**Storage warning:** hourly L2 files are large. Downloading two symbols and both data types for the full six-month period may require substantially more disk space than the current system drive has available. Check free space first. The downloader supports smaller date ranges, symbols, and data types; e.g.:

```powershell
python scripts/download_cryptohftdata.py --start 2025-06-28 --end 2025-06-30 --symbols BTCUSDT --data-types orderbook trades
```

## Processing status and limitations

The existing `scripts/prepare_backend_pilot.py` and `data_processed/backend_pilot/` are the **legacy Tardis 2025-01-01 five-minute integration pilot**. They have not been migrated to CryptoHFTData, and their output is not evidence of CryptoHFTData processing or model-training readiness. Preserve the Tardis files as a separate validation artifact.

CryptoHFTData rows use `received_time` in nanoseconds, `event_type` (`snapshot`/`update`), Binance update IDs, and absolute level quantities. A correct adapter must:

1. Decompress each file and read the Parquet schema using PyArrow.
2. Build the book from a complete snapshot, apply updates in stored/sequence order, and invalidate it after sequence gaps until a new snapshot arrives.
3. Carry state across file boundaries only when sequence continuity is verified. Some hours contain updates without an in-hour snapshot.
4. Interpret Binance trade side using `is_buyer_maker`: true means seller-aggressor; false means buyer-aggressor.
5. Record missing hours, invalid sequences, snapshot age, and source-file hashes in a QA manifest; never fill missing intervals with invented observations.

The sample files under `data_raw/cryptohftdata_sample/` are only format probes, not a representative research sample. Do not train a model from them. Full research readiness still requires the sequence-aware normalizer, contiguous coverage audit, chronological train/validation/test splits, and labels/thresholds fit only on the train period.

## Backend contract (legacy pilot)

- Config: `config/default.yaml`
- Data dictionary and processing rules: `docs/data_dictionary.csv`, `docs/data_contract.md`
- Replay snapshot shape: `schemas/replay_snapshot.v1.example.json`
- Prediction response while no model is available: `schemas/predict_unavailable.v1.example.json`
- Legacy pilot scripts: `scripts/prepare_backend_pilot.py`, `scripts/compare_quotes.py`, `scripts/compare_quotes_eth.py`

All API response timestamps should use ISO-8601 UTC, prices and quantities should be decimal strings, and unavailable values should be null with a status. Do not return stale predictions or invent probabilities when no model artifact exists.
