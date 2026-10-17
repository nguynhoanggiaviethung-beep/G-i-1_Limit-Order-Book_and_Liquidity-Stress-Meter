# Limit Order Book and Liquidity Stress Meter

## Data sharing and reproducibility

GitHub stores the code, configuration, schemas, QA reports, and a checksum manifest. It intentionally does not store raw or processed market-data files. Each developer downloads the source files into local, ignored folders, then runs the same normalization and replay scripts. This keeps Git clones small while allowing the team to check that everyone is using byte-for-byte identical raw files.

The committed file `outputs/manifests/cryptohftdata_qa.json` is the checksum reference for the **exact CryptoHFTData dataset currently held by the project owner**. It records the relative source path, file size, and SHA-256 of every expected raw file. Run `scripts/verify_cryptohftdata_snapshot.py` after downloading; do not treat data as matched unless that command succeeds.

## Reproduce the current owner dataset

### Dataset scope

The reproducible baseline currently contains:

- Source: CryptoHFTData, Binance Spot L2 order book and trades
- Symbol: BTCUSDT
- UTC dates: 2025-06-28, 2025-06-29, and 2025-06-30
- Expected files: 132 total (66 orderbook and 66 trades), about 322.3 MB raw
- Expected replay output: 203,459 valid observed 1-second top-of-book rows

This is the dataset currently available to Backend and Frontend for integration. It is **not** the full research window in `config/default.yaml`: it has no ETHUSDT files and covers only three days. The six-month, two-symbol scope remains a future data-acquisition target. Missing hours in the source must stay missing; do not interpolate them.

### Windows PowerShell steps

For an existing clone, update the code from the repository root:

```powershell
git pull origin main
```

For a new clone, run:

```powershell
git clone https://github.com/nguynhoanggiaviethung-beep/G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter.git
Set-Location G-i-1_Limit-Order-Book_and_Liquidity-Stress-Meter
```

Then, from the repository root, create a local environment if needed. Python 3.11 is the project target.

```powershell
py -3.11 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt

# Download the exact current baseline, not the future six-month target.
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-06-30 `
  --symbols BTCUSDT `
  --data-types orderbook trades

# Verify every raw file against the committed SHA-256 reference.
& $PY scripts/verify_cryptohftdata_snapshot.py

# Rebuild normalized Parquet files and the QA report.
& $PY scripts/normalize_cryptohftdata.py

# Rebuild the sequence-aware 1-second BTC replay and replay QA report.
& $PY scripts/audit_cryptohftdata_replay.py
```

If the virtual environment already exists, skip the `py -3.11 -m venv .venv` line. If PowerShell cannot find the `py` launcher, install/select Python 3.11 or use the Python 3.11 executable directly to create `.venv`.

### Expected local files

The downloader writes raw source files under:

```text
data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet
data_raw/cryptohftdata/binance_spot/YYYY-MM-DD/HH/BTCUSDT_trades.parquet
data_raw/cryptohftdata/download_manifest.jsonl
```

Normalization writes under `data_processed/cryptohftdata/` and updates `outputs/manifests/cryptohftdata_qa.json`. Replay writes:

```text
data_processed/cryptohftdata/replay/BTCUSDT_book_top_1s.parquet
outputs/manifests/cryptohftdata_replay_qa.json
```

These paths are ignored by Git. Keep them on each developer's machine; do not force-add them to commits. The baseline QA manifests in `outputs/manifests/` are versioned so the team can compare counts, sequence gaps, and coverage. The raw SHA-256 verifier checks exact source-file identity before normalization.

### What a successful reproduction means

The checksum verifier must report all 132 expected files present with matching byte counts and SHA-256 hashes. After normalization, the QA report should list the three dates, BTCUSDT, 132 files (66 orderbook and 66 trades), and no normalization errors. After replay, compare the QA report and row count with the committed baseline. A matching raw checksum proves the same source files were downloaded; it does not make the partial three-day sample a complete research dataset.

The downloader requests all 24 hours per day. The baseline manifest includes only files actually available from the provider; unavailable hours are logged as `missing` in each developer's local `download_manifest.jsonl`. The raw-file count and checksums define the expected baseline. The downloader is resumable and rate-limited to at most 55 requests per minute. A free API key is optional; if the provider issued one, store it in the local environment variable `CRYPTOHFTDATA_API_KEY` and never commit it or paste it into chat.

If checksum verification fails, first inspect the verifier output and local download manifest. Do not overwrite the committed reference manifest just to make a failing download pass. The provider may have changed or removed an object; in that case the team must agree on the replacement data and update the checksum reference deliberately.

## Extending to the full research window

The intended research scope is 2025-06-28 through 2025-12-31, BTCUSDT and ETHUSDT. It is not yet present in the committed QA manifest. Do not run a full-window download and call it the shared baseline until the resulting coverage has been audited and a new reviewed checksum manifest is committed. For a later approved acquisition, the downloader accepts an inclusive date range, symbols, and data types, for example:

```powershell
& $PY scripts/download_cryptohftdata.py `
  --start 2025-06-28 `
  --end 2025-12-31 `
  --symbols BTCUSDT ETHUSDT `
  --data-types orderbook trades
```

Check free disk space first. The six-month request is much larger and may contain unavailable hours. Record actual coverage per symbol and data type; do not infer continuous coverage from the requested dates.

## Separate Tardis validation pilot

The Tardis pilot for 2025-01-01 is a separate validation artifact. Do not mix it into the CryptoHFTData research baseline. `outputs/manifests/source_log.json` records hashes for the owner's pilot files, but the CryptoHFTData downloader does not fetch those Tardis files. A teammate who needs to reproduce the pilot must obtain those exact files separately and verify their hashes against that source log.

## Processing details and limitations

CryptoHFTData rows use `received_time` in nanoseconds, `event_type` (`snapshot`/`update`), Binance update IDs, and absolute level quantities. The adapter must:

1. Read each downloaded Parquet file with PyArrow.
2. Build the book from a complete snapshot, apply updates in sequence order, and invalidate it after sequence gaps until a new snapshot arrives.
3. Carry state across file boundaries only when sequence continuity is verified. Some hours contain updates without an in-hour snapshot.
4. Interpret Binance trade side using `is_buyer_maker`: `true` means seller-aggressor; `false` means buyer-aggressor.
5. Record missing hours, invalid sequences, snapshot age, and source-file hashes in QA; never fill missing intervals with invented observations.

The sample files under `data_raw/cryptohftdata_sample/` are format probes, not a representative research sample. Do not train a model from them. Full research readiness still requires the complete agreed coverage, sequence-aware replay, chronological train/validation/test splits, and labels/thresholds fit only on the train period.

## Backend contract

- Config: `config/default.yaml`
- Data dictionary and processing rules: `docs/data_dictionary.csv`, `docs/data_contract.md`
- Replay snapshot shape: `schemas/replay_snapshot.v1.example.json`
- Prediction response while no model is available: `schemas/predict_unavailable.v1.example.json`
- Data preparation scripts: `scripts/download_cryptohftdata.py`, `scripts/verify_cryptohftdata_snapshot.py`, `scripts/normalize_cryptohftdata.py`, `scripts/audit_cryptohftdata_replay.py`
- Legacy pilot scripts: `scripts/prepare_backend_pilot.py`, `scripts/compare_quotes.py`, `scripts/compare_quotes_eth.py`

All API response timestamps should use ISO-8601 UTC, prices and quantities should be decimal strings, and unavailable values should be null with a status. Do not return stale predictions or invent probabilities when no model artifact exists.
