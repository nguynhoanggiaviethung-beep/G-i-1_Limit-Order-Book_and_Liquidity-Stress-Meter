"""Guard commands retained from the retired candle-based research branch."""


def refuse_legacy_ohlcv(script_name: str) -> None:
    raise SystemExit(
        f"{script_name} belongs to the retired OHLCV pipeline and is disabled in this LOB-only project. "
        "Use sync-freeze-free LOB commands in README.md: collect_lob.py, validate_lob.py, "
        "build_lob_dataset.py, and run_lob_evaluation.py."
    )
