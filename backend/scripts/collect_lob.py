"""Run the LOB collector without the API. Use either this or the API process, not both."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_settings
from app.ingestion.collector import BinanceLOBCollector


async def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    collector = BinanceLOBCollector(settings)
    await collector.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
