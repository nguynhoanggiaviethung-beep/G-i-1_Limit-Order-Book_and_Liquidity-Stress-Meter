"""Write the running FastAPI app's OpenAPI document to frontend/openapi."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402


def main() -> None:
    target = ROOT / "frontend" / "openapi" / "schema.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote OpenAPI schema to {target}")


if __name__ == "__main__":
    main()
