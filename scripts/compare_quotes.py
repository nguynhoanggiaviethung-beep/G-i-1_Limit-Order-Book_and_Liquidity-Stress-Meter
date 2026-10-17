"""Compatibility command for the canonical BTCUSDT pilot output."""
import json
from prepare_backend_pilot import compare_quotes


if __name__ == "__main__":
    print(json.dumps(compare_quotes("BTCUSDT"), ensure_ascii=False, indent=2))
