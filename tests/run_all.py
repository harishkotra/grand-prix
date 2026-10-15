"""Run the whole offline suite: python tests/run_all.py

No model, no network, no downloads. Anything that needs a real engine is probed
separately by tools/probe_engines.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = ("test_engines", "test_sim", "test_race")


def main() -> int:
    failures = 0
    for name in MODULES:
        module = importlib.import_module(name)
        failures += module.run(name)
    print()
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
