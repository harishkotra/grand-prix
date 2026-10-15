"""Tiny dependency-free test harness, so the suite runs without installing pytest."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES: list = []


def case(function):
    CASES.append(function)
    return function


def run(module_name: str) -> int:
    failures = 0
    functions = [function for function in CASES if function.__module__ == module_name]
    if not functions:
        print(f"WARN no cases registered for {module_name}")
    for function in functions:
        try:
            function()
        except Exception:  # noqa: BLE001 - a test harness reports everything
            failures += 1
            print(f"FAIL {module_name}.{function.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {module_name}.{function.__name__}")
    return failures
