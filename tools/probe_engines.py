"""Live-invoke every configured engine with a real question and report the truth.

A model can be downloaded, listed, and still not serve the endpoint. This asks
each one the actual race question against a real state, prints the distribution
it returned, and says plainly which ones answered and which ones did not. It is
the check to run before believing a panel is alive.

    python3 tools/probe_engines.py
    python3 tools/probe_engines.py --engines julia-1-144m,laya-421m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines import EngineError, build_engine  # noqa: E402
from race import INSTRUCTIONS, OPTIONS, QUESTION_ID  # noqa: E402
from sim import Race, build_track  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STATE = {
    "task": "steer",
    "track": {
        "width_m": 18.0,
        "bends_ahead": [
            {"direction": "left", "radius_m": 24.0, "distance_m": 0.0, "length_m": 45.0},
            {"direction": "straight", "radius_m": None, "distance_m": 45.0, "length_m": 120.0},
        ],
    },
    "car": {
        "lateral_m": 3.4,
        "side": "right of centre",
        "heading_error_deg": 6.0,
        "speed_mps": 26.0,
        "lap": 1,
        "laps_total": 3,
        "off_track_ticks": 0,
        "spins": 0,
        "position": 3,
        "cars_total": 5,
        "gap_to_leader_m": 42.0,
    },
    "goal": "Hold within 5 m of the centre line at 26 m/s and finish 3 laps first.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--engines", default="", help="comma-separated keys; default is every configured racer")
    parser.add_argument("--repeat", type=int, default=3, help="how many times to ask each engine")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    specs = config.get("racers", [])
    if args.engines:
        wanted = {key.strip() for key in args.engines.split(",") if key.strip()}
        specs = [spec for spec in specs if spec["key"] in wanted]

    # Confirm the state we are about to send is the same shape the race sends.
    race = Race(build_track(), [("probe", "#fff")], laps=1)
    assert set(race.state_for("probe")) == set(STATE), "probe state drifted from the race state"

    answered = 0
    for spec in specs:
        engine = build_engine(spec)
        print(f"\n=== {engine.label}  ({engine.kind}, {engine.note})")
        ok, detail = engine.probe()
        print(f"    probe: {'ok' if ok else 'FAILED'} - {detail}")
        if not ok:
            continue
        for attempt in range(1, args.repeat + 1):
            try:
                decision = engine.decide(STATE, QUESTION_ID, INSTRUCTIONS, OPTIONS, "sample", f"probe:{engine.key}", attempt)
            except EngineError as error:
                print(f"    [{attempt}] ERROR {error}")
                break
            distribution = "  ".join(f"{label}={probability:.3f}" for label, probability in decision.probabilities.items())
            print(
                f"    [{attempt}] {decision.latency_ms:7.1f} ms  chose={decision.label:<10} "
                f"reported={decision.reported_choice}  conf={decision.confidence}  {distribution}"
            )
            if decision.warnings:
                print(f"          warnings: {decision.warnings}")
            if decision.output_tokens not in (0, None):
                print(f"          NOTE: {decision.output_tokens} output tokens, this is not a pure decision read")
        else:
            answered += 1

    print(f"\n{answered}/{len(specs)} engines answered. Nothing is reported as working that did not answer.")
    return 0 if answered == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
