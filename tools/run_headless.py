"""Run a real race headless and print the summary. No browser, no pacing.

This is how a race is measured before it is filmed: every decision is recorded to
runs/<race_id>.jsonl, and the summary table is printed here.

    python3 tools/run_headless.py --laps 1
    python3 tools/run_headless.py --laps 1 --engines julia-1-144m,kev-0.8b
    python3 tools/run_headless.py --laps 1 --replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from race import MOCK_SPECS, RaceSession  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--out", default=str(ROOT / "runs"))
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--engines", default="")
    parser.add_argument("--track", default="", help="map name, e.g. circuit, oval, hairpin")
    parser.add_argument("--replay", action="store_true", help="use the synthetic racers")
    parser.add_argument("--policy", default="", choices=["", "sample", "argmax"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    specs = list(MOCK_SPECS) if args.replay else list(config.get("racers", []))
    if args.engines:
        wanted = {key.strip() for key in args.engines.split(",") if key.strip()}
        specs = [spec for spec in specs if spec["key"] in wanted]
    if not specs:
        print("no racers selected", file=sys.stderr)
        return 2

    race_config = dict(config.get("race", {}))
    race_config["laps"] = args.laps
    race_config["realtime"] = False
    if args.track:
        race_config["track"] = args.track
    if args.policy:
        race_config["policy"] = args.policy
    if args.seed:
        race_config["seed"] = args.seed

    session = RaceSession(specs, race_config, Path(args.out))
    asyncio.run(session.run())

    print(f"\nrace {session.race_id}  track={session.track_name}  laps={session.race.laps}  policy={session.policy}  seed={session.seed}")
    print(f"sim time {session.race.time:.1f} s")
    for note in session.notes:
        print(f"NOTE {note}")

    header = f"{'model':<16}{'status':<13}{'decisions':>10}{'dec/s':>8}{'p50 ms':>9}{'p95 ms':>9}{'entropy':>9}{'laps':>6}{'spins':>7}{'finish s':>10}"
    print()
    print(header)
    print("-" * len(header))
    snapshot = session.snapshot()
    cars = {car["key"]: car for car in snapshot["race"]["cars"]}
    for racer in snapshot["racers"]:
        car = cars[racer["key"]]
        finish = f"{car['finish_time']:.1f}" if car["finish_time"] is not None else "-"
        print(
            f"{racer['label']:<16}{racer['status']:<13}{racer['decisions']:>10}{racer['decisions_per_sec']:>8.1f}"
            f"{racer['latency_p50_ms']:>9.1f}{racer['latency_p95_ms']:>9.1f}"
            f"{(racer['mean_entropy'] if racer['mean_entropy'] is not None else float('nan')):>9.3f}"
            f"{car['lap']:>6}{car['spins']:>7}{finish:>10}"
        )
    print()
    print(f"telemetry  {session.telemetry_path}")
    print(f"summary    {session.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
