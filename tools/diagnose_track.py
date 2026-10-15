"""Measure every generated circuit instead of assuming it is drivable.

Prints the length, the corner-radius distribution, and the result of driving a lap
with a perfect driver (one that steers for the bend it is about to enter). If a
perfect driver cannot hold a track, no decision model can either, and the track is
wrong rather than the model.

    python3 tools/diagnose_track.py              # every map
    python3 tools/diagnose_track.py hairpin      # one map, in detail
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import BUCKET_K, TRACK_PRESETS, Race, build_track, heuristic_steer  # noqa: E402

DT = 1.0 / 60.0


def reference_lap(track) -> tuple[bool, int, int, float, float]:
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=400.0)
    car = race.cars["solo"]
    while not car.finished and race.time < 400.0:
        car.steer_label = heuristic_steer(
            track.segments_ahead(car.index, 60.0), car.lateral, math.degrees(car.heading_error)
        )
        race.step(DT)
    return car.finished, car.lap, car.spins, race.time, car.best_lateral_abs


def detail(name: str) -> None:
    track = build_track(name)
    radii = sorted(1.0 / abs(k) for k in track.curvature if abs(k) > 1e-6)
    print(f"map              {name} ({TRACK_PRESETS[name]['label']})")
    print(f"samples          {len(track)}")
    print(f"length           {track.length:.1f} m")
    print(f"half width       {track.half_width:.1f} m")
    print(f"corner samples   {len(radii)} of {len(track)}")
    if radii:
        print(f"tightest radius  {radii[0]:.1f} m")
        print(f"p10 radius       {radii[len(radii) // 10]:.1f} m")
        print(f"median radius    {radii[len(radii) // 2]:.1f} m")
        print(f"loosest radius   {radii[-1]:.1f} m")
    print(f"max |curvature|  {max(abs(k) for k in track.curvature):.4f} 1/m")
    print(f"bucket radii     {', '.join(f'{label}={1 / abs(k):.0f}m' for label, k in BUCKET_K.items() if k)}")
    finished, lap, spins, time_s, worst = reference_lap(track)
    print()
    print(f"reference driver finished={finished} lap={lap} spins={spins} "
          f"time={time_s:.1f}s worst_lateral={worst:.2f} m (half width {track.half_width:.0f} m)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("map", nargs="?", default="", help="map name; omit for all maps")
    args = parser.parse_args()

    if args.map:
        if args.map not in TRACK_PRESETS:
            print(f"unknown map {args.map!r}; known: {', '.join(TRACK_PRESETS)}", file=sys.stderr)
            return 2
        detail(args.map)
        return 0

    sharpest = 1.0 / max(abs(k) for k in BUCKET_K.values())
    print(f"sharpest steering radius {sharpest:.1f} m: any corner tighter than this is")
    print("undrivable by every model, including a perfect one.\n")
    header = f"{'map':12s}{'length':>8s}{'half_w':>8s}{'tightest':>10s}{'ref lap':>9s}{'spins':>7s}{'worst lat':>11s}{'ok':>5s}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for name in TRACK_PRESETS:
        track = build_track(name)
        radii = [1.0 / abs(k) for k in track.curvature if abs(k) > 1e-6]
        finished, lap, spins, time_s, worst = reference_lap(track)
        ok = finished and spins == 0 and min(radii) > sharpest
        all_ok = all_ok and ok
        print(f"{name:12s}{track.length:8.0f}{track.half_width:8.1f}{min(radii):10.1f}"
              f"{time_s:9.1f}{spins:7d}{worst:11.2f}{'yes' if ok else 'NO':>5s}")
    print()
    print("every map drivable by the reference driver" if all_ok else "SOME MAPS ARE NOT DRIVABLE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
