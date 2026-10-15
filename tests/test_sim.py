"""Track and physics tests. Pure functions, no I/O."""

from __future__ import annotations

import json
import math

from harness import case, run

from sim import (
    BASE_SPEED_MPS,
    BUCKET_K,
    OFF_TRACK_SPIN_TICKS,
    SPIN_PENALTY_S,
    TRACK_PRESETS,
    Race,
    build_track,
    heuristic_steer,
    track_catalogue,
)

DT = 1.0 / 60.0


@case
def track_is_closed_smooth_and_drivable() -> None:
    track = build_track()
    assert len(track) == 400
    assert 500.0 < track.length < 2000.0, track.length
    radii = [1.0 / abs(k) for k in track.curvature if abs(k) > 1e-6]
    assert min(radii) > 16.0, f"tightest corner {min(radii):.1f} m is not drivable at speed"
    assert max(abs(k) for k in track.curvature) < 0.5
    for value in list(track.xs) + list(track.ys) + list(track.headings) + list(track.curvature):
        assert math.isfinite(value)
    # A closed loop: the last point must be adjacent to the first, not far away.
    gap = math.dist((track.xs[0], track.ys[0]), (track.xs[-1], track.ys[-1]))
    step = track.length / len(track)
    assert gap < 3.0 * step, f"track is not closed (gap {gap:.1f} m vs step {step:.1f} m)"


@case
def lateral_offset_signs_are_right() -> None:
    track = build_track()
    index = 40
    x, y = track.point(index)
    heading = track.heading_at(index)
    # Offset the probe to the car's right (clockwise from the heading).
    right_x = x - math.sin(heading) * 4.0
    right_y = y + math.cos(heading) * 4.0
    _, lateral = track.project(right_x, right_y, index)
    assert lateral > 0, lateral
    left_x = x + math.sin(heading) * 4.0
    left_y = y - math.cos(heading) * 4.0
    _, lateral_left = track.project(left_x, left_y, index)
    assert lateral_left < 0, lateral_left


@case
def segments_ahead_describes_the_road() -> None:
    track = build_track()
    found = None
    for index in range(len(track)):
        segments = track.segments_ahead(index)
        if segments and segments[0]["direction"] != "straight":
            found = segments
            break
    assert found is not None, "the track must contain at least one bend"
    first = found[0]
    assert first["direction"] in {"left", "right"}
    assert first["distance_m"] == 0.0, "the bend under the car starts at zero distance"
    assert first["radius_m"] > 16.0
    assert first["length_m"] > 0.0
    # Distances must be non-decreasing down the list.
    distances = [segment["distance_m"] for segment in found]
    assert distances == sorted(distances), distances
    assert len(found) >= 2, "a circuit needs more than one stretch of road"


@case
def a_car_that_steers_with_the_track_completes_a_lap() -> None:
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=200.0)
    car = race.cars["solo"]
    steps = 0
    while not car.finished and race.time < 200.0:
        # The reference driver: reads the bends ahead, corrects its own errors.
        car.steer_label = heuristic_steer(track.segments_ahead(car.index, 60.0), car.lateral, math.degrees(car.heading_error))
        race.step(DT)
        steps += 1
    assert car.finished, f"reference driver did not finish in {race.time:.1f} s ({steps} steps)"
    assert car.lap == 1
    assert car.spins == 0, car.spins
    assert car.best_lateral_abs < 9.0, f"reference driver left the {track.half_width} m half-width: {car.best_lateral_abs:.2f} m"


@case
def the_grid_does_not_hand_out_a_free_lap() -> None:
    """Regression: cars were gridded behind the line and 'finished' after metres."""
    track = build_track()
    race = Race(track, [("a", "#fff"), ("b", "#000")], laps=1, time_cap_s=60.0)
    for key, car in race.cars.items():
        assert car.lap == 0, key
        assert car.since_lap_m == 0.0, key
    for _ in range(120):  # two seconds of driving
        for car in race.cars.values():
            car.steer_label = heuristic_steer(
                track.segments_ahead(car.index, 60.0), car.lateral, math.degrees(car.heading_error)
            )
        race.step(DT)
    for key, car in race.cars.items():
        assert car.lap == 0, f"{key} was credited a lap after only {car.distance_m:.1f} m"
        assert not car.finished, key
        assert car.distance_m < 0.5 * track.length, car.distance_m


@case
def a_lap_costs_a_circuit_of_travel_and_time() -> None:
    """A lap must be a whole circuit, not a nudge over the start line."""
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=200.0)
    car = race.cars["solo"]
    while not car.finished and race.time < 200.0:
        car.steer_label = heuristic_steer(
            track.segments_ahead(car.index, 60.0), car.lateral, math.degrees(car.heading_error)
        )
        race.step(DT)
    assert car.finished, f"reference driver did not finish in {race.time:.1f} s"
    assert car.finish_time is not None
    clean = track.length / BASE_SPEED_MPS
    assert car.distance_m >= 0.95 * track.length, car.distance_m
    assert 0.85 * clean <= car.finish_time <= 1.6 * clean, (car.finish_time, clean)


@case
def every_map_is_drivable_by_the_reference_driver() -> None:
    """Every preset has to be a circuit a competent driver can hold. A corner tighter
    than the sharpest steering option is undrivable by every model including a perfect
    one, which is how the first version of this project measured nothing."""
    sharpest = 1.0 / max(abs(k) for k in BUCKET_K.values())
    for name in TRACK_PRESETS:
        track = build_track(name)
        radii = [1.0 / abs(k) for k in track.curvature if abs(k) > 1e-6]
        assert min(radii) > sharpest, f"{name}: tightest corner {min(radii):.1f} m is undrivable"
        assert abs(track.length - TRACK_PRESETS[name]["length_m"]) < 1.0, (name, track.length)
        race = Race(track, [("ref", "#fff")], laps=1, time_cap_s=400.0)
        car = race.cars["ref"]
        while not car.finished and race.time < 400.0:
            car.steer_label = heuristic_steer(
                track.segments_ahead(car.index, 60.0), car.lateral, math.degrees(car.heading_error)
            )
            race.step(DT)
        assert car.finished, f"{name}: the reference driver never finished a lap"
        assert car.spins == 0, f"{name}: the reference driver spun {car.spins} times"
        assert car.best_lateral_abs < track.half_width, (name, car.best_lateral_abs)


@case
def driving_backwards_does_not_earn_distance() -> None:
    """Regression: the travel delta was a bare modulo, so a small backward step read
    as almost a whole lap forward. That is what put 1020 m on the board at lap 0."""
    track = build_track("circuit")
    race = Race(track, [("solo", "#fff")], laps=2, time_cap_s=60.0)
    car = race.cars["solo"]
    car.heading = track.heading_at(car.index) + math.pi  # face the wrong way
    for _ in range(120):
        car.steer_label = "straight"
        race.step(DT)
    assert car.distance_m < 0.2 * track.length, f"backwards driving earned {car.distance_m:.1f} m"
    assert car.lap == 0


@case
def the_map_catalogue_lists_every_map() -> None:
    entries = track_catalogue()
    assert {entry["name"] for entry in entries} == set(TRACK_PRESETS)
    for entry in entries:
        assert entry["label"], entry
        assert entry["length_m"] > 400.0, entry
        assert entry["half_width_m"] > 0.0, entry


@case
def leaving_the_track_slows_the_car_and_then_spins_it() -> None:
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=60.0)
    car = race.cars["solo"]
    # Drive straight off the outside of the loop by refusing to steer at all.
    for _ in range(1500):
        car.steer_label = "straight"
        race.step(DT)
        if car.spins:
            break
    assert car.spins >= 1, "a car that never steers must eventually leave the track"
    assert car.penalty_s > 0.0 or car.off_ticks == 0
    # After a spin the car is placed back on the centre line.
    assert abs(car.lateral) < 1.0, car.lateral


@case
def off_track_speed_is_lower_than_clean_speed() -> None:
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=60.0)
    car = race.cars["solo"]
    car.steer_label = "straight"
    for _ in range(200):
        race.step(DT)
        if car.off_ticks > 0 and car.penalty_s == 0.0:
            assert car.speed < BASE_SPEED_MPS
            break
    else:
        raise AssertionError("never went off track")


@case
def a_frozen_car_does_not_move() -> None:
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=60.0)
    car = race.cars["solo"]
    car.frozen = True
    x, y, heading = car.x, car.y, car.heading
    for _ in range(60):
        race.step(DT)
    assert (car.x, car.y, car.heading) == (x, y, heading)
    assert car.lap == 0


@case
def state_for_is_json_serialisable_and_complete() -> None:
    track = build_track()
    race = Race(track, [("a", "#fff"), ("b", "#000")], laps=3, time_cap_s=60.0)
    for _ in range(120):
        race.cars["a"].steer_label = "left"
        race.step(DT)
    state = race.state_for("a")
    json.dumps(state)  # must not raise
    assert state["task"] == "steer"
    for key in ("lateral_m", "side", "heading_error_deg", "lap", "position", "cars_total"):
        assert key in state["car"], key
    assert state["car"]["cars_total"] == 2
    assert state["track"]["width_m"] == 2 * race.track.half_width
    assert isinstance(state["goal"], str) and state["goal"]


@case
def spin_penalty_is_bounded_and_recovers() -> None:
    track = build_track()
    race = Race(track, [("solo", "#fff")], laps=1, time_cap_s=60.0)
    car = race.cars["solo"]
    car.penalty_s = SPIN_PENALTY_S
    car.steer_label = "straight"
    for _ in range(int(SPIN_PENALTY_S / DT) + 5):
        race.step(DT)
    assert car.penalty_s == 0.0
    assert car.speed > 0.0
    assert OFF_TRACK_SPIN_TICKS >= 10


if __name__ == "__main__":
    raise SystemExit(run("test_sim"))
