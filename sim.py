"""Track, car physics and the race tick loop. Pure Python, no I/O, no model calls.

The circuit is built so that its curvature is controlled by construction, not by
luck. A parametric loop is sampled, its raw curvature is smoothed, the smoothed
profile is rescaled so the total turning is exactly one full revolution, and the
positions are then integrated back from that profile. The result is a closed,
smooth circuit whose tightest corner radius is a function of the smoothing
distance, which means the tightest corner can be made drivable on purpose.

That matters here: the five steering options correspond to fixed turn radii
(18 m, 38 m, straight), so a track with a 4 m corner would be undrivable by every
model, including a perfect one, and the race would measure nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

# The five steering options, ordered hard left to hard right. Curvature is in 1/m,
# so a positive value turns right and the turn radius is 1/|k|.
#
# The extremes are deliberately moderate. A 144M decision model given this task
# collapses onto the two outer options and almost never uses the gentle ones, so
# when the extremes were 18 m arcs on a 39 m corner the car spiralled inside and
# spun 15 times a lap. At 28 m the wrong extreme is a wrong-way turn the car can
# recover from, and a correct one holds a corner the track is built to match.
STEER_BUCKETS: tuple[tuple[str, float, str], ...] = (
    ("hard_left", -0.036, "Steer hard left: a firm left arc, about a 28 m radius."),
    ("left", -0.018, "Steer left: a gentle left arc, about a 55 m radius."),
    ("straight", 0.0, "Hold the wheel straight: no steering input."),
    ("right", 0.018, "Steer right: a gentle right arc, about a 55 m radius."),
    ("hard_right", 0.036, "Steer hard right: a firm right arc, about a 28 m radius."),
)
BUCKET_K = {label: k for label, k, _ in STEER_BUCKETS}
# Tuned so a real decision model can complete a lap. The steering buckets have fixed
# turn radii (18 m and 38 m) that do not change with speed, so the speed only sets
# how much ground a single decision covers. At 26 m/s a model deciding 15 times a
# second commits to 1.7 m of track per decision and spins 22 times a lap; at 14 m/s
# it commits to 0.9 m and the race becomes a race.
BASE_SPEED_MPS = 14.0
OFF_TRACK_SPEED = 0.65
OFF_TRACK_SPIN_TICKS = 75
SPIN_PENALTY_S = 1.5
TRACK_HALF_WIDTH_M = 13.0
CORNER_THRESHOLD = 1.0 / 400.0  # curvature above this counts as a corner
LOOKAHEAD_M = 130.0
# The grid starts this many track points past the start line, so no car is sitting
# behind the line waiting to be credited a lap for nudging across it.
GRID_INDEX = 8
# A lap only counts once the car has travelled most of the circuit since the last
# one. Without this, a car placed near the line finishes the race on its first tick.
LAP_MIN_TRAVEL = 0.90
DEFAULT_TRACK = "circuit"

# Named maps, so a demo video can show the models on more than one circuit. Every
# preset is validated by tests/test_sim.py: the reference driver must complete a
# clean lap on each of them.
TRACK_PRESETS: dict[str, dict[str, Any]] = {
    # Corners close to a steering bucket's own radius are far easier than corners
    # between buckets: the 28 m bucket holds a 30 m corner almost exactly, whereas a
    # 45 m corner can only be approximated by alternating buckets. The default map
    # sits on the extreme bucket's radius on purpose, so the models race instead of
    # spinning. The hairpin map is the hard one and is meant to punish them.
    "circuit": {
        "label": "Circuit",
        "length_m": 700.0,
        "smoothing_m": 40.0,
        "half_width": 15.0,
        "shape": (150.0, 100.0),
        "wobble": (14.0, 10.0),
    },
    "speedway": {
        "label": "Speedway",
        "length_m": 900.0,
        "smoothing_m": 60.0,
        "half_width": 14.0,
        "shape": (200.0, 120.0),
        "wobble": (8.0, 6.0),
    },
    "switchback": {
        "label": "Switchback",
        "length_m": 750.0,
        "smoothing_m": 40.0,
        "half_width": 12.0,
        "shape": (160.0, 110.0),
        "wobble": (14.0, 10.0),
    },
    "hairpin": {
        "label": "Hairpin",
        "length_m": 700.0,
        "smoothing_m": 80.0,
        "half_width": 11.0,
        "shape": (160.0, 110.0),
        "wobble": (24.0, 18.0),
    },
    "oval": {
        "label": "Oval",
        "length_m": 700.0,
        "smoothing_m": 45.0,
        "half_width": 14.0,
        "shape": (190.0, 105.0),
        "wobble": (8.0, 6.0),
    },
}


def track_catalogue() -> list[dict[str, Any]]:
    """The maps the UI can offer, in a stable order."""
    return [
        {
            "name": name,
            "label": preset["label"],
            "length_m": preset["length_m"],
            "half_width_m": preset["half_width"],
        }
        for name, preset in TRACK_PRESETS.items()
    ]


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


@dataclass
class Track:
    xs: list[float]
    ys: list[float]
    headings: list[float]
    curvature: list[float]
    cum: list[float]
    length: float
    half_width: float = TRACK_HALF_WIDTH_M

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return min(self.xs), min(self.ys), max(self.xs), max(self.ys)

    def point(self, index: int) -> tuple[float, float]:
        index %= len(self)
        return self.xs[index], self.ys[index]

    def heading_at(self, index: int) -> float:
        return self.headings[index % len(self)]

    def s_at(self, index: int) -> float:
        return self.cum[index % len(self)]

    def radius_at(self, index: int) -> float | None:
        k = self.curvature[index % len(self)]
        if abs(k) < CORNER_THRESHOLD:
            return None
        return 1.0 / abs(k)

    def project(self, x: float, y: float, hint: int) -> tuple[int, float]:
        """Nearest centreline index near `hint`, plus signed lateral offset in metres."""
        best_index = hint
        best_distance = float("inf")
        count = len(self)
        for offset in range(-25, 26):
            index = (hint + offset) % count
            dx = x - self.xs[index]
            dy = y - self.ys[index]
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_index = index
        heading = self.headings[best_index]
        dx = x - self.xs[best_index]
        dy = y - self.ys[best_index]
        # Positive lateral means the car is to the right of the direction of travel.
        lateral = -math.sin(heading) * dx + math.cos(heading) * dy
        return best_index, lateral

    def segments_ahead(self, index: int, distance_m: float = LOOKAHEAD_M) -> list[dict[str, Any]]:
        """The road ahead as a list of bends, starting with the one under the car.

        Each entry is a contiguous stretch of one character: a straight, or a left
        or right bend of a given radius. This is what a driver reads off the road,
        so this is what the model is handed.
        """
        count = len(self)
        segments: list[dict[str, Any]] = []
        current_sign: int | None = None
        current_start = 0
        current_distance = 0.0
        step = 0
        travelled = 0.0
        while travelled < distance_m and step < count:
            i = (index + step) % count
            k = self.curvature[i]
            sign = 0 if abs(k) < CORNER_THRESHOLD else (1 if k > 0 else -1)
            if sign != current_sign:
                if current_sign is not None:
                    segments.append(self._describe_segment(current_start, i, current_distance))
                current_sign = sign
                current_start = i
                # Distance from the car to where this new stretch begins.
                current_distance = travelled
            travelled = (self.cum[i] - self.cum[index]) % self.length
            step += 1
        if current_sign is not None:
            segments.append(self._describe_segment(current_start, (index + step) % count, current_distance))
        return segments

    def _describe_segment(self, start: int, end: int, distance_m: float) -> dict[str, Any]:
        span = max(1, (end - start) % len(self))
        curvatures = [self.curvature[(start + i) % len(self)] for i in range(span)]
        direction = "straight"
        radius: float | None = None
        if any(abs(k) >= CORNER_THRESHOLD for k in curvatures):
            peak = max(curvatures, key=abs)
            radius = round(1.0 / abs(peak), 1)
            direction = "right" if peak > 0 else "left"
        return {
            "direction": direction,
            "radius_m": radius,
            "distance_m": round(max(0.0, distance_m), 1),
            "length_m": round(span * (self.length / len(self)), 1),
        }


def build_track(name: str = DEFAULT_TRACK, **overrides: Any) -> Track:
    """A closed circuit with controlled corner radii. See the module docstring.

    `name` picks a preset from TRACK_PRESETS and any of its parameters can be
    overridden by keyword, which is how the tuning tools sweep track shapes.
    """
    params: dict[str, Any] = dict(TRACK_PRESETS[name])
    params.pop("label", None)
    params.update(overrides)
    samples = int(params.get("samples", 400))
    length_m = float(params["length_m"])
    smoothing_m = float(params["smoothing_m"])
    half_width = float(params["half_width"])
    shape = tuple(params["shape"])
    wobble = tuple(params["wobble"])
    ds = length_m / samples
    # 1. A closed parametric shape, used only as the curvature *profile* to smooth.
    raw: list[tuple[float, float]] = []
    for i in range(samples):
        theta = 2.0 * math.pi * i / samples
        raw.append(
            (
                shape[0] * math.cos(theta) + wobble[0] * math.cos(3.0 * theta),
                shape[1] * math.sin(theta) + wobble[1] * math.sin(4.0 * theta),
            )
        )
    profile = _curvature_from_points(raw)
    # 2. Smooth the profile. This is what sets the tightest corner.
    window = max(3, int(round(smoothing_m / ds)) | 1)
    profile = _smooth(profile, window)
    # 3. Rescale so the total turning is exactly one revolution: guarantees closure.
    total = sum(profile) * ds
    if abs(total) > 1e-9:
        profile = [k * (2.0 * math.pi) / total for k in profile]
    # 4. Integrate heading, then position, at constant speed.
    headings = [0.0]
    for i in range(samples - 1):
        headings.append(headings[-1] + profile[i] * ds)
    xs = [0.0]
    ys = [0.0]
    for i in range(samples - 1):
        xs.append(xs[-1] + math.cos(headings[i]) * ds)
        ys.append(ys[-1] + math.sin(headings[i]) * ds)
    # 5. Distribute any residual closure error along the loop rather than at a seam.
    error_x = xs[-1] + math.cos(headings[-1]) * ds - xs[0]
    error_y = ys[-1] + math.sin(headings[-1]) * ds - ys[0]
    xs = [x - error_x * i / samples for i, x in enumerate(xs)]
    ys = [y - error_y * i / samples for i, y in enumerate(ys)]
    # 6. Recompute geometry from the final positions.
    headings = []
    for i in range(samples):
        dx = xs[(i + 1) % samples] - xs[(i - 1) % samples]
        dy = ys[(i + 1) % samples] - ys[(i - 1) % samples]
        headings.append(math.atan2(dy, dx))
    curvature: list[float] = []
    cum: list[float] = [0.0]
    for i in range(samples):
        turn = wrap_angle(headings[(i + 1) % samples] - headings[(i - 1) % samples])
        span = math.dist((xs[(i - 1) % samples], ys[(i - 1) % samples]), (xs[(i + 1) % samples], ys[(i + 1) % samples]))
        curvature.append(turn / span if span > 1e-6 else 0.0)
        cum.append(cum[-1] + math.dist((xs[i], ys[i]), (xs[(i + 1) % samples], ys[(i + 1) % samples])))
    # 7. Scale the finished geometry so the circuit is exactly `length_m` long. The
    # shape only sets the curvature *profile*, so the requested length is otherwise
    # whatever the integration happened to produce. A uniform scale keeps the shape,
    # scales the length by that factor and scales every corner radius by it too.
    measured = cum[-1]
    if measured > 1e-6:
        factor = length_m / measured
        xs = [x * factor for x in xs]
        ys = [y * factor for y in ys]
        curvature = [k / factor for k in curvature]
        cum = [c * factor for c in cum]
    length = cum[-1]
    return Track(xs=xs, ys=ys, headings=headings, curvature=curvature, cum=cum[:-1], length=length, half_width=half_width)


def _curvature_from_points(points: Sequence[tuple[float, float]]) -> list[float]:
    count = len(points)
    headings = [
        math.atan2(
            points[(i + 1) % count][1] - points[(i - 1) % count][1],
            points[(i + 1) % count][0] - points[(i - 1) % count][0],
        )
        for i in range(count)
    ]
    curvature: list[float] = []
    for i in range(count):
        turn = wrap_angle(headings[(i + 1) % count] - headings[(i - 1) % count])
        span = math.dist(points[(i - 1) % count], points[(i + 1) % count])
        curvature.append(turn / span if span > 1e-6 else 0.0)
    return curvature


def _smooth(values: Sequence[float], window: int = 5) -> list[float]:
    half = max(1, window // 2)
    count = len(values)
    kernel = [1.0] * (2 * half + 1)
    total = sum(kernel)
    return [
        sum(values[(i + offset) % count] * kernel[offset + half] for offset in range(-half, half + 1)) / total
        for i in range(count)
    ]


def heuristic_steer(
    bends_ahead: Sequence[dict[str, Any]],
    lateral_m: float,
    heading_error_deg: float,
    anticipation_m: float = 22.0,
) -> str:
    """A competent human-style driver: the reference bar, used by the mock engine.

    It steers for the bend it is about to enter rather than the one it is already
    in, and it corrects its lateral offset and its heading error. A decision model
    that beats this looks good; the track is tuned so that this driver can hold
    the line, which means the track is not the thing being measured.
    """
    chosen: dict[str, Any] | None = bends_ahead[0] if bends_ahead else None
    for segment in bends_ahead:
        if float(segment.get("distance_m", 0.0)) <= anticipation_m:
            chosen = segment
        else:
            break
    demand = 0.0
    radius = (chosen or {}).get("radius_m")
    if radius:
        demand += (1.0 / float(radius)) * (1.0 if chosen.get("direction") == "right" else -1.0)
    demand += -lateral_m * 0.006
    demand += -heading_error_deg * 0.004
    return min(BUCKET_K, key=lambda label: abs(BUCKET_K[label] - demand))


@dataclass
class Car:
    key: str
    color: str
    x: float
    y: float
    heading: float
    index: int = 0
    lateral: float = 0.0
    heading_error: float = 0.0
    speed: float = BASE_SPEED_MPS
    steer_label: str = "straight"
    lap: int = 0
    previous_s: float = 0.0
    distance_m: float = 0.0
    since_lap_m: float = 0.0  # travel since the last counted lap
    off_ticks: int = 0
    spins: int = 0
    penalty_s: float = 0.0
    finished: bool = False
    finish_time: float | None = None
    best_lateral_abs: float = 0.0
    frozen: bool = False  # set when a racer's engine is unavailable: it does not move

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "color": self.color,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading": round(self.heading, 4),
            "lateral": round(self.lateral, 2),
            "heading_error_deg": round(math.degrees(self.heading_error), 1),
            "speed": round(self.speed, 2),
            "steer": self.steer_label,
            "lap": self.lap,
            "distance_m": round(self.distance_m, 1),
            "off_track": self.off_ticks > 0,
            "off_ticks": self.off_ticks,
            "spins": self.spins,
            "penalty_s": round(self.penalty_s, 2),
            "finished": self.finished,
            "finish_time": self.finish_time,
            "frozen": self.frozen,
        }


class Race:
    """Holds every car, advances physics, and describes each car for its model."""

    def __init__(self, track: Track, racers: Sequence[tuple[str, str]], laps: int = 3, time_cap_s: float = 120.0) -> None:
        self.track = track
        self.laps = laps
        self.time_cap_s = time_cap_s
        self.time = 0.0
        # The track geometry never changes, so it is built once and reused. The UI
        # polls a snapshot at 30 Hz and must not rebuild 400 points each time.
        self.track_json: dict[str, Any] = {
            "length_m": round(track.length, 1),
            "half_width_m": track.half_width,
            "points": [[round(x, 1), round(y, 1)] for x, y in zip(track.xs, track.ys)],
        }
        self.cars: dict[str, Car] = {}
        for slot, (key, color) in enumerate(racers):
            # Stagger the grid laterally so the cars do not overlap, and place it
            # just past the start line so nobody gets a free lap for crossing it.
            lateral = (slot - (len(racers) - 1) / 2.0) * 3.0
            index = (GRID_INDEX + 12 * slot) % len(track)
            x, y = track.point(index)
            heading = track.heading_at(index)
            x += -math.sin(heading) * lateral
            y += math.cos(heading) * lateral
            self.cars[key] = Car(key=key, color=color, x=x, y=y, heading=heading, index=index, lateral=lateral)
        self.finish_order: list[str] = []

    # -- state ---------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "laps": self.laps,
            "time_cap_s": self.time_cap_s,
            "finished": list(self.finish_order),
            "cars": [car.to_json() for car in self.cars.values()],
            "track": self.track_json,
        }

    def state_for(self, key: str) -> dict[str, Any]:
        """The exact state object handed to the model. JSON-serialisable by design."""
        car = self.cars[key]
        bends = self.track.segments_ahead(car.index)
        position = sorted(self.cars.values(), key=lambda c: (c.finished, c.distance_m), reverse=True)
        place = next((i + 1 for i, c in enumerate(position) if c.key == key), len(position))
        leader = position[0]
        return {
            "task": "steer",
            "track": {
                "width_m": round(self.track.half_width * 2, 1),
                "bends_ahead": bends,
            },
            "car": {
                "lateral_m": round(car.lateral, 2),
                "side": "right of centre" if car.lateral > 0 else ("left of centre" if car.lateral < 0 else "on the centre line"),
                "heading_error_deg": round(math.degrees(car.heading_error), 1),
                "speed_mps": round(car.speed, 1),
                "lap": car.lap,
                "laps_total": self.laps,
                "off_track_ticks": car.off_ticks,
                "spins": car.spins,
                "position": place,
                "cars_total": len(self.cars),
                "gap_to_leader_m": round(max(0.0, leader.distance_m - car.distance_m), 1),
            },
            "goal": (
                f"Hold within 5 m of the centre line at {BASE_SPEED_MPS:.0f} m/s, stay on a {self.track.half_width * 2:.0f} m wide "
                f"circuit, and finish {self.laps} laps ahead of the other cars."
            ),
        }

    # -- physics -------------------------------------------------------------
    def step(self, dt: float) -> None:
        self.time += dt
        for car in self.cars.values():
            if car.frozen or car.finished:
                continue
            if car.penalty_s > 0.0:
                car.penalty_s = max(0.0, car.penalty_s - dt)
                car.speed = 0.0
                continue
            curvature = BUCKET_K.get(car.steer_label, 0.0)
            car.heading = wrap_angle(car.heading + curvature * car.speed * dt)
            car.x += math.cos(car.heading) * car.speed * dt
            car.y += math.sin(car.heading) * car.speed * dt
            index, lateral = self.track.project(car.x, car.y, car.index)
            previous_s = self.track.s_at(car.index)
            car.index = index
            car.lateral = lateral
            car.heading_error = wrap_angle(car.heading - self.track.heading_at(index))
            car.best_lateral_abs = max(car.best_lateral_abs, abs(lateral))
            on_track = abs(lateral) <= self.track.half_width
            if on_track:
                car.off_ticks = 0
                car.speed = BASE_SPEED_MPS
            else:
                car.off_ticks += 1
                car.speed = BASE_SPEED_MPS * OFF_TRACK_SPEED
                if car.off_ticks >= OFF_TRACK_SPIN_TICKS:
                    car.spins += 1
                    car.off_ticks = 0
                    car.penalty_s = SPIN_PENALTY_S
                    car.x, car.y = self.track.point(index)
                    car.heading = self.track.heading_at(index)
                    car.lateral = 0.0
                    car.speed = 0.0
            travelled = self.track.s_at(index) - previous_s
            # Unwrap: a small backward step must not read as most of a lap forward,
            # which is what a bare modulo does and what inflated the standings.
            if travelled < -0.5 * self.track.length:
                travelled += self.track.length
            elif travelled > 0.5 * self.track.length:
                travelled -= self.track.length
            travelled = max(0.0, travelled)
            car.distance_m += travelled
            car.since_lap_m += travelled
            crossed_line = previous_s > 0.75 * self.track.length and self.track.s_at(index) < 0.25 * self.track.length
            if crossed_line and car.since_lap_m >= LAP_MIN_TRAVEL * self.track.length:
                car.lap += 1
                car.since_lap_m = 0.0
                if car.lap >= self.laps and not car.finished:
                    car.finished = True
                    car.finish_time = self.time
                    self.finish_order.append(car.key)

    @property
    def over(self) -> bool:
        return bool(self.cars) and (
            len(self.finish_order) >= len(self.cars) or self.time >= self.time_cap_s
        )

    def standings(self) -> list[dict[str, Any]]:
        ordered = sorted(self.cars.values(), key=lambda c: (c.finished, c.distance_m), reverse=True)
        return [
            {
                "position": i + 1,
                "key": car.key,
                "lap": car.lap,
                "distance_m": round(car.distance_m, 1),
                "spins": car.spins,
                "finished": car.finished,
                "finish_time": car.finish_time,
            }
            for i, car in enumerate(ordered)
        ]
