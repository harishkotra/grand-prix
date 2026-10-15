"""The race orchestrator: one decision loop per racer, honest status, telemetry.

Design decisions worth stating, because they are the whole point of the piece:

- Every racer decides on its own cadence. A model that answers in 17 ms gets a
  decision roughly every 50 ms; a model that takes 900 ms gets one roughly every
  second and its car holds the last steering input in between. The reaction-time
  spread is therefore visible in the race itself, not just in a table.
- Sampling happens here, with a seeded RNG, from the distribution the model
  returned. The distribution and the drawn action are both recorded, so a race
  replays identically and a bad move can still be read as a close call.
- An engine that cannot answer is reported unavailable and its car is frozen. It
  is never given a fabricated answer and never silently swapped for another model.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from engines import Decision, Engine, EngineError, Option, build_engine
from sim import DEFAULT_TRACK, STEER_BUCKETS, Race, build_track, track_catalogue

QUESTION_ID = "steer"
INSTRUCTIONS = (
    "You are steering a race car on a closed circuit. Choose the steering action for the "
    "next 50 milliseconds. Stay close to the centre line, stay on the track, and complete "
    "the laps as fast as you can. Options run from hard left to hard right."
)
OPTIONS: tuple[Option, ...] = tuple(Option(label, description) for label, _, description in STEER_BUCKETS)
MAX_CONSECUTIVE_ERRORS = 3

# The synthetic racers. They are for the offline test suite and for
# tools/run_headless.py --replay, not for the board: the dashboard lists only real
# engines, so a demo never has a mock on screen next to a model.
MOCK_SPECS: tuple[dict[str, Any], ...] = (
    {"key": "mock-a", "label": "Mock A", "type": "mock", "color": "#8b949e", "noise": 0.2, "note": "synthetic"},
    {"key": "mock-b", "label": "Mock B", "type": "mock", "color": "#6e7681", "noise": 0.4, "note": "synthetic"},
)


def entropy(probabilities: dict[str, float]) -> float:
    """Normalised Shannon entropy in 0..1. 0 is a certain model, 1 is a flat one."""
    values = [p for p in probabilities.values() if p > 0]
    if len(values) < 2:
        return 0.0
    raw = -sum(p * math.log(p) for p in values)
    return raw / math.log(len(values))


@dataclass
class RacerRuntime:
    engine: Engine
    status: str = "loading"
    detail: str = ""
    decisions: int = 0
    tick: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    entropies: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    actions: dict[str, int] = field(default_factory=dict)
    last: Decision | None = None
    last_error: str = ""
    race_time: float = 0.0
    in_flight: bool = False
    next_due: float = 0.0
    active_since: float | None = None

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 5:
            return max(self.latencies) if self.latencies else 0.0
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    def to_json(self) -> dict[str, Any]:
        decisions_per_sec = 0.0
        if self.active_since is not None and self.decisions:
            span = max(1e-6, self.race_time - self.active_since)
            decisions_per_sec = self.decisions / span
        return {
            "key": self.engine.key,
            "label": self.engine.label,
            "color": self.engine.color,
            "note": self.engine.note,
            "kind": self.engine.kind,
            "status": self.status,
            "detail": self.detail,
            "decisions": self.decisions,
            "errors": self.errors,
            "decisions_per_sec": round(decisions_per_sec, 2),
            "latency_p50_ms": round(self.p50_latency, 1),
            "latency_p95_ms": round(self.p95_latency, 1),
            "mean_entropy": round(statistics.fmean(self.entropies), 3) if self.entropies else None,
            "actions": dict(self.actions),
            "last_label": self.last.label if self.last else None,
            "last_probabilities": self.last.probabilities if self.last else None,
            "last_confidence": self.last.confidence if self.last else None,
            "last_error": self.last_error,
        }


class RaceSession:
    def __init__(
        self,
        engine_specs: Sequence[dict[str, Any]],
        race_config: dict[str, Any],
        out_dir: Path,
        race_id: str | None = None,
    ) -> None:
        self.engines = [build_engine(spec) for spec in engine_specs]
        self.config = race_config
        self.policy = race_config.get("policy", "sample")
        self.seed = race_config.get("seed", 20261015)
        self.sim_hz = int(race_config.get("sim_hz", 60))
        self.tick_hz = float(race_config.get("tick_hz", 20))
        self.timeout = float(race_config.get("decision_timeout_s", 12))
        self.realtime = bool(race_config.get("realtime", True))
        self.track_name = race_config.get("track", DEFAULT_TRACK)
        self.race = Race(
            build_track(self.track_name),
            [(engine.key, engine.color) for engine in self.engines],
            laps=int(race_config.get("laps", 3)),
            time_cap_s=float(race_config.get("time_cap_s", 150)),
        )
        self.runtimes = {engine.key: RacerRuntime(engine=engine) for engine in self.engines}
        self.race_id = race_id or time.strftime("%Y%m%d-%H%M%S")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.out_dir / f"{self.race_id}.jsonl"
        self.summary_path = self.out_dir / f"{self.race_id}.summary.json"
        self.rows: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.finished = False
        self._handle = None
        self._task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._step_count = 0
        # The HTTP layer reads this instead of walking the live objects, so the
        # read side never touches a dict the simulation thread is mutating.
        self.latest: dict[str, Any] | None = None
        self._t0 = time.perf_counter()

    # -- lifecycle -----------------------------------------------------------
    async def probe(self) -> None:
        """Probe every engine once. Nothing is assumed to be up."""
        for key, runtime in self.runtimes.items():
            ok, detail = await asyncio.to_thread(runtime.engine.probe)
            runtime.status = "ready" if ok else "unavailable"
            runtime.detail = detail
            if not ok:
                self.race.cars[key].frozen = True
                self.notes.append(f"{runtime.engine.label} unavailable: {detail}")

    async def run(self) -> None:
        await self.probe()
        for runtime in self.runtimes.values():
            if runtime.status == "ready":
                runtime.active_since = 0.0
        dt = 1.0 / self.sim_hz
        self._handle = self.telemetry_path.open("w", encoding="utf-8")
        try:
            while not self.race.over:
                step_start = time.perf_counter()
                self.race.step(dt)
                for key, runtime in self.runtimes.items():
                    runtime.race_time = self.race.time
                    # A finished car keeps its place but stops deciding: the physics
                    # no longer advances it, so asking its model again would only
                    # record the same frozen state over and over.
                    if (
                        runtime.status != "ready"
                        or runtime.in_flight
                        or self.race.cars[key].finished
                        or self.race.time < runtime.next_due
                    ):
                        continue
                    runtime.in_flight = True
                    task = asyncio.create_task(self._decide(runtime))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                self._flush()
                self._step_count += 1
                if self._step_count % 2 == 0:
                    self.latest = self.snapshot()
                if not self.realtime:
                    # Headless: advance only once the decisions that are due have
                    # actually landed, so the run is deterministic and fast rather
                    # than a race between the loop and the thread pool.
                    if self._tasks:
                        await asyncio.gather(*list(self._tasks), return_exceptions=True)
                    await asyncio.sleep(0)
                    continue
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    await asyncio.sleep(dt - elapsed)
        finally:
            self._flush()
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        # Give any in-flight decision a moment to land before writing the summary.
        await asyncio.sleep(0.05)
        self.latest = self.snapshot()
        self._write_summary()
        self.finished = True

    async def _decide(self, runtime: RacerRuntime) -> None:
        key = runtime.engine.key
        state = self.race.state_for(key)
        tick = runtime.tick
        runtime.tick += 1
        seed = f"{self.seed}:{key}"
        try:
            decision = await asyncio.to_thread(
                runtime.engine.decide,
                state,
                QUESTION_ID,
                INSTRUCTIONS,
                OPTIONS,
                self.policy,
                seed,
                tick,
                self.timeout,
            )
        except EngineError as error:
            runtime.errors += 1
            runtime.consecutive_errors += 1
            runtime.last_error = str(error)
            if runtime.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                runtime.status = "unavailable"
                runtime.detail = str(error)
                self.race.cars[key].frozen = True
                self.notes.append(f"{runtime.engine.label} went unavailable after {runtime.errors} errors: {error}")
            return
        finally:
            # Must always release, or the racer goes quiet after a single decision.
            runtime.in_flight = False
        runtime.consecutive_errors = 0
        runtime.decisions += 1
        runtime.latencies.append(decision.latency_ms)
        runtime.entropies.append(entropy(decision.probabilities))
        runtime.actions[decision.label] = runtime.actions.get(decision.label, 0) + 1
        runtime.last = decision
        car = self.race.cars[key]
        car.steer_label = decision.label
        interval = max(1.0 / self.tick_hz, (runtime.p50_latency / 1000.0) * 1.15)
        runtime.next_due = self.race.time + interval
        self.rows.append(
            {
                "t": round(self.race.time, 3),
                "racer": key,
                "tick": tick,
                "state": state,
                "probabilities": {label: round(p, 5) for label, p in decision.probabilities.items()},
                "chosen": decision.label,
                "reported_choice": decision.reported_choice,
                "confidence": decision.confidence,
                "entropy": round(entropy(decision.probabilities), 4),
                "latency_ms": round(decision.latency_ms, 3),
                "output_tokens": decision.output_tokens,
                "warnings": decision.warnings,
                "policy": decision.policy,
                "lateral_m": round(car.lateral, 3),
                "lap": car.lap,
                "off_ticks": car.off_ticks,
            }
        )

    def _flush(self) -> None:
        if not self.rows or self._handle is None:
            return
        for row in self.rows:
            self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._handle.flush()
        self.rows = []

    def _write_summary(self) -> None:
        payload = {
            "race_id": self.race_id,
            "policy": self.policy,
            "seed": self.seed,
            "sim_hz": self.sim_hz,
            "tick_hz": self.tick_hz,
            "laps": self.race.laps,
            "sim_time_s": round(self.race.time, 2),
            "wall_time_s": round(time.perf_counter() - self._t0, 2),
            "notes": self.notes,
            "finish_order": self.race.finish_order,
            "standings": self.race.standings(),
            "racers": [runtime.to_json() for runtime in self.runtimes.values()],
        }
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # -- read side -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "running": self._task is not None and not self._task.done(),
            "finished": self.finished,
            "policy": self.policy,
            "track_name": self.track_name,
            "race": self.race.snapshot(),
            "standings": self.race.standings(),
            "racers": [runtime.to_json() for runtime in self.runtimes.values()],
            "notes": self.notes,
            "telemetry_path": str(self.telemetry_path),
        }


class RaceRunner:
    """Owns the single live session, so the HTTP layer stays dumb."""

    def __init__(self, config: dict[str, Any], out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir
        self.session: RaceSession | None = None

    def engine_catalogue(self) -> list[dict[str, Any]]:
        catalogue = []
        for spec in self.config.get("racers", []):
            engine = build_engine(spec)
            ok, detail = engine.probe()
            catalogue.append(
                {
                    "key": engine.key,
                    "label": engine.label,
                    "color": engine.color,
                    "note": engine.note,
                    "kind": engine.kind,
                    "probe_ok": ok,
                    "probe_detail": detail,
                }
            )
        return catalogue

    def track_catalogue(self) -> list[dict[str, Any]]:
        return track_catalogue()

    def start(
        self,
        keys: Sequence[str] | None = None,
        replay: bool = False,
        track: str | None = None,
    ) -> RaceSession:
        if replay:
            return self.start_replay(track=track)
        specs = list(self.config.get("racers", []))
        if keys:
            wanted = set(keys)
            specs = [spec for spec in specs if spec["key"] in wanted]
        if not specs:
            raise ValueError("no racers selected")
        race_config = dict(self.config.get("race", {}))
        if track:
            race_config["track"] = track
        if self.session is not None:
            self.session.stop()
        self.session = RaceSession(specs, race_config, self.out_dir)
        self.session._task = asyncio.create_task(self.session.run())
        return self.session

    def start_replay(self, track: str | None = None) -> RaceSession:
        race_config = dict(self.config.get("race", {}))
        if track:
            race_config["track"] = track
        if self.session is not None:
            self.session.stop()
        self.session = RaceSession(
            list(MOCK_SPECS),
            race_config,
            self.out_dir,
            race_id=f"{time.strftime('%Y%m%d-%H%M%S')}-replay",
        )
        self.session._task = asyncio.create_task(self.session.run())
        return self.session
