"""Race orchestrator tests: cadence, telemetry, and the unavailable-engine rule.

These run headless (no real-time pacing) with mock engines, plus one deliberately
dead engine, which is the case that matters most: an engine that cannot answer
must freeze its car and report itself unavailable, and must never be given a
plausible-looking answer.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from harness import case, run
from mock_server import FakeServer, Handler, free_port

from race import RaceSession, entropy


def config(**overrides) -> dict:
    base = {
        "laps": 1,
        "time_cap_s": 120,
        "sim_hz": 60,
        "tick_hz": 20,
        "policy": "sample",
        "seed": 7,
        "decision_timeout_s": 5,
        "realtime": False,
    }
    base.update(overrides)
    return base


def specs() -> list[dict]:
    return [
        {"key": "mock-a", "label": "Mock A", "type": "mock", "color": "#fff", "noise": 0.15},
        {"key": "mock-b", "label": "Mock B", "type": "mock", "color": "#000", "noise": 0.3},
    ]


def session(tmp: str, engine_specs: list[dict] | None = None, **overrides) -> RaceSession:
    return RaceSession(engine_specs or specs(), config(**overrides), Path(tmp))


@case
def entropy_spans_certain_to_flat() -> None:
    assert entropy({"a": 1.0, "b": 0.0}) == 0.0
    flat = entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    assert abs(flat - 1.0) < 1e-9, flat
    assert 0.0 < entropy({"a": 0.7, "b": 0.3}) < 1.0


@case
def a_mock_race_runs_to_completion_and_writes_telemetry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        live = session(tmp)
        asyncio.run(live.run())
        assert live.finished
        assert live.race.finish_order, "at least one mock racer must finish a lap"
        rows = [json.loads(line) for line in live.telemetry_path.read_text().splitlines() if line.strip()]
        assert len(rows) > 40, len(rows)
        first = rows[0]
        for key in ("t", "racer", "tick", "state", "probabilities", "chosen", "latency_ms", "entropy"):
            assert key in first, key
        assert abs(sum(first["probabilities"].values()) - 1.0) < 1e-6
        assert first["chosen"] in first["probabilities"]
        summary = json.loads(live.summary_path.read_text())
        assert summary["race_id"] == live.race_id
        assert len(summary["racers"]) == 2
        for racer in summary["racers"]:
            assert racer["status"] == "ready", racer
            assert racer["decisions"] > 0
            assert racer["latency_p50_ms"] > 0
        assert summary["wall_time_s"] > 0


@case
def decisions_are_reproducible_from_the_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first = session(tmp, seed=99)
        asyncio.run(first.run())
        second = session(tmp, seed=99)
        asyncio.run(second.run())
        a = [(row["t"], row["chosen"]) for row in map(json.loads, first.telemetry_path.read_text().splitlines())]
        b = [(row["t"], row["chosen"]) for row in map(json.loads, second.telemetry_path.read_text().splitlines())]
        assert a == b, "the same seed must replay the same race"
        third = session(tmp, seed=100)
        asyncio.run(third.run())
        c = [(row["t"], row["chosen"]) for row in map(json.loads, third.telemetry_path.read_text().splitlines())]
        assert c != a, "a different seed must be able to change the race"


@case
def a_finished_car_stops_deciding() -> None:
    """Regression: finished cars were polled forever, recording a frozen state."""
    with tempfile.TemporaryDirectory() as tmp:
        live = session(tmp, laps=1, time_cap_s=60)
        asyncio.run(live.run())
        summary = json.loads(live.summary_path.read_text())
        standings = {row["key"]: row for row in summary["standings"]}
        rows = [json.loads(line) for line in live.telemetry_path.read_text().splitlines() if line.strip()]
        last_t: dict[str, float] = {}
        seen: dict[str, int] = {}
        for row in rows:
            last_t[row["racer"]] = max(last_t.get(row["racer"], 0.0), row["t"])
            seen[row["racer"]] = seen.get(row["racer"], 0) + 1
        finished = [key for key, row in standings.items() if row["finished"]]
        assert finished, "at least one mock racer must finish a lap"
        by_key = {racer["key"]: racer for racer in summary["racers"]}
        for key in finished:
            finish_time = standings[key]["finish_time"]
            assert finish_time is not None
            assert standings[key]["distance_m"] >= 0.9 * live.race.track.length, standings[key]
            # No decision may be requested once the car has crossed the line.
            assert last_t[key] <= finish_time + 0.15, (key, last_t[key], finish_time)
            assert by_key[key]["decisions"] == seen[key]


@case
def a_race_runs_on_the_named_map() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        live = session(tmp, laps=1, time_cap_s=60, track="oval")
        asyncio.run(live.run())
        assert live.track_name == "oval"
        assert abs(live.race.track.length - 700.0) < 1.0, live.race.track.length
        snapshot = live.snapshot()
        assert snapshot["track_name"] == "oval"
        rows = [json.loads(line) for line in live.telemetry_path.read_text().splitlines() if line.strip()]
        assert rows, "a race on a named map must still write telemetry"
        # The state handed to the model carries the map it is actually on.
        assert snapshot["race"]["track"]["length_m"] > 400.0


@case
def an_unreachable_engine_is_unavailable_and_its_car_never_moves() -> None:
    dead = [
        {"key": "dead", "label": "Dead", "type": "llama_cpp", "port": free_port(), "color": "#f00"},
        {"key": "mock-a", "label": "Mock A", "type": "mock", "color": "#fff", "noise": 0.15},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        live = session(tmp, dead, time_cap_s=20)
        asyncio.run(live.run())
        snapshot = live.snapshot()
        by_key = {racer["key"]: racer for racer in snapshot["racers"]}
        assert by_key["dead"]["status"] == "unavailable", by_key["dead"]
        assert by_key["dead"]["decisions"] == 0, "an unavailable engine must record no decisions"
        assert by_key["dead"]["last_probabilities"] is None, "no fabricated distribution"
        assert by_key["dead"]["last_label"] is None
        assert by_key["mock-a"]["decisions"] > 0
        dead_car = next(car for car in snapshot["race"]["cars"] if car["key"] == "dead")
        assert dead_car["distance_m"] == 0.0, dead_car
        assert any("unavailable" in note for note in snapshot["notes"])


@case
def an_engine_that_starts_failing_is_retired_honestly() -> None:
    """A model that answered fine and then breaks must be retired, not covered for.

    This drives the decision path directly rather than through the loop, so the
    failure lands deterministically instead of depending on timing.
    """
    with FakeServer() as server:
        with tempfile.TemporaryDirectory() as tmp:
            live = RaceSession(
                [{"key": "flaky", "label": "Flaky", "type": "llama_cpp", "port": server.port, "color": "#0f0"}],
                config(),
                Path(tmp),
            )

            async def scenario():
                await live.probe()
                runtime = live.runtimes["flaky"]
                assert runtime.status == "ready", runtime.detail
                await live._decide(runtime)
                assert runtime.decisions == 1, "the healthy engine must record a decision"
                Handler.fail_code = 500
                Handler.fail_body = '{"error":{"message":"decision head gone"}}'
                for _ in range(3):
                    await live._decide(runtime)
                return runtime

            runtime = asyncio.run(scenario())
    assert runtime.status == "unavailable", runtime.status
    assert runtime.errors == 3, runtime.errors
    assert "500" in runtime.last_error, runtime.last_error
    assert runtime.decisions == 1, "no decision may be invented once the engine is failing"
    assert live.race.cars["flaky"].frozen is True, "the car must stop, not keep its last input"


if __name__ == "__main__":
    raise SystemExit(run("test_race"))
