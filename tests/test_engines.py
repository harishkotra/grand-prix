"""Adapter tests: request shape, parsing, sampling, and every failure path.

No model is needed. The fake server in `mock_server.py` stands in, and the
failure paths (HTTP 500, closed port, missing option, missing credentials) are
exercised for real rather than mocked away.
"""

from __future__ import annotations

import os

from harness import case, run
from mock_server import FakeServer, Handler, free_port

from engines import (
    EngineError,
    LlamaCppEngine,
    MockEngine,
    OllamaEngine,
    Option,
    WorkersAIEngine,
    draw_action,
    normalise,
    parse_answer,
    question_body,
    unwrap_envelope,
)
from sim import heuristic_steer

OPTIONS = (
    Option("hard_left", "hard left"),
    Option("left", "left"),
    Option("straight", "straight"),
    Option("right", "right"),
    Option("hard_right", "hard right"),
)


def _engine(port: int) -> LlamaCppEngine:
    return LlamaCppEngine("test", "Test", port)


@case
def llama_engine_parses_and_normalises() -> None:
    with FakeServer() as server:
        Handler.probabilities = {"hard_left": 1.0, "left": 1.0, "straight": 2.0, "right": 1.0, "hard_right": 1.0}
        decision = _engine(server.port).decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
    assert decision.label == "straight", decision.label
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-9
    assert abs(decision.probabilities["straight"] - 1.0 / 3.0) < 1e-9
    assert decision.output_tokens == 0


@case
def request_body_matches_the_contract() -> None:
    body = question_body({"car": {}}, "steer", "instructions", OPTIONS)
    question = body["questions"]["steer"]
    assert question["type"] == "choice"
    assert question["instructions"] == "instructions"
    assert isinstance(question["criteria"], dict)
    assert list(question["criteria"].keys()) == [option.label for option in OPTIONS]
    assert "model" not in body


@case
def sampling_is_reproducible_and_distribution_driven() -> None:
    probabilities = {"a": 0.7, "b": 0.3}
    first = draw_action(probabilities, "sample", "seed:1", 5)
    second = draw_action(probabilities, "sample", "seed:1", 5)
    assert first == second, "same seed and tick must give the same draw"
    assert draw_action(probabilities, "argmax", "seed:1", 5) == "a"
    # Over many ticks the draws must respect the distribution, not just argmax it.
    draws = [draw_action(probabilities, "sample", "seed:2", tick) for tick in range(400)]
    share_b = draws.count("b") / len(draws)
    assert 0.2 < share_b < 0.42, share_b


@case
def missing_option_is_an_error_not_a_guess() -> None:
    payload = {
        "answers": {
            "steer": {
                "type": "choice",
                "choice": "left",
                "probabilities": {"hard_left": 0.4, "left": 0.6},
            }
        }
    }
    try:
        parse_answer(payload, "test", "steer", OPTIONS, "argmax", "seed", 1, 1.0)
    except EngineError as error:
        assert "omitted options" in str(error), str(error)
    else:
        raise AssertionError("a short probability map must not be accepted")


@case
def zero_probabilities_are_rejected() -> None:
    try:
        normalise({"a": 0.0, "b": 0.0})
    except EngineError:
        return
    raise AssertionError("all-zero probabilities must be rejected")


@case
def http_error_surfaces_the_body() -> None:
    with FakeServer() as server:
        Handler.fail_code = 500
        Handler.fail_body = '{"error":{"message":"decision head missing"}}'
        try:
            _engine(server.port).decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
        except EngineError as error:
            assert "500" in str(error) and "decision head missing" in str(error), str(error)
        else:
            raise AssertionError("HTTP 500 must raise")


@case
def not_a_decision_model_501_is_reported() -> None:
    with FakeServer() as server:
        Handler.fail_code = 501
        Handler.fail_body = '{"error":{"code":501,"message":"This model is not a decision model"}}'
        try:
            _engine(server.port).decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
        except EngineError as error:
            assert "not a decision model" in str(error), str(error)
        else:
            raise AssertionError("501 must raise")


@case
def closed_port_reads_unreachable() -> None:
    port = free_port()
    engine = _engine(port)
    ok, detail = engine.probe()
    assert ok is False and str(port) in detail, detail
    try:
        engine.decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
    except EngineError as error:
        assert "unreachable" in str(error), str(error)
    else:
        raise AssertionError("a closed port must raise")


@case
def server_that_demands_a_model_field_gets_a_visible_retry() -> None:
    with FakeServer() as server:
        Handler.require_model = True
        engine = _engine(server.port)
        engine.model = "ggml-org/Julia-1-GGUF:Q8_0"
        decision = engine.decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
        assert decision.label == "straight"
        assert len(Handler.requests) == 2, "expected one rejected call then one retry"
        assert Handler.requests[1]["model"] == "ggml-org/Julia-1-GGUF:Q8_0"


@case
def ollama_sends_the_model_field() -> None:
    with FakeServer() as server:
        engine = OllamaEngine("tev1", "Tev1", "tev1:0.8b", port=server.port)
        engine.decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
    assert Handler.last_request["model"] == "tev1:0.8b"


@case
def workers_ai_without_credentials_is_unavailable() -> None:
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AUTH_TOKEN"):
        os.environ.pop(name, None)
    engine = WorkersAIEngine("clef-flash", "Clef-flash", "@cf/cloudflare/clef-flash")
    ok, detail = engine.probe()
    assert ok is False and "CLOUDFLARE_ACCOUNT_ID" in detail, detail
    try:
        engine.decide({"task": "steer"}, "steer", "go", OPTIONS, "argmax", "seed", 1)
    except EngineError as error:
        assert "CLOUDFLARE" in str(error), str(error)
    else:
        raise AssertionError("missing credentials must raise, never fake an answer")


@case
def workers_ai_result_envelope_is_unwrapped() -> None:
    payload = {
        "result": {
            "answers": {
                "steer": {
                    "type": "choice",
                    "choice": "left",
                    "probabilities": {option.label: 0.2 for option in OPTIONS},
                    "confidence": 0.4,
                }
            }
        },
        "success": True,
    }
    body = unwrap_envelope(payload)
    assert "answers" in body
    decision = parse_answer(payload, "clef-flash", "steer", OPTIONS, "argmax", "seed", 1, 3.0)
    assert decision.label in {option.label for option in OPTIONS}


@case
def mock_engine_steers_back_toward_the_line() -> None:
    engine = MockEngine("mock", "Mock")
    straight = [{"direction": "straight", "radius_m": None, "distance_m": 0.0, "length_m": 60.0}]
    # Well to the right of the centre line on a straight: must steer left.
    assert engine.ideal_label({"car": {"lateral_m": 6.0, "heading_error_deg": 0.0}, "track": {"bends_ahead": straight}}, OPTIONS) in {
        "hard_left",
        "left",
    }
    # Angled into the wall on a straight: must also steer left.
    assert engine.ideal_label({"car": {"lateral_m": 0.0, "heading_error_deg": 8.0}, "track": {"bends_ahead": straight}}, OPTIONS) in {
        "hard_left",
        "left",
    }
    # A tight left-hander right under the car: the tightest left bucket.
    tight_left = [{"direction": "left", "radius_m": 18.0, "distance_m": 0.0, "length_m": 40.0}]
    assert engine.ideal_label({"car": {"lateral_m": 0.0, "heading_error_deg": 0.0}, "track": {"bends_ahead": tight_left}}, OPTIONS) == "hard_left"
    # A gentle right-hander: the gentle right bucket, not the hard one.
    gentle_right = [{"direction": "right", "radius_m": 45.0, "distance_m": 0.0, "length_m": 60.0}]
    assert engine.ideal_label({"car": {"lateral_m": 0.0, "heading_error_deg": 0.0}, "track": {"bends_ahead": gentle_right}}, OPTIONS) == "right"


@case
def heuristic_driver_anticipates_the_next_bend() -> None:
    straight_now_then_left = [
        {"direction": "straight", "radius_m": None, "distance_m": 0.0, "length_m": 30.0},
        {"direction": "left", "radius_m": 18.0, "distance_m": 18.0, "length_m": 40.0},
    ]
    assert heuristic_steer(straight_now_then_left, 0.0, 0.0) == "hard_left"
    far_away_bend = [
        {"direction": "straight", "radius_m": None, "distance_m": 0.0, "length_m": 200.0},
        {"direction": "left", "radius_m": 18.0, "distance_m": 120.0, "length_m": 40.0},
    ]
    assert heuristic_steer(far_away_bend, 0.0, 0.0) == "straight"


if __name__ == "__main__":
    raise SystemExit(run("test_engines"))
