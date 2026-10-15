"""Engine adapters: one uniform decision call in front of every racer.

Every racer in this project is a *decision model*. A decision model reads a state
once and returns a probability for every option you supplied, in a single forward
pass. Nothing is generated, nothing is parsed, and there is no refusal path. The
wire contract is `POST /v1/systemone`.

Three rules this module exists to enforce:

1. An unavailable engine reports unavailable. It never returns a fabricated
   answer, and it never silently degrades into a different engine.
2. The full distribution is kept, not just the winning label. The whole point of
   the showcase is that you can see the model's belief, so the belief is data.
3. Sampling is ours, not the model's. We sample from the returned distribution
   with a seeded RNG, so a race is reproducible and both the distribution and the
   drawn action are recorded.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from sim import heuristic_steer

SYSTEMONE_PATH = "/v1/systemone"


class EngineError(RuntimeError):
    """An engine could not answer. Never swallowed into a plausible answer."""


@dataclass(frozen=True)
class Option:
    """One option we hand the model. Order matters: options are labelled in order."""

    label: str
    description: str


@dataclass
class Decision:
    racer: str
    label: str
    probabilities: dict[str, float]
    confidence: float | None
    latency_ms: float
    policy: str
    reported_choice: str | None = None
    warnings: list[str] = field(default_factory=list)
    output_tokens: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "racer": self.racer,
            "label": self.label,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 3),
            "policy": self.policy,
            "reported_choice": self.reported_choice,
            "warnings": self.warnings,
            "output_tokens": self.output_tokens,
        }


def normalise(probabilities: dict[str, Any]) -> dict[str, float]:
    """Clamp to non-negative and rescale to sum to 1, preserving option order."""
    cleaned: dict[str, float] = {}
    for label, value in probabilities.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        cleaned[label] = number if number > 0.0 else 0.0
    total = sum(cleaned.values())
    if total <= 0.0:
        raise EngineError("every returned probability was zero or missing")
    return {label: value / total for label, value in cleaned.items()}


def draw_action(
    probabilities: dict[str, float],
    policy: str,
    seed: Any,
    tick: int,
) -> str:
    """Pick one option. `sample` draws from the distribution; `argmax` takes the top.

    Sampling is seeded per (racer, tick) so the same race replays identically.
    `random.Random(str)` is stable across processes in CPython (sha512 based), so
    the seed string does not depend on PYTHONHASHSEED.
    """
    if not probabilities:
        raise EngineError("no options to choose from")
    if policy == "argmax":
        return max(probabilities.items(), key=lambda item: item[1])[0]
    if policy != "sample":
        raise EngineError(f"unknown policy {policy!r}")
    rng = random.Random(f"{seed}:{tick}")
    threshold = rng.random()
    cumulative = 0.0
    for label, probability in probabilities.items():
        cumulative += probability
        if threshold <= cumulative:
            return label
    return next(reversed(probabilities))


def question_body(
    state: Any,
    question_id: str,
    instructions: str,
    options: Sequence[Option],
    model: str | None = None,
) -> dict[str, Any]:
    """The request half of the contract. `criteria` must be label -> description."""
    body: dict[str, Any] = {
        "state": state,
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {option.label: option.description for option in options},
            }
        },
    }
    if model is not None:
        body["model"] = model
    return body


def _http_post(url: str, body: dict[str, Any], timeout: float, headers: dict[str, str]) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise EngineError(f"HTTP {error.code} from {url}: {detail}") from error
    except urllib.error.URLError as error:
        raise EngineError(f"unreachable {url}: {error.reason}") from error
    except TimeoutError as error:
        raise EngineError(f"timeout after {timeout}s calling {url}") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise EngineError(f"non-JSON reply from {url}: {raw[:200]!r}") from error


def unwrap_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Workers AI and similar proxies wrap the contract in a `result` envelope."""
    if isinstance(payload.get("answers"), dict):
        return payload
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("answers"), dict):
        return result
    raise EngineError(f"reply has no answers block: {json.dumps(payload)[:200]}")


def parse_answer(
    payload: dict[str, Any],
    racer: str,
    question_id: str,
    options: Sequence[Option],
    policy: str,
    seed: Any,
    tick: int,
    latency_ms: float,
) -> Decision:
    """Shared parsing of the answer half of the contract."""
    body = unwrap_envelope(payload)
    answers = body.get("answers") or {}
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        raise EngineError(f"no answer for question {question_id!r}")
    raw_probabilities = answer.get("probabilities")
    if not isinstance(raw_probabilities, dict) or not raw_probabilities:
        raise EngineError(f"answer for {question_id!r} carried no probabilities")
    expected = [option.label for option in options]
    missing = [label for label in expected if label not in raw_probabilities]
    if missing:
        raise EngineError(f"answer omitted options {missing}")
    probabilities = normalise({label: raw_probabilities[label] for label in expected})
    warnings = list(body.get("warnings") or [])
    usage = body.get("usage") or {}
    output_tokens = usage.get("output_tokens")
    return Decision(
        racer=racer,
        label=draw_action(probabilities, policy, seed, tick),
        probabilities=probabilities,
        confidence=answer.get("confidence"),
        latency_ms=latency_ms,
        policy=policy,
        reported_choice=answer.get("choice"),
        warnings=warnings,
        output_tokens=output_tokens,
    )


class Engine:
    """Base class. Subclasses only implement the request half."""

    kind = "base"

    def __init__(self, key: str, label: str, color: str = "#ffffff", note: str = "") -> None:
        self.key = key
        self.label = label
        self.color = color
        self.note = note

    # -- to implement -------------------------------------------------------
    def probe(self) -> tuple[bool, str]:
        raise NotImplementedError

    def request_body(self, state: Any, question_id: str, instructions: str, options: Sequence[Option]) -> dict[str, Any]:
        raise NotImplementedError

    def endpoint(self) -> str:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------
    def headers(self) -> dict[str, str]:
        return {}

    def decide(
        self,
        state: Any,
        question_id: str,
        instructions: str,
        options: Sequence[Option],
        policy: str,
        seed: Any,
        tick: int,
        timeout: float = 15.0,
    ) -> Decision:
        body = self.request_body(state, question_id, instructions, options)
        started = time.perf_counter()
        try:
            payload = _http_post(self.endpoint(), body, timeout, self.headers())
        except EngineError as error:
            if "model" in str(error).lower() and "model" not in body:
                # A few servers require an explicit model field. Retry once, visibly.
                body["model"] = getattr(self, "model", None) or self.key
                payload = _http_post(self.endpoint(), body, timeout, self.headers())
            else:
                raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        return parse_answer(payload, self.key, question_id, options, policy, seed, tick, latency_ms)


class LlamaCppEngine(Engine):
    """llama.cpp's native decision endpoint. No option cap, no model field needed."""

    kind = "llama_cpp"

    def __init__(self, key: str, label: str, port: int, model: str = "", color: str = "#ffffff", note: str = "") -> None:
        super().__init__(key, label, color, note)
        self.port = port
        self.model = model

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}{SYSTEMONE_PATH}"

    def probe(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
            return True, str(body.get("status", "ok"))
        except urllib.error.URLError as error:
            return False, f"no server on :{self.port} ({error.reason})"
        except Exception as error:  # noqa: BLE001 - probe must never raise
            return False, f"probe failed: {error}"

    def request_body(self, state: Any, question_id: str, instructions: str, options: Sequence[Option]) -> dict[str, Any]:
        return question_body(state, question_id, instructions, options)


class OllamaEngine(Engine):
    """Ollama 0.35+ serves the same contract with an explicit model field."""

    kind = "ollama"

    def __init__(self, key: str, label: str, model: str, port: int = 11434, color: str = "#ffffff", note: str = "") -> None:
        super().__init__(key, label, color, note)
        self.port = port
        self.model = model

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}{SYSTEMONE_PATH}"

    def probe(self) -> tuple[bool, str]:
        request = urllib.request.Request(self.endpoint(), data=b"{}", method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return True, f"unexpected {response.status}"
        except urllib.error.HTTPError as error:
            if error.code == 400:
                return True, "endpoint present (model required)"
            return False, f"HTTP {error.code}"
        except urllib.error.URLError as error:
            return False, f"no server on :{self.port} ({error.reason})"

    def request_body(self, state: Any, question_id: str, instructions: str, options: Sequence[Option]) -> dict[str, Any]:
        return question_body(state, question_id, instructions, options, model=self.model)


class WorkersAIEngine(Engine):
    """Cloudflare Workers AI. API-only route for a model whose weights we do not run."""

    kind = "workers_ai"

    def __init__(
        self,
        key: str,
        label: str,
        model: str,
        account_env: str = "CLOUDFLARE_ACCOUNT_ID",
        token_env: str = "CLOUDFLARE_AUTH_TOKEN",
        color: str = "#ffffff",
        note: str = "",
    ) -> None:
        super().__init__(key, label, color, note)
        self.model = model
        self.account_env = account_env
        self.token_env = token_env

    def credentials(self) -> tuple[str, str] | None:
        account = os.environ.get(self.account_env)
        token = os.environ.get(self.token_env)
        if account and token:
            return account, token
        return None

    def endpoint(self) -> str:
        credentials = self.credentials()
        if credentials is None:
            raise EngineError(f"needs {self.account_env} and {self.token_env}")
        return f"https://api.cloudflare.com/client/v4/accounts/{credentials[0]}/ai/run/{self.model}"

    def headers(self) -> dict[str, str]:
        credentials = self.credentials()
        if credentials is None:
            raise EngineError(f"needs {self.account_env} and {self.token_env}")
        return {"Authorization": f"Bearer {credentials[1]}"}

    def probe(self) -> tuple[bool, str]:
        if self.credentials() is None:
            return False, f"needs {self.account_env} + {self.token_env}"
        return True, "credentials present (not exercised)"

    def request_body(self, state: Any, question_id: str, instructions: str, options: Sequence[Option]) -> dict[str, Any]:
        return question_body(state, question_id, instructions, options, model=self.model)


class MockEngine(Engine):
    """Deterministic synthetic racer. Used by the tests and by replay mode.

    It is a mock and says so. It reads the same state the real models get, picks a
    plausible steering action from it, and returns a distribution around that
    action, so the race is watchable and the physics are exercised without any
    model running.
    """

    kind = "mock"

    def __init__(self, key: str, label: str, color: str = "#8b949e", note: str = "mock", noise: float = 0.25) -> None:
        super().__init__(key, label, color, note)
        self.noise = noise

    def probe(self) -> tuple[bool, str]:
        return True, "mock"

    def endpoint(self) -> str:
        return "mock://local"

    def request_body(self, state: Any, question_id: str, instructions: str, options: Sequence[Option]) -> dict[str, Any]:
        return question_body(state, question_id, instructions, options)

    def ideal_label(self, state: dict[str, Any], options: Sequence[Option]) -> str:
        car = state.get("car", {})
        track = state.get("track", {})
        return heuristic_steer(
            track.get("bends_ahead") or [],
            float(car.get("lateral_m", 0.0)),
            float(car.get("heading_error_deg", 0.0)),
        )

    def decide(
        self,
        state: Any,
        question_id: str,
        instructions: str,
        options: Sequence[Option],
        policy: str,
        seed: Any,
        tick: int,
        timeout: float = 15.0,
    ) -> Decision:
        started = time.perf_counter()
        labels = [option.label for option in options]
        ideal = self.ideal_label(state if isinstance(state, dict) else {}, options)
        rng = random.Random(f"mock:{self.key}:{seed}:{tick}")
        weights = {label: (1.0 - self.noise) if label == ideal else self.noise / (len(labels) - 1) for label in labels}
        # A small jitter so the mock is not perfectly deterministic across ticks.
        weights[ideal] += rng.uniform(-0.05, 0.05)
        probabilities = normalise(weights)
        latency_ms = (time.perf_counter() - started) * 1000.0 + rng.uniform(1.0, 6.0)
        return Decision(
            racer=self.key,
            label=draw_action(probabilities, policy, seed, tick),
            probabilities=probabilities,
            confidence=max(probabilities.values()),
            latency_ms=latency_ms,
            policy=policy,
            reported_choice=ideal,
        )


def build_engine(spec: dict[str, Any]) -> Engine:
    kind = spec.get("type", "mock")
    common = {"color": spec.get("color", "#ffffff"), "note": spec.get("note", "")}
    if kind == "llama_cpp":
        return LlamaCppEngine(spec["key"], spec.get("label", spec["key"]), int(spec["port"]), spec.get("model", ""), **common)
    if kind == "ollama":
        return OllamaEngine(spec["key"], spec.get("label", spec["key"]), spec["model"], int(spec.get("port", 11434)), **common)
    if kind == "workers_ai":
        return WorkersAIEngine(
            spec["key"],
            spec.get("label", spec["key"]),
            spec["model"],
            spec.get("account_env", "CLOUDFLARE_ACCOUNT_ID"),
            spec.get("token_env", "CLOUDFLARE_AUTH_TOKEN"),
            **common,
        )
    if kind == "mock":
        return MockEngine(spec["key"], spec.get("label", spec["key"]), noise=float(spec.get("noise", 0.25)), **common)
    raise EngineError(f"unknown engine type {kind!r}")
