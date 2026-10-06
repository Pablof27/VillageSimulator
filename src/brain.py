# Decision engine (SIMULATION_DESIGN.md §6.1, §6.2, §6.5): Laya probabilities -> sharpen -> sample.
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from typing import Literal, Protocol

from config import BrainConfig


@dataclass
class Answer:
    kind: Literal["choice", "noul"]
    raw: dict[str, float]        # engine probabilities; noul as {"yes": p, "no": 1 - p}
    sharpened: dict[str, float]  # after tau
    chosen: str                  # sampled label ("yes"/"no" for noul)
    forced: bool                 # only one legal option: the engine was not called
    latency_ms: float


class Brain(Protocol):
    def decide(self, state: str, questions: dict[str, dict]) -> dict[str, Answer]: ...


def decide(state: str, questions: dict[str, dict], brain: Brain) -> dict[str, Answer]:
    return brain.decide(state, questions)


def make_brain(cfg: BrainConfig, rng) -> Brain:
    if cfg.kind == "policy":
        return PolicyBrain(cfg.policy_weights, tau=cfg.tau, rng=rng)
    brain = MockBrain(tau=cfg.tau, rng=rng) if cfg.kind == "mock" else LayaBrain(
        cfg.model, dtype=cfg.dtype, device=cfg.device, batch_size=cfg.batch_size,
        compile=cfg.compile, cache_prompts=cfg.cache_prompts, tau=cfg.tau, rng=rng)
    return CachedBrain(brain) if cfg.cache else brain


# ---- sampling (§6.2) --------------------------------------------------------

def sharpen(p: dict[str, float], tau: float) -> dict[str, float]:
    """q_i ∝ p_i^(1/tau); tau = 0 puts all mass on the argmax (ties -> first label)."""
    if tau == 0:
        best = max(p, key=p.get)
        return {k: float(k == best) for k in p}
    top = max(p.values())
    # Dividing by the max keeps the largest weight at 1, so small tau cannot underflow to all zeros.
    weights = {k: (v / top) ** (1 / tau) for k, v in p.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def sample(p: dict[str, float], rng) -> str:
    return rng.choices(list(p), weights=list(p.values()))[0]


# ---- brains -----------------------------------------------------------------

class SamplingBrain:

    def __init__(self, tau: float, rng):
        self.tau = tau
        self.rng = rng

    def probabilities(self, state: str, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
        raise NotImplementedError

    def decide(self, state: str, questions: dict[str, dict]) -> dict[str, Answer]:
        to_ask = {name: q for name, q in questions.items() if not _is_forced(q)}
        start = time.perf_counter()
        raw = self.probabilities(state, to_ask) if to_ask else {}
        latency_ms = (time.perf_counter() - start) * 1000

        answers = {}
        for name, q in questions.items():
            if _is_forced(q):
                label = _labels(q)[0]
                answers[name] = Answer(q["type"], {label: 1.0}, {label: 1.0}, label, True, 0.0)
                continue
            p = _validated(name, q, raw.get(name))
            sharpened = sharpen(p, self.tau)
            answers[name] = Answer(q["type"], p, sharpened, sample(sharpened, self.rng), False, latency_ms)
        return answers


class LayaBrain(SamplingBrain):
    def __init__(self, model_id: str, *, dtype="float16", device="gpu", batch_size=8,
                 compile=False, cache_prompts=False, tau=0.5, rng):
        super().__init__(tau, rng)
        import laya_mlx  # heavy import; not needed for MockBrain runs

        with warnings.catch_warnings():
            # Expected: the checkpoint's choice:11+ temperature is clamped (§1.1); we never ask > 10 options.
            warnings.filterwarnings("ignore", message="laya-mlx: this checkpoint ships temperatures",
                                    category=RuntimeWarning)
            self.agent = laya_mlx.load(model_id, dtype=dtype, device=device, batch_size=batch_size,
                                       compile=compile, cache_prompts=cache_prompts)

    def probabilities(self, state, questions):
        answers = self.agent.predict(state, questions)["answers"]
        return {name: _from_laya(answers[name]) for name in questions}


class MockBrain(SamplingBrain):

    def __init__(self, rule: Literal["uniform", "first"] = "uniform", *, tau=0.5, rng):
        super().__init__(tau, rng)
        self.rule = rule

    def probabilities(self, state, questions):
        out = {}
        for name, q in questions.items():
            labels = _labels(q)
            if self.rule == "first":
                out[name] = {label: float(i == 0) for i, label in enumerate(labels)}
                continue
            out[name] = {label: 1 / len(labels) for label in labels}
        return out


class PolicyBrain(SamplingBrain):
    """For the analytic policy catalogue (§6.7): each question already carries its distribution."""

    def __init__(self, weights_path: str, *, tau=1.0, rng):
        super().__init__(tau, rng)
        from policies import load_weights  # validates the weights file now rather than on the first decision

        load_weights(weights_path)

    def probabilities(self, state, questions):
        return {name: q["probabilities"] for name, q in questions.items()}


class CachedBrain(SamplingBrain):

    def __init__(self, inner: SamplingBrain):
        super().__init__(inner.tau, inner.rng)
        self.inner = inner
        self.cache: dict[str, dict[str, dict[str, float]]] = {}
        self.hits = 0

    def probabilities(self, state, questions):
        key = hashlib.sha256((state + json.dumps(questions, sort_keys=True)).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.cache[key] = self.inner.probabilities(state, questions)
        return self.cache[key]


# ---- helpers ----------------------------------------------------------------

def _labels(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "choice":
        return list(q["criteria"])
    raise ValueError(f"Unsupported question type {q['type']!r}; only choice and noul are used")


def _is_forced(q: dict) -> bool:
    return q["type"] == "choice" and len(q["criteria"]) == 1


def _from_laya(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    return answer["probabilities"]


def _validated(name: str, q: dict, p: dict[str, float] | None) -> dict[str, float]:
    if p is None or set(p) != set(_labels(q)):
        raise ValueError(f"Question {name!r}: labels {sorted(p or {})} do not match {sorted(_labels(q))}")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()) or sum(p.values()) <= 0:
        raise ValueError(f"Question {name!r}: invalid probabilities {p}")
    return {label: p[label] for label in _labels(q)}
