# Run output (SIMULATION_DESIGN.md §8.1): events.jsonl, decisions.jsonl, generations.csv, agents.csv.
#
# Usage from the generation loop:
#     with RunLog(cfg) as log:
#         for gen in range(cfg.population.G):
#             log.start_generation(gen, agents)
#             ... run_day(cfg, village, brain, rng, log); log.end_day(village) ...
#             log.end_generation(village.agents)
import csv
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml

from config import Config
from genes import Gene, level
from metrics import generation_metrics


class LogEventType(Enum):
    decision = "decision"
    gather = "gather"
    hunt = "hunt"
    raid = "raid"
    raid_aborted = "raid_aborted"
    trade = "trade"
    gift = "gift"
    night = "night"
    death = "death"
    generation = "generation"
    population = "population"
    day_end = "day_end"


Phase = Literal["morning", "gathering", "hunt", "raid", "market", "night", "generation"]


@dataclass
class LogEvent:
    run: str
    gen: int
    day: int
    phase: Phase
    type: LogEventType
    payload: dict

    def to_dict(self) -> dict:
        """Flat JSON-ready dict: {run, gen, day, phase, type, **payload}."""
        head = {"run": self.run, "gen": self.gen, "day": self.day, "phase": self.phase, "type": self.type.value}
        return _plain({**head, **self.payload})


class RunLog:
    def __init__(self, cfg: Config, run_dir: str | Path | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir or Path(cfg.log.out_dir) / cfg.run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        self._events = open(self.run_dir / "events.jsonl", "w", encoding="utf-8")
        self._decisions = open(self.run_dir / "decisions.jsonl", "w", encoding="utf-8") if cfg.log.prompts else None
        for name in ("generations.csv", "agents.csv"):
            (self.run_dir / name).unlink(missing_ok=True)
        self.gen = 0
        self.events: list[dict] = []  # current generation only, consumed by the metrics
        self._call = 0
        self._last_day = 0
        self._t0 = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._events.close()
        if self._decisions:
            self._decisions.close()

    def start_generation(self, gen: int, agents) -> None:
        self.gen = gen
        self.events = []
        self._last_day = 0
        self._t0 = time.perf_counter()
        self.emit(LogEventType.population, "generation", 0, agents=[
            {"id": a.id, "genome": {g.value: a.genome[g] for g in Gene}, "parents": a.parents,
             "health": a.health, "stock": a.stock} for a in agents])

    def end_day(self, village) -> None:
        """Marks the day as complete and flushes it, so a viewer can follow the run live."""
        self.emit(LogEventType.day_end, "night", village.day_count, season=village.season, alive=village.n_alive)
        self._events.flush()

    def emit(self, type_: LogEventType | str, phase: Phase, day: int, **payload) -> None:
        event = LogEvent(self.cfg.run_name, self.gen, day, phase, LogEventType(type_), payload).to_dict()
        _write_json(self._events, event)
        self.events.append(event)
        self._last_day = day

    def decision(self, day: int, phase: Phase, agent, state: str, questions: dict, answers: dict) -> None:
        """One `decision` event per question; state and questions go to decisions.jsonl (joined on `call`)."""
        self._call += 1
        persona_levels = {g.value: level(agent.genome[g]) for g in Gene}
        for name, ans in answers.items():
            self.emit(LogEventType.decision, phase, day, agent=agent.id, name=name, call=self._call,
                      persona_levels=persona_levels, raw_probs=ans.raw, sharpened_probs=ans.sharpened,
                      chosen=ans.chosen, forced=ans.forced, latency_ms=ans.latency_ms)
        if self._decisions:
            _write_json(self._decisions, _plain({"run": self.cfg.run_name, "gen": self.gen, "day": day, "phase": phase,
                                                 "call": self._call, "agent": agent.id,
                                                 "state": state, "questions": questions}))

    def end_generation(self, agents) -> dict:
        """Compute §8.2 metrics, log the `generation` event and append to the CSVs; returns the generation row."""
        wall_seconds = time.perf_counter() - self._t0
        row, agent_rows = generation_metrics(self.gen, agents, self.events, self.cfg, wall_seconds)
        self.emit(LogEventType.generation, "generation", self._last_day, **row)
        _append_csv(self.run_dir / "generations.csv", [row])
        _append_csv(self.run_dir / "agents.csv", agent_rows)
        self._events.flush()
        if self._decisions:
            self._decisions.flush()
        return row


def _plain(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {_plain(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    return obj


def _write_json(f, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
