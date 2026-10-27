# Reads a run's events.jsonl (SIMULATION_DESIGN.md §8.1) back as GenStart / DayLog / GenEnd items.
# Follows a file that is still being written (live viewing). Logs written before the `population` and
# `day_end` events existed fall back to agents.csv for the genomes.
import csv
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from config import Config
from genes import Gene
from resources import Resource

GENES = tuple(g.value for g in Gene)
GENE_COLS = dict(zip(GENES, ("agg", "alt", "risk", "spec")))
OTHER = {"food": "wood", "wood": "food"}
_HEAD = re.compile(r'"gen": (\d+), "day": (\d+), "phase": "(\w+)", "type": "(\w+)"')


@dataclass(slots=True)
class Vitals:
    health: int
    food: int
    wood: int
    alive: bool = True

    def copy(self) -> "Vitals":
        return Vitals(self.health, self.food, self.wood, self.alive)

    def add(self, resource: str, n: int) -> None:
        setattr(self, resource, getattr(self, resource) + n)


@dataclass
class AgentInfo:
    id: int
    genome: dict[str, float]
    parents: tuple[int, ...] = ()


@dataclass
class Plan:
    activity: str = "rest"  # gather_food | gather_wood | hunt | raid | rest
    route: str | None = None
    target: int | None = None


@dataclass
class GenStart:
    gen: int
    agents: dict[int, AgentInfo]
    state: dict[int, Vitals]


@dataclass
class DayLog:
    gen: int
    day: int
    season: dict[str, float]
    start: dict[int, Vitals]
    end: dict[int, Vitals]
    plans: dict[int, Plan]
    gathers: list[dict] = field(default_factory=list)
    hunt: dict | None = None
    raids: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    gifts: list[dict] = field(default_factory=list)
    nights: list[dict] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)


@dataclass
class GenEnd:
    gen: int
    metrics: dict


def load_config(run_dir: Path) -> Config:
    data = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8")) or {}
    data["brain"] = {**(data.get("brain") or {}), "kind": "mock"}  # the viewer never needs the decision engine
    return Config.from_dict(data)


def generations_logged(run_dir: Path) -> int:
    """Number of finished generations (rows of generations.csv)."""
    path = Path(run_dir) / "generations.csv"
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


class EventReader:
    """Turns events.jsonl into GenStart / DayLog / GenEnd items, one `next_item()` at a time.

    With `follow=True` a missing or unfinished file means "not yet": `next_item()` returns None and can be called
    again later. `want_gen(gen)` skips whole generations cheaply (their lines are not parsed). `not_before` ignores
    an events.jsonl older than that timestamp (a previous run in the same directory)."""

    def __init__(self, run_dir, cfg: Config | None = None, follow=False, want_gen=None, not_before=0.0):
        self.run_dir = Path(run_dir)
        self.cfg = cfg or load_config(self.run_dir)
        self.follow = follow
        self.want_gen = want_gen or (lambda gen: True)
        self.not_before = not_before
        self.gene_means: dict[int, dict[str, float]] = {}
        self.generations_done = 0
        self.eof = False
        self._file = None
        self._partial = ""
        self._items: deque = deque()
        self._gen: int | None = None
        self._started = False
        self._state: dict[int, Vitals] = {}
        self._day: tuple[int, int] | None = None
        self._events: list[dict] = []
        self._season: dict | None = None
        self._csv: dict | None = None

    @property
    def finished(self) -> bool:
        return not self._items and (self.eof or self.generations_done >= self.cfg.population.G)

    def next_item(self, budget: int = 20000):
        while not self._items and budget > 0:
            line = self._readline()
            if line is None:
                break
            self._consume(line)
            budget -= 1
        return self._items.popleft() if self._items else None

    # ---- reading ------------------------------------------------------------

    def _readline(self) -> str | None:
        if self._file is None:
            path = self.run_dir / "events.jsonl"
            if not path.exists() or os.path.getmtime(path) < self.not_before:
                self.eof = not self.follow
                return None
            self._file = open(path, encoding="utf-8")
        chunk = self._file.readline()
        if not chunk:
            self.eof = not self.follow
            return None
        if not chunk.endswith("\n"):
            self._partial += chunk
            return None
        line, self._partial = self._partial + chunk, ""
        return line

    def _consume(self, line: str) -> None:
        head = _HEAD.search(line, 0, 400)
        if head is None:
            return
        gen, day, phase, kind = int(head[1]), int(head[2]), head[3], head[4]
        if kind == "generation":
            self._end_generation(gen, json.loads(line))
            return
        if not self.want_gen(gen) or (kind == "decision" and phase != "morning"):
            return
        if gen != self._gen:
            self._close_day()
            self._gen, self._started, self._state = gen, False, {}
        event = json.loads(line)
        if kind == "population":
            self._start_generation(gen, {a["id"]: AgentInfo(a["id"], a["genome"], tuple(a["parents"]))
                                         for a in event["agents"]},
                                   {a["id"]: Vitals(a["health"], a["stock"]["food"], a["stock"]["wood"])
                                    for a in event["agents"]})
            return
        if (gen, day) != self._day:
            self._close_day()
            self._day = (gen, day)
        if kind == "day_end":
            self._season = event["season"]
            self._close_day()
            return
        self._events.append(event)

    # ---- items --------------------------------------------------------------

    def _start_generation(self, gen: int, agents: dict[int, AgentInfo], state: dict[int, Vitals]) -> None:
        self._started, self._state = True, state
        self.gene_means.setdefault(gen, {g: sum(a.genome[g] for a in agents.values()) / max(1, len(agents))
                                         for g in GENES})
        self._items.append(GenStart(gen, agents, {i: v.copy() for i, v in state.items()}))

    def _close_day(self) -> None:
        if self._day is None:
            return
        (gen, day), events, season = self._day, self._events, self._season
        self._day, self._events, self._season = None, [], None
        if not self._started:
            agents = self._fallback_roster(gen, events)
            a = self.cfg.agent
            self._start_generation(gen, agents, {i: Vitals(a.health_init, a.stock_init[Resource.FOOD],
                                                           a.stock_init[Resource.WOOD]) for i in agents})
        if season is None:
            from simulation import season_for
            season = {r.value: m for r, m in season_for(self.cfg, gen, day).items()}
        item = build_day(gen, day, season, events, self._state, self.cfg)
        self._state = item.end
        self._items.append(item)

    def _end_generation(self, gen: int, event: dict) -> None:
        self.generations_done += 1
        self.gene_means[gen] = {g: event[f"mean_{c}"] for g, c in GENE_COLS.items()}
        if not self.want_gen(gen):
            return
        self._close_day()
        if self._started and self._gen == gen:
            self._items.append(GenEnd(gen, event))

    def _fallback_roster(self, gen: int, events: list[dict]) -> dict[int, AgentInfo]:
        """Genomes from agents.csv, else the persona levels of the day's decisions (level midpoints)."""
        if self._csv is None:
            self._csv = defaultdict(dict)
            path = self.run_dir / "agents.csv"
            if path.exists():
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        i = int(row["agent"])
                        parents = tuple(int(p) for p in row["parents"].split(";") if p)
                        genome = {g: float(row[c]) for g, c in GENE_COLS.items()}
                        self._csv[int(row["gen"])][i] = AgentInfo(i, genome, parents)
        if self._csv.get(gen):
            return self._csv[gen]
        agents = {}
        for e in events:
            if e["type"] == "decision" and e["agent"] not in agents:
                agents[e["agent"]] = AgentInfo(e["agent"], {g: (lvl + 0.5) / 5 for g, lvl in e["persona_levels"].items()})
        return agents


# ---- one day ------------------------------------------------------------------

def build_day(gen: int, day: int, season: dict, events: list[dict], start: dict[int, Vitals], cfg: Config) -> DayLog:
    by_type = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    plans = _plans(start, by_type)
    return DayLog(gen, day, season, start, apply_day(start, plans, by_type, cfg), plans,
                  by_type["gather"], next(iter(by_type["hunt"]), None), by_type["raid"], by_type["trade"],
                  by_type["gift"], by_type["night"], by_type["death"])


def _plans(start: dict[int, Vitals], by_type) -> dict[int, Plan]:
    plans = {i: Plan() for i, v in start.items() if v.alive}
    for e in by_type["decision"]:
        plan = plans.get(e["agent"])
        if plan is None:
            continue
        if e["name"] == "activity":
            plan.activity = e["chosen"]
        elif e["name"] == "route":
            plan.route = e["chosen"]
        elif e["name"] == "target" and e["chosen"] != "nobody":
            plan.target = int(e["chosen"].split("_")[1])
    for e in by_type["raid_aborted"]:
        plans[e["agent"]].activity = "gather_food"
    for e in by_type["gather"]:
        plans[e["agent"]].activity, plans[e["agent"]].route = f"gather_{e['resource']}", e["route"]
    for plan in plans.values():
        if plan.activity != "raid":
            plan.target = None
        if not plan.activity.startswith("gather"):
            plan.route = None
    return plans


def apply_day(start: dict[int, Vitals], plans: dict[int, Plan], by_type, cfg: Config) -> dict[int, Vitals]:
    """The agents' state at the end of the day, replaying the phases of §5 in order."""
    s = {i: v.copy() for i, v in start.items()}
    for e in by_type["gather"]:
        s[e["agent"]].add(e["resource"], e["got"])
        if e["injured"]:
            s[e["agent"]].health -= cfg.gathering.far_injury_dmg
    for i, plan in plans.items():
        if plan.activity == "rest":
            s[i].health = min(cfg.agent.health_max, s[i].health + cfg.agent.rest_regen)
    for e in by_type["hunt"]:
        for i in e["party"]:
            s[i].food += e["payout"]
        for i in e["contributors"]:
            s[i].health -= cfg.hunt.effort_cost
    for e in by_type["raid"]:
        winner, loser = (e["attacker"], e["target"]) if e["attacker_won"] else (e["target"], e["attacker"])
        for r, n in e["loot"].items():
            s[loser].add(r, -n)
            s[winner].add(r, n)
        if e["dmg_to"] is not None:
            s[e["dmg_to"]].health -= cfg.raid.fight_dmg
    for e in by_type["trade"]:
        if e["accepted"]:
            r, o = e["resource"], OTHER[e["resource"]]
            p, q = s[e["proposer"]], s[e["responder"]]
            p.add(o, -e["give"])
            q.add(o, e["give"])
            q.add(r, -e["get"])
            p.add(r, e["get"])
    for e in by_type["gift"]:
        if e["given"]:
            s[e["donor"]].add(e["resource"], -1)
            s[e["recipient"]].add(e["resource"], 1)
    for e in by_type["night"]:
        a = s[e["agent"]]
        a.food -= cfg.agent.need[Resource.FOOD] * e["ate"]
        a.wood -= cfg.agent.need[Resource.WOOD] * e["burned"]
        a.health = e["health_after"]
    for e in by_type["death"]:
        a = s[e["agent"]]
        a.alive, a.food, a.wood = False, 0, 0
    return s
