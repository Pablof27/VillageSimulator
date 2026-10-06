# Calibration E0 (SIMULATION_DESIGN.md §9): render each prompt of the catalogue in a fixed standard situation,
# sweep the relevant gene over persona levels 0-4 (others at level 2) and check that Laya's raw (tau = 1)
# probability of the focal action moves in the expected direction. Decisions that fail are "weak channels".
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from brain import make_brain
from config import Config
from genes import Gene, Genome
from prompts import accept, defend, effort, gift, morning, offer
from resources import Resource
from world import Agent, Day, Hunt, Village

MIN_DELTA = 0.15       # pass: P moves at least this much from the first to the last condition
MARGINAL_DELTA = 0.05  # marginal: a smaller but still monotone effect
MONOTONE_TOL = 0.05    # largest step allowed against the expected direction

FOOD, WOOD = Resource.FOOD, Resource.WOOD
LEVELS = range(5)

Render = Callable[[Config], tuple[str, dict]]


@dataclass
class Check:
    name: str
    question: str
    focal: str
    direction: int                       # +1: P(focal) should rise across conditions, -1: fall
    conditions: list[tuple[str, Render]]  # ordered from the "low" to the "high" end of the sweep
    gene: Gene | None = None             # None for situation contrasts (urgency)


def calibrate(cfg: Config, out_dir: str | Path, brain=None, progress=print) -> dict:
    """Runs every check, prints the table and writes <out_dir>/calibration.json; returns the report."""
    brain = brain or make_brain(cfg.brain, random.Random(cfg.seed))
    results = [_run_check(check, cfg, brain) for check in checks()]
    for line in _table(results):
        progress(line)

    report = {
        "model": cfg.brain.model if cfg.brain.kind == "laya" else cfg.brain.kind,
        "persona_mode": cfg.prompts.persona_mode,
        "criteria": {"min_delta": MIN_DELTA, "marginal_delta": MARGINAL_DELTA, "monotone_tol": MONOTONE_TOL},
        "weak_channels": [r["name"] for r in results if r["status"] == "fail"],
        "checks": results,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def checks() -> list[Check]:
    """The acceptance criteria of E0: one gene sweep per decision plus the urgency contrasts."""
    agg, alt, risk = Gene.AGGRESSION, Gene.ALTRUISM, Gene.RISK
    return [
        _sweep("activity/raid", agg, "activity", "raid", +1, _morning),
        _sweep("route", risk, "route", "far", +1, _morning),
        _sweep("effort", alt, "effort", "contribute", +1, _effort),
        _sweep("defend/aggression", agg, "defend", "fight", +1, _defend),
        _sweep("defend/risk", risk, "defend", "fight", +1, _defend),
        _sweep("offer", alt, "offer", "greedy", -1, _offer),
        _sweep("accept", alt, "accept", "yes", +1, _accept),
        _sweep("gift", alt, "gift", "yes", +1, _gift),
        Check("urgency/wood", "activity", "gather_wood", +1, [
            ("wood=3", lambda cfg: _morning(cfg, _genome(), stock=(3, 3))),
            ("wood=0", lambda cfg: _morning(cfg, _genome(), stock=(3, 0))),
        ]),
        Check("urgency/food", "activity", "gather_food", +1, [
            ("food=3", lambda cfg: _morning(cfg, _genome(), stock=(3, 3))),
            ("food=0", lambda cfg: _morning(cfg, _genome(), stock=(0, 3))),
        ]),
    ]


def evaluate(probabilities: list[float], direction: int) -> tuple[float, bool, str]:
    """(delta in the expected direction, monotone within tolerance, pass | marginal | fail)."""
    # Rounded like Laya's 4-decimal output, so 0.24 - 0.09 counts as 0.15 rather than 0.1499999.
    delta = round(direction * (probabilities[-1] - probabilities[0]), 4)
    steps = [round(direction * (b - a), 4) for a, b in zip(probabilities, probabilities[1:])]
    monotone = all(step >= -MONOTONE_TOL for step in steps)
    if delta >= MIN_DELTA and monotone:
        return delta, monotone, "pass"
    if delta >= MIN_DELTA or (delta >= MARGINAL_DELTA and monotone):
        return delta, monotone, "marginal"
    return delta, monotone, "fail"


# ---- standard situation -----------------------------------------------------
# Day 8 of 20 in a village of 20, normal season, no news, no memory, no reputations.

def _genome(gene: Gene | None = None, level: int = 2) -> Genome:
    genome = {g: 0.5 for g in Gene}  # middle of level 2
    if gene is not None:
        genome[gene] = (level + 0.5) / 5
    return genome


def _village(genome: Genome, stock=(2, 2), others_stock=(3, 3), n=20) -> tuple[Village, Agent, list[Agent]]:
    me = Agent(1, genome, 7, {FOOD: stock[0], WOOD: stock[1]})
    others = [Agent(i, _genome(), 7, {FOOD: others_stock[0], WOOD: others_stock[1]}) for i in range(2, n + 1)]
    yesterday = Day(Hunt(False), {FOOD: 1.0, WOOD: 1.0}, 0)
    return Village([me, *others], 8, {FOOD: 1.0, WOOD: 1.0}, yesterday, []), me, others


def _morning(cfg, genome, stock=(2, 2)):
    v, me, others = _village(genome, stock)
    return morning(cfg, v, me, others[:3])


def _effort(cfg, genome):
    v, me, others = _village(genome)
    return effort(cfg, v, me, [me, *others[:4]], 3)


def _defend(cfg, genome):
    v, me, others = _village(genome)
    return defend(cfg, v, me, others[0])


def _offer(cfg, genome):
    v, me, others = _village(genome, stock=(0, 3), others_stock=(4, 2))
    return offer(cfg, v, me, others[0], FOOD, cfg.market.offers)


def _accept(cfg, genome):
    v, me, others = _village(genome, stock=(4, 2), others_stock=(0, 3))
    return accept(cfg, v, me, others[0], FOOD, "fair", 1, 1)


def _gift(cfg, genome):
    v, me, others = _village(genome, stock=(4, 4), others_stock=(0, 2))
    return gift(cfg, v, me, others[0], FOOD)


# ---- helpers ----------------------------------------------------------------

def _sweep(name, gene, question, focal, direction, render) -> Check:
    conditions = [(f"L{lvl}", lambda cfg, lvl=lvl: render(cfg, _genome(gene, lvl))) for lvl in LEVELS]
    return Check(name, question, focal, direction, conditions, gene)


def _run_check(check: Check, cfg: Config, brain) -> dict:
    measured = []
    for label, render in check.conditions:
        state, questions = render(cfg)
        question = {check.question: questions[check.question]}  # questions are scored independently
        probs = brain.probabilities(state, question)[check.question]
        measured.append({"condition": label, "p": probs[check.focal], "probabilities": probs,
                         "state": state, "question": question[check.question]})
    delta, monotone, status = evaluate([m["p"] for m in measured], check.direction)
    return {"name": check.name, "gene": check.gene.value if check.gene else None, "question": check.question,
            "focal": check.focal, "direction": check.direction, "delta": round(delta, 4),
            "monotone": monotone, "status": status, "conditions": measured}


def _table(results: list[dict]) -> list[str]:
    lines = [f"{'decision':<19}{'gene':<15}{'focal':<18}{'P per condition':<46}{'delta':>7}  {'mono':<5} status",
             "-" * 118]
    for r in results:
        sign = "-" if r["direction"] < 0 else ""
        values = " ".join(f"{c['condition']}={c['p']:.2f}" for c in r["conditions"])
        lines.append(f"{r['name']:<19}{r['gene'] or '-':<15}{sign + 'P(' + r['focal'] + ')':<18}{values:<46}"
                     f"{r['delta']:>+7.2f}  {'yes' if r['monotone'] else 'no':<5} {r['status'].upper()}")
    weak = [r["name"] for r in results if r["status"] == "fail"]
    lines.append(f"weak channels: {', '.join(weak) if weak else 'none'}")
    return lines
