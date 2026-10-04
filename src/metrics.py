# Per-generation metrics (SIMULATION_DESIGN.md §8.2) and emergence indicators (§8.4),
# computed from one generation's logged events plus the agents' final state.
from collections import Counter, defaultdict

import numpy as np

from genes import Gene, level

GENE_COLS = {Gene.AGGRESSION: "agg", Gene.ALTRUISM: "alt", Gene.RISK: "risk", Gene.SPECIALIZATION: "spec"}
DEATH_COLS = {"starvation": "deaths_starvation", "injury": "deaths_injury",
              "fight": "deaths_fight", "hunt_exhaustion": "deaths_hunt"}
OFFERS = ("generous", "fair", "greedy")
# (gene column, personal-rate column) pairs whose Spearman correlation validates gene expression.
CORRELATIONS = {"corr_agg_raid": ("agg", "raid_rate"), "corr_alt_gift": ("alt", "gift_rate"),
                "corr_alt_contrib": ("alt", "contrib_rate"), "corr_risk_far": ("risk", "far_rate")}
NAN = float("nan")


def generation_metrics(gen: int, agents, events: list[dict], cfg, wall_seconds: float) -> tuple[dict, list[dict]]:
    """Returns (generations.csv row, agents.csv rows)."""
    by_type = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    agent_rows = agent_metrics(gen, agents, by_type)
    agent_days = sum(a.days_alive for a in agents)
    survivors = [a for a in agents if a.alive]

    row = {"gen": gen}
    row |= _gene_stats(agents)
    row |= _population_stats(survivors, by_type["death"])
    row |= _raid_stats(by_type["raid"], agent_days)
    row |= _hunt_stats(by_type["hunt"], agent_days)
    row |= _trade_stats(by_type["trade"], agent_days)
    row |= _gift_stats(by_type["gift"])
    row |= {name: spearman([r[g] for r in agent_rows], [r[b] for r in agent_rows])
            for name, (g, b) in CORRELATIONS.items()}
    row |= _engine_stats(by_type["decision"], wall_seconds)
    return {k: _round(v) for k, v in row.items()}, [{k: _round(v) for k, v in r.items()} for r in agent_rows]


def agent_metrics(gen: int, agents, by_type: dict) -> list[dict]:
    counts = defaultdict(Counter)
    for e in by_type["raid"]:
        counts[e["attacker"]]["raids"] += 1
    for e in by_type["gift"]:
        counts[e["donor"]]["gift_requests"] += 1
        counts[e["donor"]]["gifts"] += e["given"]
    for e in _held_hunts(by_type["hunt"]):
        for i in e["party"]:
            counts[i]["hunts"] += 1
        for i in e["contributors"]:
            counts[i]["contributions"] += 1
    for e in by_type["gather"]:
        counts[e["agent"]]["gathers"] += 1
        counts[e["agent"]]["far"] += e["route"] == "far"
    for e in by_type["trade"]:
        counts[e["proposer"]]["trades_proposed"] += 1

    rows = []
    for a in agents:
        c = counts[a.id]
        rows.append({
            "gen": gen, "agent": a.id,
            **{col: a.genome[g] for g, col in GENE_COLS.items()},
            "alive": int(a.alive), "days_alive": a.days_alive, "health_end": a.health, "wealth_end": a.wealth,
            "raids": c["raids"], "raid_rate": _ratio(c["raids"], a.days_alive),
            "gift_requests": c["gift_requests"], "gifts": c["gifts"], "gift_rate": _ratio(c["gifts"], c["gift_requests"]),
            "hunts": c["hunts"], "contributions": c["contributions"],
            "contrib_rate": _ratio(c["contributions"], c["hunts"]),
            "gathers": c["gathers"], "far": c["far"], "far_rate": _ratio(c["far"], c["gathers"]),
            "trades_proposed": c["trades_proposed"],
        })
    return rows


# ---- sections of the generation row -----------------------------------------

def _gene_stats(agents) -> dict:
    out = {}
    for g, col in GENE_COLS.items():
        values = np.array([a.genome[g] for a in agents])
        out[f"mean_{col}"], out[f"std_{col}"] = values.mean(), values.std()
    for g in (Gene.AGGRESSION, Gene.ALTRUISM, Gene.RISK):
        out[f"frac_L4_{GENE_COLS[g]}"] = np.mean([level(a.genome[g]) == 4 for a in agents])
    spec = np.array([a.genome[Gene.SPECIALIZATION] for a in agents])
    out["frac_spec_extreme"] = np.mean((spec < 0.2) | (spec > 0.8))
    return out


def _population_stats(survivors, deaths: list[dict]) -> dict:
    causes = Counter(e["cause"] for e in deaths)
    wealth = [a.wealth for a in survivors]
    return {
        "survivors": len(survivors),
        **{col: causes[cause] for cause, col in DEATH_COLS.items()},
        "gini_wealth": gini(wealth),
        "mean_health_end": _mean([a.health for a in survivors]),
        "mean_wealth_end": _mean(wealth),
    }


def _raid_stats(raids: list[dict], agent_days: int) -> dict:
    gained = [e["attacker_won"] and sum(e["loot"].values()) > 0 for e in raids]
    return {"raid_rate": _ratio(len(raids), agent_days), "raid_success": _mean(gained)}


def _hunt_stats(hunts: list[dict], agent_days: int) -> dict:
    held = _held_hunts(hunts)
    with_shirker = [e for e in held if e["known_shirkers"]]
    without_shirker = [e for e in held if not e["known_shirkers"]]
    return {
        "hunt_participation": _ratio(sum(len(e["party"]) for e in hunts), agent_days),
        "contribution_rate": _contribution_rate(held),
        "hunt_success_rate": _mean([e["success"] for e in held]),
        # §8.4 reputation-based cooperation: does a known shirker in the party lower contribution?
        "contrib_rate_with_shirker": _contribution_rate(with_shirker),
        "contrib_rate_without_shirker": _contribution_rate(without_shirker),
    }


def _trade_stats(trades: list[dict], agent_days: int) -> dict:
    offers = Counter(e["offer"] for e in trades)
    return {
        "trade_rate": _ratio(len(trades), agent_days),
        "accept_rate": _mean([e["accepted"] for e in trades]),
        **{f"offer_{name}": _ratio(offers[name], len(trades)) for name in OFFERS},
    }


def _gift_stats(gifts: list[dict]) -> dict:
    return {
        "gift_rate": _mean([e["given"] for e in gifts]),
        # §8.4 reciprocity: P(give | the recipient gave to me recently) vs otherwise.
        "gift_rate_reciprocal": _mean([e["given"] for e in gifts if e["reciprocal"]]),
        "gift_rate_nonreciprocal": _mean([e["given"] for e in gifts if not e["reciprocal"]]),
    }


def _engine_stats(decisions: list[dict], wall_seconds: float) -> dict:
    return {
        "laya_calls": len({e["call"] for e in decisions if not e["forced"]}),
        "forced_moves": sum(e["forced"] for e in decisions),
        "wall_seconds": wall_seconds,
    }


# ---- statistics -------------------------------------------------------------

def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return NAN if len(x) == 0 else 0.0
    ranks = np.arange(1, len(x) + 1)
    return float(2 * (ranks * x).sum() / (len(x) * x.sum()) - (len(x) + 1) / len(x))


def spearman(x, y) -> float:
    """Spearman rank correlation over the pairs where both values are finite; NaN if undefined."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return NAN
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return NAN
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for v in np.unique(values):
        tied = values == v
        ranks[tied] = ranks[tied].mean()
    return ranks


def _held_hunts(hunts: list[dict]) -> list[dict]:
    return [e for e in hunts if not e["cancelled"]]


def _contribution_rate(held: list[dict]) -> float:
    return _ratio(sum(len(e["contributors"]) for e in held), sum(len(e["party"]) for e in held))


def _ratio(num, den) -> float:
    return num / den if den else NAN


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else NAN


def _round(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value
