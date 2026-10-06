# Analytic policy catalogue (SIMULATION_DESIGN.md §6.7): the same six functions and outputs as prompts.py, but
# every question carries a probability distribution from a conditional logit over features of the situation:
#     U(o) = sum_k W[o][k] * x_k + sum_j S[j] * y_j(o),    p = softmax(U);    noul: p(yes) = sigmoid(U(yes))
# Features use only what the matching prompt renders, with the thresholds of labels.py (information parity).
import json
import math
from functools import lru_cache
from pathlib import Path

import yaml

from config import Config
from genes import Gene, gather_mult, level
from labels import (
    CROWDED_BELOW, assessment_level, health_level, need_level, recent_events, relation_kind, reputation_kind,
    strength_level, wealth_level,
)
from resources import Resource, other
from world import MemoryEventKind as MEK

FOOD, WOOD = Resource.FOOD, Resource.WOOD
HUNT_KINDS = (MEK.SHIRKED_IN_MY_HUNT, MEK.WORKED_IN_MY_HUNT)
URGENCY = {0: 1.0, 1: 0.5, 2: 0.0}     # need_level -> urgency
HEALTH_LOW = {0: 1.0, 1: 0.5, 2: 0.0}  # health_level -> health_low
OFFER_VALUE = {"generous": -1, "fair": 0, "greedy": 1}  # get - give, from the proposer's view
DEAL = {"generous": 1, "fair": 0, "greedy": -1}         # the same offer from the responder's view

# Feature names per decision. Gene terms follow the relevance matrix of §4.4.
RELATION = ("friend", "enemy", "refused")
REPUTATION = ("raider", "generous")
MORNING = ("bias", "urgency_food", "urgency_wood", "urgency_max", "health_low", "yield_food", "yield_wood",
           "crowded_food", "crowded_wood", "hunt_prev", "z_agg", "z_risk")
CANDIDATE = ("rich", "poor", "strong", "weak", *RELATION, *REPUTATION)
EFFORT = ("bias", "z_alt", "slack", "hunt_prev", "shirker_frac", "reliable_frac", "hunt_memory", "health_low")
DEFEND = ("bias", "assess", "z_agg", "z_risk", "stake", "health_low", *RELATION)  # the defend prompt shows no reputation
OFFER = ("bias", "z_alt", *RELATION, *REPUTATION)
OFFER_OPTION = ("value", "value_x_alt", "value_x_refused", "value_x_friend", "value_x_raider")
ACCEPT = ("bias", "deal", "z_alt", *RELATION, *REPUTATION)
GIFT = ("bias", "z_alt", "surplus", *RELATION, *REPUTATION)

# decision -> weight block -> allowed features. Options without a block have U = 0 (the reference option).
# `candidate` and `each` are shared weights over option features; every other block weighs context features.
SCHEMA = {
    "activity": {o: MORNING for o in ("gather_food", "gather_wood", "hunt", "raid", "rest")},
    "route": {"near": MORNING, "far": MORNING},
    "target": {"candidate": CANDIDATE, "nobody": MORNING},
    "effort": {"contribute": EFFORT, "free_ride": EFFORT},
    "defend": {"fight": DEFEND, "yield": DEFEND},
    "offer": {"each": OFFER_OPTION, **{o: OFFER for o in OFFER_VALUE}},
    "accept": {"yes": ACCEPT},
    "gift": {"yes": GIFT},
}

ACTIVITY_TEXT = {"gather_food": "Forage alone.", "gather_wood": "Cut wood alone.", "hunt": "Join the group hunt.",
                 "raid": "Rob another villager.", "rest": "Stay home and recover."}
_YAML_BOOL = {True: "yes", False: "no"}


# ---- catalogue (same signatures as prompts.py) ------------------------------

def morning(cfg: Config, village, agent, candidates: list) -> tuple[str, dict]:
    """Questions `activity`, `route` and `target`; `raid`/`target` omitted if `candidates` is empty."""
    w, x = _weights(cfg), features_morning(cfg, village, agent)
    labels = [o for o in ACTIVITY_TEXT if o != "raid" or candidates]
    questions = {
        "activity": _choice("Choose what to do today.", w["activity"], x, {o: ACTIVITY_TEXT[o] for o in labels}),
        "route": _choice("Where do you gather?", w["route"], x, {"near": "Nearby, safe.", "far": "Far, riskier."}),
    }
    if candidates:
        options = {f"agent_{c.id}": candidate_features(cfg, village, agent, c) for c in candidates}
        criteria = {f"agent_{c.id}": f"Rob Agent {c.id}." for c in candidates}
        criteria["nobody"] = "Change your mind and forage instead."
        questions["target"] = _choice("Whom do you rob today?", w["target"], x, criteria, options, shared="candidate")
    return _state("morning", agent, x), questions


def effort(cfg: Config, village, agent, party, threshold) -> tuple[str, dict]:
    x = features_effort(cfg, village, agent, party, threshold)
    question = _choice("Choose your effort.", _weights(cfg)["effort"], x,
                       {"contribute": "Work hard.", "free_ride": "Hold back."})
    return _state("effort", agent, x), {"effort": question}


def defend(cfg: Config, village, agent, attacker) -> tuple[str, dict]:
    """`agent` is the target of the raid."""
    x = features_defend(cfg, village, agent, attacker)
    question = _choice("What do you do?", _weights(cfg)["defend"], x,
                       {"fight": "Fight back.", "yield": "Hand over half of your stock."})
    return _state("defend", agent, x), {"defend": question}


def offer(cfg: Config, village, proposer, partner, r, offers) -> tuple[str, dict]:
    """`offers` maps affordable offer names to (give other(r), get r)."""
    x = features_offer(cfg, village, proposer, partner)
    options = {name: offer_features(x, name) for name in offers}
    criteria = {name: f"Offer {give} {other(r).value} for {get} {r.value}." for name, (give, get) in offers.items()}
    question = _choice(f"What do you offer Agent {partner.id}?", _weights(cfg)["offer"], x, criteria, options,
                       shared="each")
    return _state("offer", proposer, x), {"offer": question}


def accept(cfg: Config, village, responder, proposer, r, offer, give, get) -> tuple[str, dict]:
    x = features_accept(cfg, village, responder, proposer, offer)
    return _state("accept", responder, x), {"accept": _noul("Do you accept the trade?", _weights(cfg)["accept"], x)}


def gift(cfg: Config, village, donor, recipient, r) -> tuple[str, dict]:
    x = features_gift(cfg, village, donor, recipient, r)
    question = _noul(f"Do you give Agent {recipient.id} one {r.value}?", _weights(cfg)["gift"], x)
    return _state("gift", donor, x), {"gift": question}


# ---- features (§6.7.3) ------------------------------------------------------

def features_morning(cfg: Config, village, agent) -> dict[str, float]:
    urgency = {r: URGENCY[need_level(agent.stock[r])] for r in Resource}
    yesterday = village.yesterday
    return {
        "bias": 1.0,
        "urgency_food": urgency[FOOD],
        "urgency_wood": urgency[WOOD],
        "urgency_max": max(urgency.values()),
        "health_low": HEALTH_LOW[health_level(agent.health)],
        # E_r / base_yield - 1: the expected yield the gather options state, relative to a generalist's.
        "yield_food": gather_mult(agent.genome, FOOD) * village.season[FOOD] - 1,
        "yield_wood": gather_mult(agent.genome, WOOD) * village.season[WOOD] - 1,
        "crowded_food": float(yesterday.congestion[FOOD] < CROWDED_BELOW),
        "crowded_wood": float(yesterday.congestion[WOOD] < CROWDED_BELOW),
        "hunt_prev": _hunt_prev(yesterday.hunt),
        "z_agg": _z(cfg, agent, Gene.AGGRESSION),
        "z_risk": _z(cfg, agent, Gene.RISK),
    }


def candidate_features(cfg: Config, village, agent, candidate) -> dict[str, float]:
    wealth, strength = wealth_level(candidate.wealth), strength_level(candidate.health)
    return {
        "rich": float(wealth == 1), "poor": float(wealth == -1),
        "strong": float(strength == 1), "weak": float(strength == -1),
        **_relation(cfg, village, agent, candidate.id),
        **_reputation(cfg, candidate),
    }


def features_effort(cfg: Config, village, agent, party, threshold) -> dict[str, float]:
    mates = [a for a in party if a.id != agent.id]
    th = cfg.memory.rep_threshold
    # Same labels as the effort prompt (not reputation_kind): shirker, or reliable without being a shirker.
    shirkers = sum(a.record.free_rode >= th for a in mates)
    reliable = sum(a.record.contributed >= th and a.record.free_rode < th for a in mates)
    shown = cfg.prompts.render_reputation and mates
    return {
        "bias": 1.0,
        "z_alt": _z(cfg, agent, Gene.ALTRUISM),
        "slack": (len(party) - threshold) / len(party),
        "hunt_prev": _hunt_prev(village.yesterday.hunt),
        "shirker_frac": shirkers / len(mates) if shown else 0.0,
        "reliable_frac": reliable / len(mates) if shown else 0.0,
        "hunt_memory": _hunt_memory(cfg, village, agent),
        "health_low": HEALTH_LOW[health_level(agent.health)],
    }


def features_defend(cfg: Config, village, agent, attacker) -> dict[str, float]:
    return {
        "bias": 1.0,
        "assess": float(assessment_level(agent.health, attacker.health)),
        "z_agg": _z(cfg, agent, Gene.AGGRESSION),
        "z_risk": _z(cfg, agent, Gene.RISK),
        "stake": min(1.0, agent.wealth / cfg.evolution.wealth_ref),
        "health_low": HEALTH_LOW[health_level(agent.health)],
        **_relation(cfg, village, agent, attacker.id),
    }


def features_offer(cfg: Config, village, proposer, partner) -> dict[str, float]:
    return {"bias": 1.0, "z_alt": _z(cfg, proposer, Gene.ALTRUISM),
            **_relation(cfg, village, proposer, partner.id), **_reputation(cfg, partner)}


def offer_features(context: dict[str, float], name: str) -> dict[str, float]:
    """Option features of one offer: its value to the proposer, alone and interacted with persona and history."""
    value = OFFER_VALUE[name]
    return {"value": value, "value_x_alt": value * context["z_alt"], "value_x_refused": value * context["refused"],
            "value_x_friend": value * context["friend"], "value_x_raider": value * context["raider"]}


def features_accept(cfg: Config, village, responder, proposer, offer_name: str) -> dict[str, float]:
    return {"bias": 1.0, "deal": float(DEAL[offer_name]), "z_alt": _z(cfg, responder, Gene.ALTRUISM),
            **_relation(cfg, village, responder, proposer.id), **_reputation(cfg, proposer)}


def features_gift(cfg: Config, village, donor, recipient, r) -> dict[str, float]:
    spare_min = cfg.gifts.spare_min
    return {"bias": 1.0, "z_alt": _z(cfg, donor, Gene.ALTRUISM),
            "surplus": min(1.0, max(0.0, (donor.stock[r] - spare_min) / spare_min)),
            **_relation(cfg, village, donor, recipient.id), **_reputation(cfg, recipient)}


# ---- conditional logit (§6.7.2) ---------------------------------------------

def _choice(instructions: str, weights: dict, context: dict[str, float], criteria: dict[str, str],
            option_features: dict[str, dict[str, float]] | None = None, shared: str | None = None) -> dict:
    terms = {}
    for label in criteria:
        t = {f: w * context[f] for f, w in weights.get(label, {}).items()}
        if option_features and label in option_features:
            for f, w in weights.get(shared, {}).items():
                t[f] = t.get(f, 0.0) + w * option_features[label][f]
        terms[label] = {f: v for f, v in t.items() if v}
    utilities = {label: sum(t.values()) for label, t in terms.items()}
    top = max(utilities.values())  # subtracting the max avoids overflow and does not change the softmax
    exp = {label: math.exp(u - top) for label, u in utilities.items()}
    total = sum(exp.values())
    question = {"type": "choice", "instructions": instructions, "criteria": criteria,
                "probabilities": {label: e / total for label, e in exp.items()},
                "utilities": utilities, "terms": terms}
    if option_features:
        question["option_features"] = option_features
    return question


def _noul(instructions: str, weights: dict, context: dict[str, float]) -> dict:
    terms = {f: w * context[f] for f, w in weights.get("yes", {}).items() if w * context[f]}
    u = sum(terms.values())
    p_yes = 1 / (1 + math.exp(-u)) if u >= 0 else math.exp(u) / (1 + math.exp(u))
    return {"type": "noul", "instructions": instructions,
            "probabilities": {"yes": p_yes, "no": 1 - p_yes},
            "utilities": {"yes": u, "no": 0.0}, "terms": {"yes": terms}}


# ---- weights ----------------------------------------------------------------

def load_weights(path: str | Path) -> dict:
    """Reads and validates a weights file (unknown decisions, blocks or features raise)."""
    return _load_weights(str(path))


@lru_cache
def _load_weights(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    encoding = data.pop("gene_encoding", "continuous")
    _check(encoding in ("continuous", "level"), f"{path}: gene_encoding must be continuous or level")
    _check(not set(data) - set(SCHEMA), f"{path}: unknown decisions {sorted(set(data) - set(SCHEMA))}")
    weights = {"gene_encoding": encoding}
    for decision, schema in SCHEMA.items():
        # YAML 1.1 reads the keys `yes:` / `no:` as booleans.
        blocks = {_YAML_BOOL.get(k, k) if isinstance(k, bool) else k: v for k, v in (data.get(decision) or {}).items()}
        _check(not set(blocks) - set(schema), f"{path}: unknown {decision} blocks {sorted(set(blocks) - set(schema))}")
        for block, features in blocks.items():
            features = features or {}
            unknown = set(features) - set(schema[block])
            _check(not unknown, f"{path}: {decision}.{block} has unknown features {sorted(unknown)}")
            _check(all(isinstance(v, (int, float)) and math.isfinite(v) for v in features.values()),
                   f"{path}: {decision}.{block} weights must be finite numbers")
            blocks[block] = {f: float(v) for f, v in features.items()}
        weights[decision] = blocks
    return weights


# ---- helpers ----------------------------------------------------------------

def _weights(cfg: Config) -> dict:
    return load_weights(cfg.brain.policy_weights)


def _state(decision: str, agent, features: dict[str, float]) -> str:
    return json.dumps({"engine": "policy", "decision": decision, "agent": agent.id, "features": features})


def _z(cfg: Config, agent, gene: Gene) -> float:
    """Gene input in [-1, 1]; 0 when the persona is not rendered (E7)."""
    if cfg.prompts.persona_mode == "none":
        return 0.0
    g = agent.genome[gene]
    return (level(g) - 2) / 2 if _weights(cfg)["gene_encoding"] == "level" else 2 * g - 1


def _hunt_prev(hunt) -> float:
    if hunt is None or not hunt.took_place:
        return 0.0
    return 1.0 if hunt.success else -1.0


def _hunt_memory(cfg: Config, village, agent) -> float:
    """(worked - shirked) / memory_render over the hunt memories the effort prompt renders."""
    render = cfg.memory.render
    if not cfg.prompts.render_memory or render == 0:
        return 0.0
    events = recent_events(agent.memory, village.day_count, cfg.memory.days, HUNT_KINDS)[:render]
    return sum(1 if e.kind == MEK.WORKED_IN_MY_HUNT else -1 for e in events) / render


def _relation(cfg: Config, village, agent, other_id: int) -> dict[str, float]:
    kind = relation_kind(agent.memory, other_id, village.day_count, cfg.memory.days) \
        if cfg.prompts.render_memory else None
    return {"friend": float(kind == "friend"), "enemy": float(kind in ("enemy", "refused")),
            "refused": float(kind == "refused")}


def _reputation(cfg: Config, other_agent) -> dict[str, float]:
    kind = reputation_kind(other_agent.record, cfg.memory.rep_threshold) if cfg.prompts.render_reputation else None
    return {"raider": float(kind == "raider"), "generous": float(kind == "generous")}


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)
