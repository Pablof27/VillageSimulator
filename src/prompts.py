# Prompt catalogue (SIMULATION_DESIGN.md §6.4): state = persona + core + situation.

from math import floor
from config import Config
from resources import Resource, other
from world import MemoryEventKind as MEK
from genes import Gene, gather_mult, persona
from labels import (
    CROWDED_BELOW, assessment, cap, event_sentence, health_status, need_status, recent_events,
    relation_sentence, reputation_label, strength_label, wealth_label,
)

HUNT_KINDS = (MEK.SHIRKED_IN_MY_HUNT, MEK.WORKED_IN_MY_HUNT)

OFFER_NOTES = {
    "generous": "Kind to Agent {q}; almost surely accepted. You would keep {left} {o}.",
    "fair": "An even deal; probably accepted. You would keep {left} {o}.",
    "greedy": "Hard on Agent {q}; may be refused. You would keep {left} {o} and have a spare {r}.",
}
ACCEPT_VERDICTS = {
    "generous": "This deal is good for you.",
    "fair": "This deal is fair.",
    "greedy": "This deal is poor for you but you would still have enough for tonight.",
}


# ---- core sentences --------------------------------------------------

def needs_sentence(stock: dict, need: dict) -> str:
    lacking = [r for r in Resource if stock[r] < need[r]]
    if not lacking:
        return "You have enough for tonight."
    return f"You lack {' and '.join(r.value for r in lacking)} for tonight."


def season_word(mult: float) -> str:
    return "scarce" if mult < 1 else "plentiful" if mult > 1 else "normal"


def season_sentence(season: dict) -> str:
    return f"Food is {season_word(season[Resource.FOOD])} and wood is {season_word(season[Resource.WOOD])} this season."


def core_sentence(cfg: Config, village, agent) -> str:
    return (
        f"You live in a village of {village.n_alive} people, day {village.day_count} of {cfg.population.D}. "
        f"Health {agent.health}/{cfg.agent.health_max}. "
        f"You have {agent.stock[Resource.FOOD]} food and {agent.stock[Resource.WOOD]} wood; tonight you need 1 of each. "
        f"{needs_sentence(agent.stock, cfg.agent.need)} {season_sentence(village.season)}"
    )


# ---- prompt catalogue ------------------------------------------------

def morning(cfg: Config, village, agent, candidates: list) -> tuple[str, dict]:
    """Questions `activity`, `route` and `target` (§6.4.1); `raid`/`target` omitted if `candidates` is empty."""
    y = village.yesterday
    situation = [
        _congestion_sentence(y.congestion),
        _hunt_sentence(y.hunt),
        _raids_sentence(y.raids),
        *_memories(cfg, village, agent),
    ]
    state = _state(cfg, village, agent, (Gene.AGGRESSION, Gene.RISK, Gene.SPECIALIZATION), situation)

    expected = {r: cfg.gathering.base_yield * gather_mult(agent.genome, r) * village.season[r] for r in Resource}
    activity = {
        "gather_food": f"Forage alone. Expect about {_num(expected[Resource.FOOD])} food. Safe. "
                       f"{need_status(agent.stock[Resource.FOOD], Resource.FOOD.value)}",
        "gather_wood": f"Cut wood alone. Expect about {_num(expected[Resource.WOOD])} wood. Safe. "
                       f"{need_status(agent.stock[Resource.WOOD], Resource.WOOD.value)}",
        "hunt": f"Join the group hunt. If enough hunters work hard, every hunter gets {cfg.hunt.payout} food. "
                "If it fails you get nothing today.",
    }
    if candidates:
        activity["raid"] = ("Rob another villager and take half of their stock. "
                            "They may fight back and hurt you, and everyone will know you did it.")
    activity["rest"] = (f"Stay home and recover {cfg.agent.rest_regen} health. You collect nothing. "
                        f"{health_status(agent.health)}").strip()

    questions = {
        "activity": {"type": "choice", "instructions": "Choose what to do today.", "criteria": activity},
        "route": {"type": "choice", "instructions": "Where do you gather?", "criteria": {
            "near": "Nearby. Safe. Normal yield.",
            "far": f"Far away. Half again as much yield, but a 1 in {round(1 / cfg.gathering.far_injury_p)} chance "
                   f"of an injury costing {cfg.gathering.far_injury_dmg} health.",
        }},
    }
    if candidates:
        targets = {}
        for c in candidates:
            parts = [f"{cap(wealth_label(c.wealth))} and {strength_label(c.health)}."]
            rep = _reputation(cfg, c)
            if rep:
                parts.append(f"{cap(rep)}.")
            rel = _relation(cfg, village, agent, c.id, named=False)
            if rel:
                parts.append(rel)
            targets[f"agent_{c.id}"] = " ".join(parts)
        targets["nobody"] = "Change your mind and forage instead."
        questions["target"] = {"type": "choice", "instructions": "Whom do you rob today?", "criteria": targets}
    return state, questions


def effort(cfg: Config, village, agent, party, threshold) -> tuple[str, dict]:
    """Question `effort` (§6.4.2)."""
    m = len(party)
    mates = [a for a in party if a.id != agent.id]
    situation = [f"You are in a hunting party of {m}: {_agent_list([a.id for a in mates])}."]
    if cfg.prompts.render_reputation:
        th = cfg.memory.rep_threshold
        shirkers = [a.id for a in mates if a.record.free_rode >= th]
        reliable = [a.id for a in mates if a.record.contributed >= th and a.record.free_rode < th]
        if shirkers:
            verb = "is a known shirker" if len(shirkers) == 1 else "are known shirkers"
            situation.append(f"{_agent_list(shirkers)} {verb}.")
        if reliable:
            verb = "is a reliable hunter" if len(reliable) == 1 else "are reliable hunters"
            situation.append(f"{_agent_list(reliable)} {verb}.")
    situation.append(_hunt_sentence(village.yesterday.hunt))
    situation += _memories(cfg, village, agent, HUNT_KINDS)
    state = _state(cfg, village, agent, (Gene.ALTRUISM,), situation)

    payout = min(cfg.hunt.payout, floor(cfg.hunt.supply_per_capita * village.n_alive / m))
    questions = {"effort": {
        "type": "choice",
        "instructions": f"The hunt succeeds only if at least {threshold} of the {m} hunters work hard. "
                        "Choose your effort.",
        "criteria": {
            "contribute": f"Work hard. Costs you {cfg.hunt.effort_cost} health. "
                          "Makes the hunt more likely to succeed for everyone.",
            "free_ride": f"Hold back. Costs nothing. If the others succeed you still get the same {payout} food.",
        },
    }}
    return state, questions


def defend(cfg: Config, village, agent, attacker) -> tuple[str, dict]:
    """Question `defend` (§6.4.3); `agent` is the target of the raid."""
    verdict, fight_note, yield_note = assessment(agent.health, attacker.health)
    situation = [f"Agent {attacker.id} is trying to rob you. {verdict}",
                 _relation(cfg, village, agent, attacker.id, named=True)]
    state = _state(cfg, village, agent, (Gene.AGGRESSION, Gene.RISK), [s for s in situation if s])

    questions = {"defend": {"type": "choice", "instructions": "What do you do?", "criteria": {
        "fight": f"Fight back. {fight_note} The loser loses half of their stock and {cfg.raid.fight_dmg} health.",
        "yield": f"Hand over half of your stock and stay unharmed. {yield_note}",
    }}}
    return state, questions


def offer(cfg: Config, village, proposer, partner, r, offers) -> tuple[str, dict]:
    """Question `offer` (§6.4.4); `offers` maps affordable offer names to (give other(r), get r)."""
    o, q = other(r), partner.id
    rep = _reputation(cfg, partner)
    situation = [
        f"You have no {r.value} for tonight and spare {o.value}. Agent {q} has spare {r.value}" + (f" and is {rep}." if rep else "."),
        _relation(cfg, village, proposer, q, named=True),
    ]
    state = _state(cfg, village, proposer, (Gene.ALTRUISM,), [s for s in situation if s])

    criteria = {
        name: f"Offer {give} {o.value} for {get} {r.value}. "
              + OFFER_NOTES[name].format(q=q, left=proposer.stock[o] - give, o=o.value, r=r.value)
        for name, (give, get) in offers.items()
    }
    questions = {"offer": {
        "type": "choice",
        "instructions": f"What do you offer Agent {q}? A poor offer may be refused.",
        "criteria": criteria,
    }}
    return state, questions


def accept(cfg: Config, village, responder, proposer, r, offer, give, get) -> tuple[str, dict]:
    """Question `accept` (§6.4.5, noul); `r` is the resource the proposer lacks (Resource enum)."""
    p = proposer.id
    rep = _reputation(cfg, proposer)
    situation = [
        f"Agent {p} has no {r.value} for tonight and offers you {give} {other(r).value} for {get} {r.value}. {ACCEPT_VERDICTS[offer]}",
        f"Agent {p} is {rep}." if rep else "",
        _relation(cfg, village, responder, p, named=True),
    ]
    state = _state(cfg, village, responder, (Gene.ALTRUISM,), [s for s in situation if s])
    return state, {"accept": {"type": "noul", "instructions": "Do you accept the trade?"}}


def gift(cfg: Config, village, donor, recipient, r) -> tuple[str, dict]:
    """Question `gift` (§6.4.6, noul)."""
    n = recipient.id
    rep = _reputation(cfg, recipient)
    situation = [
        f"Agent {n} has no {r.value} for tonight and will lose health. You have more {r.value} than you need.",
        f"Agent {n} is {rep}." if rep else "",
        _relation(cfg, village, donor, n, named=True),
    ]
    state = _state(cfg, village, donor, (Gene.ALTRUISM,), [s for s in situation if s])
    return state, {"gift": {"type": "noul", "instructions": f"Do you give Agent {n} one {r.value}?"}}


# ---- helpers ----------------------------------------------------------------

def _num(x: float) -> str:
    return f"{round(x, 1):g}"


def _agent_list(ids: list[int]) -> str:
    if len(ids) == 1:
        return f"Agent {ids[0]}"
    return f"Agents {', '.join(map(str, ids[:-1]))} and {ids[-1]}"


def _state(cfg: Config, village, agent, genes: tuple[Gene, ...], situation: list[str]) -> str:
    parts = [persona(agent.genome, genes, cfg.prompts.persona_mode), core_sentence(cfg, village, agent), *situation]
    return " ".join(p for p in parts if p)


def _reputation(cfg: Config, other_agent) -> str | None:
    if not cfg.prompts.render_reputation:
        return None
    return reputation_label(other_agent.record, cfg.memory.rep_threshold)


def _relation(cfg: Config, village, agent, other_id: int, named: bool) -> str | None:
    if not cfg.prompts.render_memory:
        return None
    return relation_sentence(agent.memory, other_id, village.day_count, cfg.memory.days, named)


def _memories(cfg: Config, village, agent, kinds=None) -> list[str]:
    if not cfg.prompts.render_memory:
        return []
    events = recent_events(agent.memory, village.day_count, cfg.memory.days, kinds)
    return [event_sentence(ev) for ev in reversed(events[:cfg.memory.render])]


def _hunt_sentence(hunt) -> str:
    if hunt is None or not hunt.took_place:
        return "There was no hunt yesterday."
    if hunt.success:
        return f"Yesterday's hunt succeeded with {hunt.contributors} of {hunt.party_size} hunters working hard."
    return f"Yesterday's hunt failed because only {hunt.contributors} of {hunt.party_size} hunters worked hard."


def _congestion_sentence(congestion: dict) -> str:
    places = {Resource.FOOD: "fields", Resource.WOOD: "forest"}
    crowded = [r for r in Resource if congestion[r] < CROWDED_BELOW]
    if not crowded:
        return ""
    verb = "was" if crowded == [Resource.WOOD] else "were"
    return f"Yesterday the {' and the '.join(places[r] for r in crowded)} {verb} crowded and yields were low."


def _raids_sentence(n: int) -> str:
    if n < 1:
        return ""
    return "Yesterday there was 1 raid in the village." if n == 1 else f"Yesterday there were {n} raids in the village."

