# Prompt catalogue (SIMULATION_DESIGN.md §6.4): state = persona + core + situation.

from math import floor
from resources import Resource, other
from world import MemoryEventKind as MEK
from genes import Gene, level, gather_mult

PERSONA = {
    Gene.AGGRESSION: (
        "You are a gentle farmer who abhors violence and never takes what is not yours.",
        "You are a quiet villager who dislikes conflict.",
        "You are an ordinary villager who takes care of your own.",
        "You are a tough brawler who takes what you need.",
        "You are a feared raider who lives by taking from others.",
    ),
    Gene.ALTRUISM: (
        "You are a miser who never shares anything with anyone.",
        "You are a self-reliant villager who seldom helps others.",
        "You are an ordinary villager who helps when it is convenient.",
        "You are a kind neighbour who likes to help others.",
        "You are a devoted helper who always shares with those in need.",
    ),
    Gene.RISK: (
        "You are extremely cautious and avoid every risk.",
        "You prefer safe, predictable choices.",
        "You take moderate risks when the reward is worth it.",
        "You are bold and enjoy taking chances.",
        "You are a reckless gambler who loves danger.",
    ),
    Gene.SPECIALIZATION: (
        "You are a master woodcutter but a poor forager.",
        "You are a good woodcutter and a mediocre forager.",
        "You are equally able at foraging and woodcutting.",
        "You are a good forager and a mediocre woodcutter.",
        "You are a master forager but a poor woodcutter.",
    ),
}

EVENT_TEMPLATES = {
    MEK.ROBBED_ME: "{O} robbed you on day {d}.",
    MEK.I_ROBBED: "You robbed {o} on day {d}.",
    MEK.GAVE_ME: "{O} gave you {r} on day {d}.",
    MEK.I_GAVE: "You gave {o} {r} on day {d}.",
    MEK.TRADED_FAIR: "{O} traded fairly with you on day {d}.",
    MEK.TRADED_GENEROUS_TO_ME: "{O} traded generously with you on day {d}.",
    MEK.TRADED_GREEDY_TO_ME: "{O} drove a hard bargain with you on day {d}.",
    MEK.REFUSED_ME: "{O} refused your offer on day {d}.",
    MEK.I_REFUSED: "You refused {os} offer on day {d}.",
    MEK.SHIRKED_IN_MY_HUNT: "{O} held back in your hunt on day {d}.",
    MEK.WORKED_IN_MY_HUNT: "{O} worked hard in your hunt on day {d}.",
}
NEGATIVE_KINDS = (MEK.ROBBED_ME, MEK.REFUSED_ME, MEK.SHIRKED_IN_MY_HUNT)
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


# ---- genes ------------------------------------------------------------------

def persona(genome, genes: tuple[Gene, ...], mode: str = "gated") -> str:
    if mode == "none":
        return ""
    if mode == "full":
        genes = Gene.asdict()
    return " ".join(PERSONA[g][level(genome[g])] for g in genes)


# ---- labels (§3.4) ----------------------------------------------------------

def need_status(stock_r: int, r: str) -> str:
    if stock_r == 0:
        return f"URGENT: you have no {r} for tonight."
    if stock_r <= 2:
        return f"You have little {r}."
    return f"You already have enough {r}."


def health_status(h: int) -> str:
    if h <= 3:
        return "Your health is dangerously low."
    if h <= 6:
        return "Your health is low."
    return ""


def wealth_label(w: int) -> str:
    # TODO: magic numbers. Add to configuration???
    return "rich" if w >= 8 else "poor" if w <= 3 else "of average wealth"


def strength_label(h: int) -> str:
    return "strong" if h >= 7 else "weak" if h <= 3 else "of average strength"


def reputation_label(record, threshold: int) -> str | None:
    if record.raids >= threshold:
        return "a known raider"
    if record.gifts >= threshold:
        return "known to be generous"
    if record.free_rode >= threshold:
        return "a known shirker"
    if record.contributed >= threshold:
        return "a reliable hunter"
    return None


def recent_events(memory, day: int, window: int, kinds=None, other_id: int | None = None) -> list:
    """Memory events within `window` days, most recent first."""
    return [
        ev for ev in reversed(memory)
        if day - ev.day <= window
        and (kinds is None or ev.kind in kinds)
        and (other_id is None or ev.other_id == other_id)
    ]


def event_sentence(e, named: bool = True) -> str:
    name = f"Agent {e.other_id}" if named else None
    text = EVENT_TEMPLATES[e.kind].format(
        O=name or "", o=name or "them", os=f"{name}'s" if name else "their", d=e.day, r=e.details.value
    ).strip()
    return _cap(text)


def relation_sentence(memory, other_id: int, day: int, window: int, named: bool = False) -> str | None:
    events = recent_events(memory, day, window, other_id=other_id)
    return event_sentence(events[0], named) if events else None


def assessment(h_self: int, h_other: int) -> tuple[str, str, str]:
    """(assessment, fight_note, yield_note) for the defender."""
    diff = h_self - h_other
    if diff >= 3:
        return ("You are much stronger than the attacker.", "You would very likely win.",
                "Needless: you would likely win a fight.")
    if diff <= -3:
        return ("The attacker is much stronger than you.", "You would very likely lose.",
                "Wise: the attacker is much stronger.")
    return "You are evenly matched.", "It could go either way.", "Safe but costly."


# ---- core sentences (§6.3) --------------------------------------------------

def needs_sentence(stock: dict, need: dict) -> str:
    lacking = [r for r in Resource if stock[r] < need[r]]
    if not lacking:
        return "You have enough for tonight."
    return f"You lack {' and '.join(r.value for r in lacking)} for tonight."


def season_word(mult: float) -> str:
    return "scarce" if mult < 1 else "plentiful" if mult > 1 else "normal"


def season_sentence(season: dict) -> str:
    return f"Food is {season_word(season[Resource.FOOD])} and wood is {season_word(season[Resource.WOOD])} this season."


def core_sentence(village, agent) -> str:
    cfg = village.cfg
    return (
        f"You live in a village of {village.n_alive} people, day {village.day} of {cfg.population.D}. "
        f"Health {agent.health}/{cfg.agent.health_max}. "
        f"You have {agent.stock[Resource.FOOD]} food and {agent.stock[Resource.WOOD]} wood; tonight you need 1 of each. "
        f"{needs_sentence(agent.stock, cfg.agent.need)} {season_sentence(village.season)}"
    )


# ---- helpers ----------------------------------------------------------------

def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _num(x: float) -> str:
    return f"{round(x, 1):g}"


def _wealth(agent) -> int:
    return agent.stock[Resource.FOOD] + agent.stock[Resource.WOOD]


def _agent_list(ids: list[int]) -> str:
    if len(ids) == 1:
        return f"Agent {ids[0]}"
    return f"Agents {', '.join(map(str, ids[:-1]))} and {ids[-1]}"


def _others(village, agent) -> list:
    return [a for a in village.agents if a.alive and a.health > 0 and a.id != agent.id]


def _state(village, agent, genes: tuple[str, ...], situation: list[str]) -> str:
    parts = [persona(agent.genome, genes, village.cfg.prompts.persona_mode), core_sentence(village, agent), *situation]
    return " ".join(p for p in parts if p)


def _reputation(village, other_agent) -> str | None:
    if not village.cfg.prompts.render_reputation:
        return None
    return reputation_label(other_agent.record, village.cfg.memory.rep_threshold)


def _relation(village, agent, other_id: int, named: bool) -> str | None:
    if not village.cfg.prompts.render_memory:
        return None
    return relation_sentence(agent.memory, other_id, village.day, village.cfg.memory.days, named)


def _memories(village, agent, kinds=None) -> list[str]:
    cfg = village.cfg
    if not cfg.prompts.render_memory:
        return []
    events = recent_events(agent.memory, village.day, cfg.memory.days, kinds)
    return [event_sentence(ev) for ev in reversed(events[:cfg.memory.render])]


def _hunt_sentence(hunt) -> str:
    if hunt is None or not hunt.took_place:
        return "There was no hunt yesterday."
    if hunt.success:
        return f"Yesterday's hunt succeeded with {hunt.contributors} of {hunt.party_size} hunters working hard."
    return f"Yesterday's hunt failed because only {hunt.contributors} of {hunt.party_size} hunters worked hard."


def _congestion_sentence(congestion: dict) -> str:
    places = {Resource.FOOD: "fields", Resource.WOOD: "forest"}
    crowded = [r for r in Resource if congestion[r] < 0.75]
    if not crowded:
        return ""
    verb = "was" if crowded == [Resource.WOOD] else "were"
    return f"Yesterday the {' and the '.join(places[r] for r in crowded)} {verb} crowded and yields were low."


def _raids_sentence(n: int) -> str:
    if n < 1:
        return ""
    return "Yesterday there was 1 raid in the village." if n == 1 else f"Yesterday there were {n} raids in the village."


def raid_candidates(village, agent) -> list:
    """§5.1: latest grudge, richest, weakest, one random other; deduplicated, at most cfg.raid.max_candidates."""
    others = _others(village, agent)
    if not others:
        return []
    by_id = {a.id: a for a in others}
    ids: list[int] = []
    for ev in recent_events(agent.memory, village.day, village.cfg.memory.days, NEGATIVE_KINDS):
        if ev.other_id in by_id:
            ids.append(ev.other_id)
            break
    for pick in (max(others, key=_wealth), min(others, key=lambda a: a.health)):
        if pick.id not in ids:
            ids.append(pick.id)
    rest = [a.id for a in others if a.id not in ids]
    if rest:
        ids.append(rest[int(village.rng.integers(len(rest)))]) # TODO: No tentiendo que es esto
    return [by_id[i] for i in ids[:village.cfg.raid.max_candidates]]


# ---- prompt catalogue (§6.4) ------------------------------------------------

def morning(village, agent) -> tuple[str, dict]:
    """Questions `activity`, `route` and `target` (§6.4.1); `raid`/`target` omitted if nobody else is alive."""
    cfg, y = village.cfg, village.yesterday
    situation = [
        _congestion_sentence(y.congestion),
        _hunt_sentence(y.hunt),
        _raids_sentence(y.raids),
        *_memories(village, agent),
    ]
    state = _state(village, agent, (Gene.AGGRESSION, Gene.RISK, Gene.SPECIALIZATION), situation)

    expected = {r: cfg.gathering.base_yield * gather_mult(agent.genome, r) * village.season[r] for r in Resource}
    candidates = raid_candidates(village, agent)
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
            parts = [f"{_cap(wealth_label(_wealth(c)))} and {strength_label(c.health)}."]
            rep = _reputation(village, c)
            if rep:
                parts.append(f"{_cap(rep)}.")
            rel = _relation(village, agent, c.id, named=False)
            if rel:
                parts.append(rel)
            targets[f"agent_{c.id}"] = " ".join(parts)
        targets["nobody"] = "Change your mind and forage instead."
        questions["target"] = {"type": "choice", "instructions": "Whom do you rob today?", "criteria": targets}
    return state, questions


def effort(village, agent, party, threshold) -> tuple[str, dict]:
    """Question `effort` (§6.4.2)."""
    cfg = village.cfg
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
    situation += _memories(village, agent, HUNT_KINDS)
    state = _state(village, agent, (Gene.ALTRUISM,), situation)

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


def defend(village, agent, attacker) -> tuple[str, dict]:
    """Question `defend` (§6.4.3); `agent` is the target of the raid."""
    verdict, fight_note, yield_note = assessment(agent.health, attacker.health)
    situation = [f"Agent {attacker.id} is trying to rob you. {verdict}",
                 _relation(village, agent, attacker.id, named=True)]
    state = _state(village, agent, (Gene.AGGRESSION, Gene.RISK), [s for s in situation if s])

    questions = {"defend": {"type": "choice", "instructions": "What do you do?", "criteria": {
        "fight": f"Fight back. {fight_note} The loser loses half of their stock and {village.cfg.raid.fight_dmg} health.",
        "yield": f"Hand over half of your stock and stay unharmed. {yield_note}",
    }}}
    return state, questions


def offer(village, proposer, partner, r, offers) -> tuple[str, dict]:
    """Question `offer` (§6.4.4); `offers` maps affordable offer names to (give other(r), get r)."""
    o, q = other(r), partner.id
    rep = _reputation(village, partner)
    situation = [
        f"You have no {r.value} for tonight and spare {o.value}. Agent {q} has spare {r}" + (f" and is {rep}." if rep else "."),
        _relation(village, proposer, q, named=True),
    ]
    state = _state(village, proposer, (Gene.ALTRUISM,), [s for s in situation if s])

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


def accept(village, responder, proposer, r, offer, give, get) -> tuple[str, dict]:
    """Question `accept` (§6.4.5, noul); `r` is the resource the proposer lacks (Resource enum)."""
    p = proposer.id
    rep = _reputation(village, proposer)
    situation = [
        f"Agent {p} has no {r.value} for tonight and offers you {give} {other(r).value} for {get} {r.value}. {ACCEPT_VERDICTS[offer]}",
        f"Agent {p} is {rep}." if rep else "",
        _relation(village, responder, p, named=True),
    ]
    state = _state(village, responder, (Gene.ALTRUISM,), [s for s in situation if s])
    return state, {"accept": {"type": "noul", "instructions": "Do you accept the trade?"}}


def gift(village, donor, recipient, r) -> tuple[str, dict]:
    """Question `gift` (§6.4.6, noul)."""
    n = recipient.id
    rep = _reputation(village, recipient)
    situation = [
        f"Agent {n} has no {r.value} for tonight and will lose health. You have more {r.value} than you need.",
        f"Agent {n} is {rep}." if rep else "",
        _relation(village, donor, n, named=True),
    ]
    state = _state(village, donor, (Gene.ALTRUISM,), [s for s in situation if s])
    return state, {"gift": {"type": "noul", "instructions": f"Do you give Agent {n} one {r.value}?"}}
