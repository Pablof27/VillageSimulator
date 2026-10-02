# Prompt catalogue (SIMULATION_DESIGN.md §6.4): state = persona + core + situation.
#
# Duck-typed inputs:
#   village: agents, day, n_alive, rng, season {r: mult}, cfg,
#            yesterday.congestion {r: factor}, yesterday.hunt (took_place, success, contributors, party_size), yesterday.raids
#   agent:   id, alive, health, stock {r: int}, genome (agg, alt, risk, spec),
#            record (raids, gifts, contributed, free_rode), memory [(day, other_id, kind, resource)]
#   cfg:     flat §3.2 parameters + persona_mode, render_reputation, render_memory
from math import floor

RESOURCES = ("food", "wood")
AGGRESSION, ALTRUISM, RISK, SPECIALIZATION = "agg", "alt", "risk", "spec"
GENES = (AGGRESSION, ALTRUISM, RISK, SPECIALIZATION)

PERSONA = {
    AGGRESSION: (
        "You are a gentle farmer who abhors violence and never takes what is not yours.",
        "You are a quiet villager who dislikes conflict.",
        "You are an ordinary villager who takes care of your own.",
        "You are a tough brawler who takes what you need.",
        "You are a feared raider who lives by taking from others.",
    ),
    ALTRUISM: (
        "You are a miser who never shares anything with anyone.",
        "You are a self-reliant villager who seldom helps others.",
        "You are an ordinary villager who helps when it is convenient.",
        "You are a kind neighbour who likes to help others.",
        "You are a devoted helper who always shares with those in need.",
    ),
    RISK: (
        "You are extremely cautious and avoid every risk.",
        "You prefer safe, predictable choices.",
        "You take moderate risks when the reward is worth it.",
        "You are bold and enjoy taking chances.",
        "You are a reckless gambler who loves danger.",
    ),
    SPECIALIZATION: (
        "You are a master woodcutter but a poor forager.",
        "You are a good woodcutter and a mediocre forager.",
        "You are equally able at foraging and woodcutting.",
        "You are a good forager and a mediocre woodcutter.",
        "You are a master forager but a poor woodcutter.",
    ),
}

# {O}: the other agent as subject, {o}: as object, {os}: possessive.
EVENT_TEMPLATES = {
    "robbed_me": "{O} robbed you on day {d}.",
    "i_robbed": "You robbed {o} on day {d}.",
    "gave_me": "{O} gave you {r} on day {d}.",
    "i_gave": "You gave {o} {r} on day {d}.",
    "traded_fair": "{O} traded fairly with you on day {d}.",
    "traded_generous_to_me": "{O} traded generously with you on day {d}.",
    "traded_greedy_to_me": "{O} drove a hard bargain with you on day {d}.",
    "refused_me": "{O} refused your offer on day {d}.",
    "i_refused": "You refused {os} offer on day {d}.",
    "shirked_in_my_hunt": "{O} held back in your hunt on day {d}.",
    "worked_in_my_hunt": "{O} worked hard in your hunt on day {d}.",
}
NEGATIVE_KINDS = ("robbed_me", "refused_me", "shirked_in_my_hunt")
HUNT_KINDS = ("shirked_in_my_hunt", "worked_in_my_hunt")
MAX_TARGETS = 4

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

def other(r: str) -> str:
    return "wood" if r == "food" else "food"


def level(g: float) -> int:
    return min(4, floor(5 * g))


def gather_mult(genome, r: str) -> float:
    return 0.5 + genome.spec if r == "food" else 1.5 - genome.spec


def persona(genome, genes: tuple[str, ...], mode: str = "gated") -> str:
    if mode == "none":
        return ""
    if mode == "full":
        genes = GENES
    return " ".join(PERSONA[g][level(getattr(genome, g))] for g in genes)


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
        if day - ev[0] <= window
        and (kinds is None or ev[2] in kinds)
        and (other_id is None or ev[1] == other_id)
    ]


def event_sentence(event, named: bool = True) -> str:
    d, other_id, kind, resource = event
    name = f"Agent {other_id}" if named else None
    text = EVENT_TEMPLATES[kind].format(
        O=name or "", o=name or "them", os=f"{name}'s" if name else "their", d=d, r=resource
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
    lacking = [r for r in RESOURCES if stock[r] < need[r]]
    if not lacking:
        return "You have enough for tonight."
    return f"You lack {' and '.join(lacking)} for tonight."


def season_word(mult: float) -> str:
    return "scarce" if mult < 1 else "plentiful" if mult > 1 else "normal"


def season_sentence(season: dict) -> str:
    return f"Food is {season_word(season['food'])} and wood is {season_word(season['wood'])} this season."


def core_sentence(village, agent) -> str:
    cfg = village.cfg
    return (
        f"You live in a village of {village.n_alive} people, day {village.day} of {cfg.D}. "
        f"Health {agent.health}/{cfg.health_max}. "
        f"You have {agent.stock['food']} food and {agent.stock['wood']} wood; tonight you need 1 of each. "
        f"{needs_sentence(agent.stock, cfg.need)} {season_sentence(village.season)}"
    )


# ---- helpers ----------------------------------------------------------------

def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _num(x: float) -> str:
    return f"{round(x, 1):g}"


def _wealth(agent) -> int:
    return agent.stock["food"] + agent.stock["wood"]


def _agent_list(ids: list[int]) -> str:
    if len(ids) == 1:
        return f"Agent {ids[0]}"
    return f"Agents {', '.join(map(str, ids[:-1]))} and {ids[-1]}"


def _others(village, agent) -> list:
    return [a for a in village.agents if a.alive and a.health > 0 and a.id != agent.id]


def _state(village, agent, genes: tuple[str, ...], situation: list[str]) -> str:
    parts = [persona(agent.genome, genes, village.cfg.persona_mode), core_sentence(village, agent), *situation]
    return " ".join(p for p in parts if p)


def _reputation(village, other_agent) -> str | None:
    if not village.cfg.render_reputation:
        return None
    return reputation_label(other_agent.record, village.cfg.rep_threshold)


def _relation(village, agent, other_id: int, named: bool) -> str | None:
    if not village.cfg.render_memory:
        return None
    return relation_sentence(agent.memory, other_id, village.day, village.cfg.memory_days, named)


def _memories(village, agent, kinds=None) -> list[str]:
    cfg = village.cfg
    if not cfg.render_memory:
        return []
    events = recent_events(agent.memory, village.day, cfg.memory_days, kinds)
    return [event_sentence(ev) for ev in reversed(events[:cfg.memory_render])]


def _hunt_sentence(hunt) -> str:
    if hunt is None or not hunt.took_place:
        return "There was no hunt yesterday."
    if hunt.success:
        return f"Yesterday's hunt succeeded with {hunt.contributors} of {hunt.party_size} hunters working hard."
    return f"Yesterday's hunt failed because only {hunt.contributors} of {hunt.party_size} hunters worked hard."


def _congestion_sentence(congestion: dict) -> str:
    places = {"food": "fields", "wood": "forest"}
    crowded = [r for r in RESOURCES if congestion.get(r, 1.0) < 0.75]
    if not crowded:
        return ""
    verb = "was" if crowded == ["wood"] else "were"
    return f"Yesterday the {' and the '.join(places[r] for r in crowded)} {verb} crowded and yields were low."


def _raids_sentence(n: int) -> str:
    if n < 1:
        return ""
    return "Yesterday there was 1 raid in the village." if n == 1 else f"Yesterday there were {n} raids in the village."


def raid_candidates(village, agent) -> list:
    """§5.1: latest grudge, richest, weakest, one random other; deduplicated, at most MAX_TARGETS."""
    others = _others(village, agent)
    if not others:
        return []
    by_id = {a.id: a for a in others}
    ids: list[int] = []
    for ev in recent_events(agent.memory, village.day, village.cfg.memory_days, NEGATIVE_KINDS):
        if ev[1] in by_id:
            ids.append(ev[1])
            break
    for pick in (max(others, key=_wealth), min(others, key=lambda a: a.health)):
        if pick.id not in ids:
            ids.append(pick.id)
    rest = [a.id for a in others if a.id not in ids]
    if rest:
        ids.append(rest[int(village.rng.integers(len(rest)))])
    return [by_id[i] for i in ids[:MAX_TARGETS]]


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
    state = _state(village, agent, (AGGRESSION, RISK, SPECIALIZATION), situation)

    expected = {r: cfg.base_yield * gather_mult(agent.genome, r) * village.season[r] for r in RESOURCES}
    candidates = raid_candidates(village, agent)
    activity = {
        "gather_food": f"Forage alone. Expect about {_num(expected['food'])} food. Safe. "
                       f"{need_status(agent.stock['food'], 'food')}",
        "gather_wood": f"Cut wood alone. Expect about {_num(expected['wood'])} wood. Safe. "
                       f"{need_status(agent.stock['wood'], 'wood')}",
        "hunt": f"Join the group hunt. If enough hunters work hard, every hunter gets {cfg.hunt_payout} food. "
                "If it fails you get nothing today.",
    }
    if candidates:
        activity["raid"] = ("Rob another villager and take half of their stock. "
                            "They may fight back and hurt you, and everyone will know you did it.")
    activity["rest"] = (f"Stay home and recover {cfg.rest_regen} health. You collect nothing. "
                        f"{health_status(agent.health)}").strip()

    questions = {
        "activity": {"type": "choice", "instructions": "Choose what to do today.", "criteria": activity},
        "route": {"type": "choice", "instructions": "Where do you gather?", "criteria": {
            "near": "Nearby. Safe. Normal yield.",
            "far": f"Far away. Half again as much yield, but a 1 in {round(1 / cfg.far_injury_p)} chance "
                   f"of an injury costing {cfg.far_injury_dmg} health.",
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
    if cfg.render_reputation:
        th = cfg.rep_threshold
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
    state = _state(village, agent, (ALTRUISM,), situation)

    payout = min(cfg.hunt_payout, floor(cfg.hunt_supply_per_capita * village.n_alive / m))
    questions = {"effort": {
        "type": "choice",
        "instructions": f"The hunt succeeds only if at least {threshold} of the {m} hunters work hard. "
                        "Choose your effort.",
        "criteria": {
            "contribute": f"Work hard. Costs you {cfg.hunt_effort_cost} health. "
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
    state = _state(village, agent, (AGGRESSION, RISK), [s for s in situation if s])

    questions = {"defend": {"type": "choice", "instructions": "What do you do?", "criteria": {
        "fight": f"Fight back. {fight_note} The loser loses half of their stock and {village.cfg.fight_dmg} health.",
        "yield": f"Hand over half of your stock and stay unharmed. {yield_note}",
    }}}
    return state, questions


def offer(village, proposer, partner, r, offers) -> tuple[str, dict]:
    """Question `offer` (§6.4.4); `offers` maps affordable offer names to (give other(r), get r)."""
    o, q = other(r), partner.id
    rep = _reputation(village, partner)
    situation = [
        f"You have no {r} for tonight and spare {o}. Agent {q} has spare {r}" + (f" and is {rep}." if rep else "."),
        _relation(village, proposer, q, named=True),
    ]
    state = _state(village, proposer, (ALTRUISM,), [s for s in situation if s])

    criteria = {
        name: f"Offer {give} {o} for {get} {r}. "
              + OFFER_NOTES[name].format(q=q, left=proposer.stock[o] - give, o=o, r=r)
        for name, (give, get) in offers.items()
    }
    questions = {"offer": {
        "type": "choice",
        "instructions": f"What do you offer Agent {q}? A poor offer may be refused.",
        "criteria": criteria,
    }}
    return state, questions


def accept(village, responder, proposer, r, offer, give, get) -> tuple[str, dict]:
    """Question `accept` (§6.4.5, noul); `r` is the resource the proposer lacks."""
    p = proposer.id
    rep = _reputation(village, proposer)
    situation = [
        f"Agent {p} has no {r} for tonight and offers you {give} {other(r)} for {get} {r}. {ACCEPT_VERDICTS[offer]}",
        f"Agent {p} is {rep}." if rep else "",
        _relation(village, responder, p, named=True),
    ]
    state = _state(village, responder, (ALTRUISM,), [s for s in situation if s])
    return state, {"accept": {"type": "noul", "instructions": "Do you accept the trade?"}}


def gift(village, donor, recipient, r) -> tuple[str, dict]:
    """Question `gift` (§6.4.6, noul)."""
    n = recipient.id
    rep = _reputation(village, recipient)
    situation = [
        f"Agent {n} has no {r} for tonight and will lose health. You have more {r} than you need.",
        f"Agent {n} is {rep}." if rep else "",
        _relation(village, donor, n, named=True),
    ]
    state = _state(village, donor, (ALTRUISM,), [s for s in situation if s])
    return state, {"gift": {"type": "noul", "instructions": f"Do you give Agent {n} one {r}?"}}
