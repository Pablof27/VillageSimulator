# Feature -> text rules (SIMULATION_DESIGN.md §3.4). The numeric `*_level` / `*_kind` helpers hold the thresholds
# once; the text functions (prompts.py) and the policy features (policies.py) are both built on them.

from world import MemoryEventKind as MEK

CROWDED_BELOW = 0.75  # yesterday's congestion factor under which a place counts as crowded

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

# Most recent memory event with another agent -> relation kind (events about my own actions have none).
RELATION_KINDS = {
    MEK.GAVE_ME: "friend", MEK.TRADED_FAIR: "friend", MEK.TRADED_GENEROUS_TO_ME: "friend",
    MEK.WORKED_IN_MY_HUNT: "friend",
    MEK.ROBBED_ME: "enemy", MEK.TRADED_GREEDY_TO_ME: "enemy", MEK.SHIRKED_IN_MY_HUNT: "enemy",
    MEK.REFUSED_ME: "refused",
}
REPUTATION_LABELS = {"raider": "a known raider", "generous": "known to be generous",
                     "shirker": "a known shirker", "reliable": "a reliable hunter"}


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ---- levels (thresholds live here only) -------------------------------------

def need_level(stock_r: int) -> int:
    """0 none, 1 little (1-2), 2 enough (>= 3)."""
    return 0 if stock_r == 0 else 1 if stock_r <= 2 else 2


def health_level(h: int) -> int:
    """0 dangerously low (<= 3), 1 low (4-6), 2 fine."""
    return 0 if h <= 3 else 1 if h <= 6 else 2


def wealth_level(w: int) -> int:
    """-1 poor (<= 3), 0 average, 1 rich (>= 8)."""
    return 1 if w >= 8 else -1 if w <= 3 else 0


def strength_level(h: int) -> int:
    """-1 weak (<= 3), 0 average, 1 strong (>= 7)."""
    return 1 if h >= 7 else -1 if h <= 3 else 0


def assessment_level(h_self: int, h_other: int) -> int:
    """+1 much stronger than the other, -1 much weaker, 0 evenly matched."""
    diff = h_self - h_other
    return 1 if diff >= 3 else -1 if diff <= -3 else 0


def reputation_kind(record, threshold: int) -> str | None:
    """The single public label shown for an agent: raider > generous > shirker > reliable."""
    if record.raids >= threshold:
        return "raider"
    if record.gifts >= threshold:
        return "generous"
    if record.free_rode >= threshold:
        return "shirker"
    if record.contributed >= threshold:
        return "reliable"
    return None


def relation_kind(memory, other_id: int, day: int, window: int) -> str | None:
    """friend | enemy | refused | None, from the most recent event with `other_id` (the one the text shows)."""
    events = recent_events(memory, day, window, other_id=other_id)
    return RELATION_KINDS.get(events[0].kind) if events else None


# ---- text -------------------------------------------------------------------

def need_status(stock_r: int, r: str) -> str:
    return (f"URGENT: you have no {r} for tonight.", f"You have little {r}.",
            f"You already have enough {r}.")[need_level(stock_r)]


def health_status(h: int) -> str:
    return ("Your health is dangerously low.", "Your health is low.", "")[health_level(h)]


def wealth_label(w: int) -> str:
    return {1: "rich", -1: "poor", 0: "of average wealth"}[wealth_level(w)]


def strength_label(h: int) -> str:
    return {1: "strong", -1: "weak", 0: "of average strength"}[strength_level(h)]


def reputation_label(record, threshold: int) -> str | None:
    kind = reputation_kind(record, threshold)
    return REPUTATION_LABELS[kind] if kind else None


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
        O=name or "", o=name or "them", os=f"{name}'s" if name else "their", d=e.day,
        r=e.details.value if e.details else "",
    ).strip()
    return cap(text)


def relation_sentence(memory, other_id: int, day: int, window: int, named: bool = False) -> str | None:
    events = recent_events(memory, day, window, other_id=other_id)
    return event_sentence(events[0], named) if events else None


def assessment(h_self: int, h_other: int) -> tuple[str, str, str]:
    """(assessment, fight_note, yield_note) for the defender."""
    return {
        1: ("You are much stronger than the attacker.", "You would very likely win.",
            "Needless: you would likely win a fight."),
        -1: ("The attacker is much stronger than you.", "You would very likely lose.",
             "Wise: the attacker is much stronger."),
        0: ("You are evenly matched.", "It could go either way.", "Safe but costly."),
    }[assessment_level(h_self, h_other)]
