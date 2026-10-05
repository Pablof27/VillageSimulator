# Feature -> text rules (SIMULATION_DESIGN.md §3.4): pure functions used by the prompt catalogue.

from world import MemoryEventKind as MEK

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


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


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
    diff = h_self - h_other
    if diff >= 3:
        return ("You are much stronger than the attacker.", "You would very likely win.",
                "Needless: you would likely win a fight.")
    if diff <= -3:
        return ("The attacker is much stronger than you.", "You would very likely lose.",
                "Wise: the attacker is much stronger.")
    return "You are evenly matched.", "It could go either way.", "Safe but costly."
