# Day-cycle mechanics (SIMULATION_DESIGN.md §5).

from config import Config
from labels import recent_events
from world import MemoryEventKind as MEK, living_others

NEGATIVE_KINDS = (MEK.ROBBED_ME, MEK.REFUSED_ME, MEK.SHIRKED_IN_MY_HUNT)


def raid_candidates(cfg: Config, village, agent, rng) -> list:
    """§5.1: latest grudge, richest, weakest, one random other; deduplicated, at most cfg.raid.max_candidates."""
    others = living_others(village, agent)
    if not others:
        return []
    by_id = {a.id: a for a in others}
    ids: list[int] = []
    for ev in recent_events(agent.memory, village.day, cfg.memory.days, NEGATIVE_KINDS):
        if ev.other_id in by_id:
            ids.append(ev.other_id)
            break
    for pick in (max(others, key=lambda a: a.wealth), min(others, key=lambda a: a.health)):
        if pick.id not in ids:
            ids.append(pick.id)
    rest = [a.id for a in others if a.id not in ids]
    if rest:
        ids.append(int(rng.choice(rest)))
    return [by_id[i] for i in ids[:cfg.raid.max_candidates]]
