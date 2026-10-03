# Day-cycle mechanics (SIMULATION_DESIGN.md §5).

from math import floor, ceil
from config import Config
from labels import recent_events
from world import MemoryEventKind as MEK, living_others, Village, Resource, mult_r, Hunt, MemoryEvent
from prompts import morning, effort
from brain import decide

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

    
def run_day(cfg: Config, v: Village, brain, rng, log) -> None:
    v.day_count += 1
    alive = [a for a in v.agents if a.alive]
    for a in alive:
        # TODO: reset scratch
        pass
    
    # MORNING

    for a in alive:
        state, questions = morning(cfg, v, a, raid_candidates(cfg, v, a, rng))
        res = decide(state, questions, brain)
        a.activity = res["activity"].chosen
        a.route = res["route"].chosen
        if a.activity == "raid":
            target = res["target"].chosen
            if target == "nobody": a.activity = "gather_food"
            # TODO: log raid aborted
            else: a.target = int(target.split("_")[1])

    # GATHERING AND REST
    
    for r in Resource:
        gatherers = [a for a in alive if a.activity == f"gather_{r.value}"]
        raws = {a.id: _raw_yield(a, r, cfg, v) for a in gatherers}
        supply = cfg.gathering.supply_per_capita * len(alive) * v.season[r]
        factor = min(1.0, supply / sum(raws.values())) if raws else 1.0
        v.yesterday.congestion[r] = factor
        for a in gatherers:
            got = _sround(raws[a.id] * factor, rng)
            a.stock[r] += got
            injured = a.route == "far" and rng.random() < cfg.gathering.far_injury_p
            if injured:
                a.health -= cfg.gathering.far_injury_dmg
                # TODO: log injury
    for a in alive:
        if a.activity == "rest":
            a.health = min(cfg.agent.health_max, a.health + cfg.agent.rest_regen)

    # HUNT
    
    party = [a for a in alive if a.activity == "hunt" and a.health > 0]
    if len(party) > 0 and len(party) < cfg.hunt.min_party:
        for a in party:
            a.stock[Resource.FOOD] += cfg.hunt.cancel_food
        v.yesterday.hunt = Hunt(took_place=False)
        # TODO: log hunt canceled
    elif party:
        threshold = ceil(cfg.hunt.threshold_frac * len(party))
        efforts = {}
        for a in party:
            state, qs = effort(cfg, v, a, party, threshold)
            efforts[a.id] = decide(state, qs, brain)["effort"].chosen
        contributors = [a for a in party if efforts[a.id] == "contribute"]
        for a in contributors:
            a.health -= cfg.hunt.effort_cost
            a.record.contributed += 1
        for a in party:
            if efforts[a.id] == "free_ride":
                a.record.free_rode += 1
        success = len(contributors) >= threshold
        payouts = min(cfg.hunt.payout, floor(cfg.hunt.supply_per_capita * len(alive) / len(party))) if success else 0
        for a in party:
            a.stock[Resource.FOOD] += payouts
        for m in party:
            for o in party:
                if o is not m:
                    mek = MEK.WORKED_IN_MY_HUNT if efforts[o.id] == "contribute" else MEK.SHIRKED_IN_MY_HUNT
                    _record_interaction(v.day_count, m, o, mek)
        v.yesterday.hunt = Hunt(took_place=True, success=success, contributors=len(contributors), party_size=len(party))
        # TODO: log hunt result
    else:
        v.yesterday.hunt = Hunt(took_place=False)

    # RAIDS
    
    v.yesterday.raids = 0
    for A in _permutation(rng, alive):
        pass


        
# PRIVATE FUNCTIONS

def _raw_yield(agent, r, cfg, v):
    far_mult = cfg.gathering.far_multiplier if agent.route == "far" else 1.0
    return cfg.gathering.base_yield * mult_r(agent, r) * v.season[r] * far_mult

def _sround(value: float, rng) -> int:
    base = floor(value)
    return base + int(rng.random() < value - base)

def _permutation(rng, items: list) -> list:
    return rng.sample(items, len(items))

def _record_interaction(day_count, m, o, mek):
    mem = MemoryEvent(day=day_count, other_id=o, kind=mek)
    m.memory.append(mem)
