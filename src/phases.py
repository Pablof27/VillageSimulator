# Day-cycle mechanics (SIMULATION_DESIGN.md §5).

from math import floor, ceil
from config import Config
from labels import recent_events
from resources import other, Resource
from world import MemoryEventKind as MEK, living_others, Village,  mult_r, Hunt, MemoryEvent
from prompts import morning, effort, defend, offer, accept, gift
from brain import decide

NEGATIVE_KINDS = (MEK.ROBBED_ME, MEK.REFUSED_ME, MEK.SHIRKED_IN_MY_HUNT)

def run_day(cfg: Config, v: Village, brain, rng, log) -> None:
    v.day_count += 1
    alive = [a for a in v.agents if a.alive]
    for a in alive:
        # TODO: reset scratch
        pass
    
    # MORNING

    for a in alive:
        state, questions = morning(cfg, v, a, _raid_candidates(cfg, v, a, rng))
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
    for a in _permutation(rng, [a for a in alive if a.activity == "raid"]):
        t = v.agent(a.target)
        if a.health <= 0 or not t.alive or t.health <= 0:
            continue
        state, qs = defend(cfg, v, a, t)
        defence = decide(state, qs, brain)["defence"].chosen
        if defence == "yield":
            loot = _take_half(t, a, cfg.raid.loot_frac)
            dmg_to = None
            attacker_won = True
        else:
            attacker_won = rng.random() < a.health / (a.health + t.health)
            winner, loser = (a, t) if attacker_won else (t, a)
            loser.health -= cfg.raid.fight_dmg
            loot = _take_half(loser, winner, cfg.raid.loot_frac)
            dmg_to = loser.id
        a.record.raids += 1
        v.yesterday.raids += 1
        _record_interaction(v.day_count, t, a, MEK.ROBBED_ME)
        _record_interaction(v.day_count, a, t, MEK.I_ROBBED)
        # TODO: log raid result

    # MARKET
    
    if cfg.market.enabled:
        for r in _permutation(rng, [r for r in Resource]):
            o = other(r)
            needy = [a for a in alive if a.health > 0 and a.stock[r] == 0 and a.stock[o] >= 2 and not a.traded_today]
            for p in _permutation(rng, needy):
                partners = [q for q in alive if q is not p and q.health > 0 and not q.traded_today and q.stock[r] >= cfg.market.spare_min]
                if not partners:
                    continue
                q = rng.choice(partners)
                offers = {name: (give, get) for name, (give, get) in cfg.market.offers.items() if p.stock[r] > give and q.stock[o] > get}
                if not offers:
                    continue
                state, qs = offer(cfg, v, p, q, r, offers)
                offered = decide(state, qs, brain)["offer"].chosen
                give, get = offers[offered]
                state, qs = accept(cfg, v, q, p, r, give, get)
                accepted = decide(state, qs, brain)["accept"].chosen == "yes"
                if accepted:
                    p.stock[o] -= give
                    q.stock[o] += give
                    q.stock[r] -= get
                    p.stock[r] += get
                    p.traded_today = True
                    q.traded_today = True
                    mek = MEK.TRADED_FAIR if offered == "fair" else MEK.TRADED_GENEROUS_TO_ME if offered == "generous" else MEK.TRADED_GREEDY_TO_ME
                    _record_interaction(v.day_count, q, p, mek)
                    _record_interaction(v.day_count, p, q, MEK.TRADED_FAIR)
                else:
                    _record_interaction(v.day_count, p, q, MEK.REFUSED_ME)
                    _record_interaction(v.day_count, q, p, MEK.I_REFUSED)
                # TODO: log trade

    # GIFTS AND NIGHT
    
    for n in _permutation(rng, [a for a in alive if a.health > 0]):
        for r in Resource:
            if n.stock[r] > 0:
                continue
            donors = [d for d in alive if d is not n and d.health > 0 and d.stock[r] >= cfg.gifts.spare_min and r not in d.asked_gift_today]
            if not donors:
                continue
            d = rng.choice(donors)
            d.asked_gift_today.add(r)
            state, qs = gift(cfg, v, d, n, r)
            given = decide(state, qs, brain)["gift"].chosen == "yes"
            if given:
                d.stock[r] -= 1
                n.stock[r] += 1
                d.record.gifts += 1
                _record_interaction(v.day_count, n, d, MEK.GAVE_ME, r)
                _record_interaction(v.day_count, d, n, MEK.I_GAVE, r)
            # TODO: log gift
            
    for a in alive:
        if a.health > 0:
            met = 0
            for r in Resource:
                if a.stock[r] >= cfg.agent.need[r]:
                    a.stock[r] -= cfg.agent.need[r]
                    met += 1
                else:
                    a.health -= cfg.agent.miss_penalty
            if met == 2:
                a.health = min(cfg.agent.max_health, a.health + cfg.agent.regen)
        a.days_alive += 1
        if a.health <= 0:
            a.alive = False
            a.stock = {r: 0 for r in Resource}
            # TODO: log death
        # TODO: log night
       

# PRIVATE FUNCTIONS

def _take_half(target, agent, fraction) -> dict:
    loot = {r: floor(fraction * target.stock[r]) for r in Resource}
    for r, amount in loot.items():
        target.stock[r] -= amount
        agent.stock[r] += amount
    return loot

def _raw_yield(agent, r, cfg, v):
    far_mult = cfg.gathering.far_multiplier if agent.route == "far" else 1.0
    return cfg.gathering.base_yield * mult_r(agent, r) * v.season[r] * far_mult

def _sround(value: float, rng) -> int:
    base = floor(value)
    return base + int(rng.random() < value - base)

def _permutation(rng, items: list) -> list:
    return rng.sample(items, len(items))

def _raid_candidates(cfg: Config, village, agent, rng) -> list:
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
    
def _record_interaction(day_count, me, other, mek, detail = None):
    mem = MemoryEvent(day=day_count, other_id=other.id, kind=mek, detail=detail)
    me.memory.append(mem)
