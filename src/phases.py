# Day-cycle mechanics (SIMULATION_DESIGN.md §5).

from itertools import permutations
from math import floor, ceil
from config import Config
from labels import recent_events
from resources import other, Resource
from world import MemoryEventKind as MEK, living_others, Village,  mult_r, Hunt, MemoryEvent
import policies
import prompts
from brain import decide

NEGATIVE_KINDS = (MEK.ROBBED_ME, MEK.REFUSED_ME, MEK.SHIRKED_IN_MY_HUNT)
HUNT_MEMORY = {"contribute": MEK.WORKED_IN_MY_HUNT, "free_ride": MEK.SHIRKED_IN_MY_HUNT}
TRADE_MEMORY = {"fair": MEK.TRADED_FAIR, "generous": MEK.TRADED_GENEROUS_TO_ME, "greedy": MEK.TRADED_GREEDY_TO_ME}


def run_day(cfg: Config, v: Village, brain, rng, log) -> None:
    v.day_count += 1
    alive = [a for a in v.agents if a.alive]
    _reset_scratch(alive)

    _morning(cfg, v, alive, brain, rng, log)
    _gathering(cfg, v, alive, rng, log)
    _rest(cfg, alive)
    _hunt(cfg, v, alive, brain, log)
    _raids(cfg, v, alive, brain, rng, log)
    if cfg.market.enabled:
        _market(cfg, v, alive, brain, rng, log)
    if cfg.gifts.enabled:
        _gifts(cfg, v, alive, brain, rng, log)
    _night(cfg, v, alive, log)


def _reset_scratch(alive):
    for a in alive:
        a.activity = a.route = a.target = a.last_damage = None
        a.traded_today = False
        a.asked_gift_today = set()


def _ask(v, a, phase, state, questions, brain, log) -> dict:
    answers = decide(state, questions, brain)
    log.decision(v.day_count, phase, a, state, questions, answers)
    return answers


def catalogue(cfg: Config):
    """The decision catalogue: analytic policies (§6.7) for `brain.kind: policy`, Laya prompts (§6.4) otherwise."""
    return policies if cfg.brain.kind == "policy" else prompts


# ---- 1. MORNING -------------------------------------------------------------

def _morning(cfg, v, alive, brain, rng, log):
    for a in alive:
        _plan_day(cfg, v, a, brain, rng, log)


def _plan_day(cfg, v, a, brain, rng, log):
    state, questions = catalogue(cfg).morning(cfg, v, a, _raid_candidates(cfg, v, a, rng))
    res = _ask(v, a, "morning", state, questions, brain, log)
    a.activity = res["activity"].chosen
    a.route = res["route"].chosen
    if a.activity != "raid":
        return
    target = res["target"].chosen
    if target == "nobody":
        a.activity = "gather_food"
        log.emit("raid_aborted", "morning", v.day_count, agent=a.id)
        return
    a.target = int(target.split("_")[1])


# ---- 2. GATHERING AND REST --------------------------------------------------

def _gathering(cfg, v, alive, rng, log):
    for r in Resource:
        gatherers = [a for a in alive if a.activity == f"gather_{r.value}"]
        raws = {a.id: _raw_yield(a, r, cfg, v) for a in gatherers}
        factor = _congestion_factor(cfg, v, r, raws, len(alive))
        v.yesterday.congestion[r] = factor
        for a in gatherers:
            _gather(cfg, v, a, r, raws[a.id], factor, rng, log)


def _congestion_factor(cfg, v, r, raws, n_alive) -> float:
    if not raws:
        return 1.0
    supply = cfg.gathering.supply_per_capita * n_alive * v.season[r]
    return min(1.0, supply / sum(raws.values()))


def _gather(cfg, v, a, r, raw, factor, rng, log):
    got = _sround(raw * factor, rng)
    a.stock[r] += got
    injured = a.route == "far" and rng.random() < cfg.gathering.far_injury_p
    if injured:
        a.health -= cfg.gathering.far_injury_dmg
        a.last_damage = "injury"
    log.emit("gather", "gathering", v.day_count, agent=a.id, resource=r, route=a.route,
             raw=raw, factor=factor, got=got, injured=injured)


def _rest(cfg, alive):
    resting = [a for a in alive if a.activity == "rest"]
    for a in resting:
        a.health = min(cfg.agent.health_max, a.health + cfg.agent.rest_regen)


# ---- 3. HUNT ----------------------------------------------------------------

def _hunt(cfg, v, alive, brain, log):
    party = [a for a in alive if a.activity == "hunt" and a.health > 0]
    v.yesterday.hunt = Hunt(took_place=False)
    if not party:
        return
    if len(party) < cfg.hunt.min_party:
        _cancel_hunt(cfg, v, party, log)
        return
    v.yesterday.hunt = _run_hunt(cfg, v, party, len(alive), brain, log)


def _cancel_hunt(cfg, v, party, log):
    for a in party:
        a.stock[Resource.FOOD] += cfg.hunt.cancel_food
    log.emit("hunt", "hunt", v.day_count, party=[a.id for a in party], threshold=None, contributors=[],
             free_riders=[], known_shirkers=[], success=False, payout=cfg.hunt.cancel_food, cancelled=True)


def _run_hunt(cfg, v, party, n_alive, brain, log) -> Hunt:
    threshold = ceil(cfg.hunt.threshold_frac * len(party))
    known_shirkers = [a.id for a in party if a.record.free_rode >= cfg.memory.rep_threshold]
    efforts = {a.id: _choose_effort(cfg, v, a, party, threshold, brain, log) for a in party}
    _pay_effort_costs(cfg, party, efforts)

    contributors = [a.id for a in party if efforts[a.id] == "contribute"]
    success = len(contributors) >= threshold
    payout = _hunt_payout(cfg, len(party), n_alive) if success else 0
    for a in party:
        a.stock[Resource.FOOD] += payout

    for me, mate in permutations(party, 2):
        _record_interaction(v.day_count, me, mate, HUNT_MEMORY[efforts[mate.id]])
    log.emit("hunt", "hunt", v.day_count, party=[a.id for a in party], threshold=threshold,
             contributors=contributors, free_riders=[a.id for a in party if efforts[a.id] == "free_ride"],
             known_shirkers=known_shirkers, success=success, payout=payout, cancelled=False)
    return Hunt(took_place=True, success=success, contributors=len(contributors), party_size=len(party))


def _choose_effort(cfg, v, a, party, threshold, brain, log) -> str:
    state, qs = catalogue(cfg).effort(cfg, v, a, party, threshold)
    return _ask(v, a, "hunt", state, qs, brain, log)["effort"].chosen


def _pay_effort_costs(cfg, party, efforts):
    for a in party:
        if efforts[a.id] == "free_ride":
            a.record.free_rode += 1
            continue
        a.health -= cfg.hunt.effort_cost
        a.last_damage = "hunt_exhaustion"
        a.record.contributed += 1


def _hunt_payout(cfg, party_size, n_alive) -> int:
    return min(cfg.hunt.payout, floor(cfg.hunt.supply_per_capita * n_alive / party_size))


# ---- 4. RAIDS ---------------------------------------------------------------

def _raids(cfg, v, alive, brain, rng, log):
    v.yesterday.raids = 0
    raiders = [a for a in alive if a.activity == "raid"]
    for a in _permutation(rng, raiders):
        _raid(cfg, v, a, v.agent(a.target), brain, rng, log)


def _raid(cfg, v, a, t, brain, rng, log):
    if a.health <= 0 or not t.alive or t.health <= 0:
        return
    state, qs = catalogue(cfg).defend(cfg, v, t, a)
    defence = _ask(v, t, "raid", state, qs, brain, log)["defend"].chosen
    loot, dmg_to, attacker_won = _resolve_raid(cfg, a, t, defence, rng)
    a.record.raids += 1
    v.yesterday.raids += 1
    _record_interaction(v.day_count, t, a, MEK.ROBBED_ME)
    _record_interaction(v.day_count, a, t, MEK.I_ROBBED)
    log.emit("raid", "raid", v.day_count, attacker=a.id, target=t.id, defence=defence,
             attacker_won=attacker_won, loot=loot, dmg_to=dmg_to)


def _resolve_raid(cfg, a, t, defence, rng) -> tuple[dict, int | None, bool]:
    if defence == "yield":
        return _take_half(t, a, cfg.raid.loot_frac), None, True
    attacker_won = rng.random() < a.health / (a.health + t.health)
    winner, loser = (a, t) if attacker_won else (t, a)
    loser.health -= cfg.raid.fight_dmg
    loser.last_damage = "fight"
    return _take_half(loser, winner, cfg.raid.loot_frac), loser.id, attacker_won


# ---- 5. MARKET --------------------------------------------------------------

def _market(cfg, v, alive, brain, rng, log):
    for r in _permutation(rng, list(Resource)):
        for p in _permutation(rng, _needy(alive, r)):
            _try_trade(cfg, v, alive, p, r, brain, rng, log)


def _needy(alive, r) -> list:
    o = other(r)
    return [a for a in alive if a.health > 0 and a.stock[r] == 0 and a.stock[o] >= 2 and not a.traded_today]


def _try_trade(cfg, v, alive, p, r, brain, rng, log):
    partners = _trade_partners(cfg, alive, p, r)
    if not partners:
        return
    q = rng.choice(partners)
    offers = _affordable_offers(cfg, p, q, r)
    if not offers:
        return
    state, qs = catalogue(cfg).offer(cfg, v, p, q, r, offers)
    offered = _ask(v, p, "market", state, qs, brain, log)["offer"].chosen
    give, get = offers[offered]
    state, qs = catalogue(cfg).accept(cfg, v, q, p, r, offered, give, get)
    accepted = _ask(v, q, "market", state, qs, brain, log)["accept"].chosen == "yes"
    _settle_trade(v.day_count, p, q, r, offered, give, get, accepted)
    log.emit("trade", "market", v.day_count, proposer=p.id, responder=q.id, resource=r,
             offer=offered, accepted=accepted, give=give, get=get)


def _trade_partners(cfg, alive, p, r) -> list:
    return [q for q in alive
            if q is not p and q.health > 0 and not q.traded_today and q.stock[r] >= cfg.market.spare_min]


def _affordable_offers(cfg, p, q, r) -> dict:
    """Offers after which the proposer keeps >= 1 of `other(r)` and the responder keeps >= 1 of `r`."""
    o = other(r)
    return {name: (give, get) for name, (give, get) in cfg.market.offers.items()
            if p.stock[o] - give >= 1 and q.stock[r] - get >= 1}


def _settle_trade(day, p, q, r, offered, give, get, accepted):
    if not accepted:
        _record_interaction(day, p, q, MEK.REFUSED_ME)
        _record_interaction(day, q, p, MEK.I_REFUSED)
        return
    o = other(r)
    p.stock[o] -= give
    q.stock[o] += give
    q.stock[r] -= get
    p.stock[r] += get
    p.traded_today = True
    q.traded_today = True
    _record_interaction(day, q, p, TRADE_MEMORY[offered])
    _record_interaction(day, p, q, MEK.TRADED_FAIR)


# ---- 6. GIFTS AND NIGHT -----------------------------------------------------

def _gifts(cfg, v, alive, brain, rng, log):
    for n in _permutation(rng, [a for a in alive if a.health > 0]):
        for r in Resource:
            _ask_for_gift(cfg, v, alive, n, r, brain, rng, log)


def _ask_for_gift(cfg, v, alive, n, r, brain, rng, log):
    if n.stock[r] > 0:
        return
    donors = _gift_donors(cfg, alive, n, r)
    if not donors:
        return
    d = rng.choice(donors)
    d.asked_gift_today.add(r)
    reciprocal = bool(recent_events(d.memory, v.day_count, cfg.memory.days, (MEK.GAVE_ME,), n.id))
    state, qs = catalogue(cfg).gift(cfg, v, d, n, r)
    given = _ask(v, d, "night", state, qs, brain, log)["gift"].chosen == "yes"
    if given:
        _give(v.day_count, d, n, r)
    log.emit("gift", "night", v.day_count, donor=d.id, recipient=n.id, resource=r,
             given=given, reciprocal=reciprocal)


def _gift_donors(cfg, alive, n, r) -> list:
    return [d for d in alive
            if d is not n and d.health > 0 and d.stock[r] >= cfg.gifts.spare_min and r not in d.asked_gift_today]


def _give(day, d, n, r):
    d.stock[r] -= 1
    n.stock[r] += 1
    d.record.gifts += 1
    _record_interaction(day, n, d, MEK.GAVE_ME, r)
    _record_interaction(day, d, n, MEK.I_GAVE, r)


def _night(cfg, v, alive, log):
    for a in alive:
        consumed = _consume(cfg, a)
        a.days_alive += 1
        _die_if_exhausted(v, a, log)
        log.emit("night", "night", v.day_count, agent=a.id, ate=consumed[Resource.FOOD],
                 burned=consumed[Resource.WOOD], health_after=a.health)


def _consume(cfg, a) -> dict:
    """Returns {resource: need met} (all False for an agent already at 0 health)."""
    if a.health <= 0:
        return {r: False for r in Resource}
    needs_met = {r: _consume_resource(cfg, a, r) for r in Resource}
    if all(needs_met.values()):
        a.health = min(cfg.agent.health_max, a.health + cfg.agent.regen)
    return needs_met


def _consume_resource(cfg, a, r) -> bool:
    if a.stock[r] < cfg.agent.need[r]:
        a.health -= cfg.agent.miss_penalty
        a.last_damage = "starvation"
        return False
    a.stock[r] -= cfg.agent.need[r]
    return True


def _die_if_exhausted(v, a, log):
    if a.health > 0:
        return
    a.alive = False
    a.stock = {r: 0 for r in Resource}
    log.emit("death", "night", v.day_count, agent=a.id, cause=a.last_damage or "starvation")


# ---- helpers ----------------------------------------------------------------

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
    grudges = [ev.other_id for ev in recent_events(agent.memory, village.day_count, cfg.memory.days, NEGATIVE_KINDS)
               if ev.other_id in by_id]
    ids = grudges[:1]
    for pick in (max(others, key=lambda a: a.wealth), min(others, key=lambda a: a.health)):
        if pick.id not in ids:
            ids.append(pick.id)
    rest = [a.id for a in others if a.id not in ids]
    if rest:
        ids.append(int(rng.choice(rest)))
    return [by_id[i] for i in ids[:cfg.raid.max_candidates]]
    
def _record_interaction(day_count, me, other, mek, detail = None):
    mem = MemoryEvent(day=day_count, other_id=other.id, kind=mek, details=detail)
    me.memory.append(mem)
