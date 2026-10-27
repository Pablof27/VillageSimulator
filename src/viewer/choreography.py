# Choreography: turns one logged day (and each change of generation) into a timeline. The day follows the phases
# of SIMULATION_DESIGN.md §5 in order: morning plans, work (gathering, rest, hunt), raids, market, gifts, night.
import math
import random
from collections import defaultdict

from resources import Resource
from viewer import sprites
from viewer.replay import OTHER, DayLog, GenEnd, GenStart
from viewer.style import BAD, GOOD, MUTED
from viewer.timeline import (INF, Banner, Bubble, Dust, Effect, Emote, Flash, Fly, Ghost, Label, Popup, Timeline,
                             clamp01, ease, lerp)

# Seconds at 1x speed. Busy phases are squeezed to at most *_MAX seconds.
DAWN, OUT, WORK, BACK, NIGHT = 1.0, 1.1, 1.7, 1.0, 1.8
RAID, TRADE, GIFT = 3.0, 2.7, 1.0
RAID_MAX, MARKET_MAX, GIFTS_MAX = 3.2, 3.6, 2.0
MAX_ITEMS = 6  # items drawn per transfer; any remainder lands with the last one
OFFER_EDGE = {"generous": GOOD, "fair": (84, 90, 106), "greedy": BAD}


def _head(i, lift=0.0):
    def at(t, sc):
        x, y = sc.head(i, t)
        return x, y - lift
    return at


def _feet(i):
    return lambda t, sc: sc.feet(i, t)


def _carry(i):
    return lambda t, sc: sc.carry_point(i, t)


def _pile(i):
    return lambda t, sc: sc.pile_point(i)


def build_day(scene, d: DayLog) -> Timeline:
    return _Day(scene, d).build()


class _Day:
    def __init__(self, sc, d: DayLog):
        self.sc, self.d, self.L, self.cfg = sc, d, sc.layout, sc.cfg
        self.tl = Timeline("day")
        self.rng = random.Random(d.gen * 100_003 + d.day)
        self.alive = [i for i in sorted(d.start) if d.start[i].alive and i in sc.shown]
        self.hp = {i: d.start[i].health for i in self.alive}
        self.fainted: set[int] = set()
        self.ground = None
        for i in self.alive:
            self.tl.track(i, self.home(i))

    def home(self, i):
        return self.L.door(self.sc.slots[i])

    def tr(self, i):
        return self.tl.tracks[i]

    def build(self) -> Timeline:
        self.tl.dark = [(0.0, self.sc.dark), (0.7, 0.0)]
        t = self.morning(0.0)
        t = self.work(t)
        t = self.raids(t)
        t = self.market(t)
        t = self.gifts(t)
        self.night(t)
        return self.tl.seal()

    # ---- morning: every agent shows its plan ------------------------------------------

    def morning(self, t0):
        tl, d = self.tl, self.d
        tl.phase(t0, "Morning")
        tl.add(Banner(f"Day {d.day}", None, t0, t0 + DAWN + 0.5, pos=(620, 22), size=34))
        for k, i in enumerate(self.alive):
            plan = d.plans[i]
            s = t0 + 0.1 + 0.45 * k / max(1, len(self.alive))
            tl.add(Bubble(_head(i), s, t0 + DAWN + 0.5, [plan.activity] + (["far"] if plan.route == "far" else [])))
        tl.at(t0 + 0.35, lambda sc: _show_plans(sc.stats, d))
        return t0 + DAWN

    # ---- work: gathering, resting, the hunt; then everyone walks home ---------------------

    def work(self, t0):
        tl = self.tl
        tl.phase(t0, "Work")
        w0, w1 = t0 + OUT, t0 + OUT + WORK
        self._gathering(t0, w0, w1)
        for i in self.alive:
            activity = self.d.plans[i].activity
            if activity == "rest":
                self._rest(i, t0, w0, w1 + BACK)
            elif activity == "raid":
                self.tr(i).mood(t0 + 0.2, w1 + BACK, "angry")
                tl.add(Emote("raid", _head(i), t0 + 0.3, w1 + BACK, size=6))
        if self.d.hunt:
            self._hunt(self.d.hunt, t0, w0, w1)
        for i in self.alive:
            tr = self.tr(i)
            if i not in self.fainted and tr.end != self.home(i):
                arrive = tr.go(w1 + self.rng.uniform(0.0, 0.2), BACK - 0.2, self.home(i))
                tl.at(arrive, lambda sc, i=i: sc.store(i))
        return w1 + BACK

    def _gathering(self, t0, w0, w1):
        tl, d, rng = self.tl, self.d, self.rng
        by_zone = defaultdict(list)
        for e in d.gathers:
            by_zone[f"{e['resource']}_{e['route']}"].append(e)
        ground = []
        for key, events in by_zone.items():
            zone, turn = self.L.zones[key], rng.uniform(0, math.tau)
            for k, e in enumerate(events):
                i, r, got = e["agent"], e["resource"], e["got"]
                spot = zone.spot(k, len(events), 0.66, turn)
                start = t0 + rng.uniform(0.0, 0.3)
                self.tr(i).go(start, w0 - start, spot)
                self.tr(i).mood(w0, w1, "work")
                for j in range(got):
                    a = turn + k + j * 2.4
                    pos = (spot[0] + 19 * math.cos(a), spot[1] + 8 * math.sin(a) - 2)
                    tp = w0 + 0.12 + (j + 0.5) / got * (WORK - 0.6)
                    ground.append((r, pos, tp))
                    tl.add(Fly(r, pos, _carry(i), tp, tp + 0.28, arc=12, size=3.8))
                    tl.at(tp + 0.28, lambda sc, i=i, r=r: sc.carry(i, r, 1))
                if got:
                    tl.add(Popup(f"+{got}", _head(i), w1 - 0.45, 1.0, GOOD))
                if e["injured"]:
                    ti = w0 + 0.6 * WORK
                    tl.add(Flash(_feet(i), ti, ti + 0.45, BAD))
                    self._hurt(i, ti, self.cfg.gathering.far_injury_dmg)
        for r in ("food", "wood"):
            mine = [e for e in d.gathers if e["resource"] == r]
            supply = self.cfg.gathering.supply_per_capita * len(self.alive) * d.season[r]
            demand = sum(e["raw"] for e in mine)
            tl.at(w0, lambda sc, r=r, s=supply, dm=demand: _set_supply(sc.stats, r, s, dm))
            left = max(0, round(supply - sum(e["got"] for e in mine)))
            for route, share in (("near", 0.55), ("far", 0.45)):
                zone, turn = self.L.zones[f"{r}_{route}"], rng.uniform(0, math.tau)
                m = min(36, round(left * share))
                ground += [(r, zone.spot(k, m, 0.88, turn), None) for k in range(m)]
        self.ground = tl.add(Ground(ground, 0.0, INF))

    def _rest(self, i, t0, w0, end):
        self.tr(i).mood(t0 + 0.2, end, "sleep")
        self.tl.add(Emote("rest", _head(i), t0 + 0.3, end, size=6))
        gain = min(self.cfg.agent.health_max, self.hp[i] + self.cfg.agent.rest_regen) - self.hp[i]
        if gain > 0:
            self.hp[i] += gain
            self.tl.add(Popup(f"+{gain}", _head(i), w0 + 0.5, 1.1, GOOD, heart=True))
            self.tl.at(w0 + 0.5, lambda sc, i=i, g=gain: sc.add_health(i, g))

    def _hunt(self, h, t0, w0, w1):
        tl, L, rng = self.tl, self.L, self.rng
        party = h["party"]
        cx, cy = L.zones["hunt"].center
        prey, caption = (cx, cy + 16), (cx, cy + 104)
        for k, i in enumerate(party):
            start = t0 + rng.uniform(0.0, 0.3)
            self.tr(i).go(start, w0 - start, L.hunt_spot(k, len(party)))
            self.tr(i).look(w0, w1, prey)
        if h["cancelled"]:
            tl.add(Label("Too few hunters: the hunt is called off", caption, w0, w1 + 0.4, MUTED))
            tl.at(w0 + 0.6, lambda sc: setattr(sc.stats, "hunt", h))
            for i in party:
                self._to_carry("food", prey, i, h["payout"], w0 + 0.6, w0 + 1.0)
            return
        workers = set(h["contributors"])
        result = w0 + 1.15
        tl.at(result, lambda sc: setattr(sc.stats, "hunt", h))
        tl.add(Beast(prey, 0.0, w1 + BACK, result, h["success"]))
        for i in party:
            tr = self.tr(i)
            if i in workers:
                tr.mood(w0, result, "work")
                spot, lunge = tr.end, lerp(tr.end, prey, 0.3)
                for k in range(2):
                    tr.go(w0 + 0.05 + k * 0.45, 0.2, lunge, hop=False)
                    tr.go(tr.free, 0.2, spot, hop=False)
                tl.add(Emote("sweat", _head(i), w0 + 0.1, result, size=5, dx=11))
                self._hurt(i, w0 + 0.95, self.cfg.hunt.effort_cost)
            else:
                tr.mood(w0, result, "lazy")
                tl.add(Emote("lazy", _head(i), w0 + 0.1, result, size=6))
        verdict = f"{len(workers)} of {len(party)} hunters worked hard ({h['threshold']} needed)"
        tl.add(Label(verdict, caption, w0 + 0.6, w1 + 0.6))
        below = (caption[0], caption[1] + 27)
        if h["success"]:
            tl.add(Label(f"Success! +{h['payout']} food each", below, result, w1 + 0.6, GOOD))
            tl.add(Flash(prey, result, result + 0.5, GOOD, 70))
            for i in party:
                self.tr(i).mood(result, w1, "happy")
                self._to_carry("food", prey, i, h["payout"], result + 0.1, w1 - 0.1)
        else:
            tl.add(Label("The prey escaped!", below, result, w1 + 0.6, BAD))
            for i in party:
                self.tr(i).mood(result, w1, "sad")

    def _to_carry(self, kind, src, i, n, ta, tb):
        shown = min(n, MAX_ITEMS)
        for j in range(shown):
            s = ta + (tb - ta - 0.3) * j / max(1, shown - 1)
            amount = 1 if j < shown - 1 else n - (shown - 1)
            self.tl.add(Fly(kind, src, _carry(i), s, s + 0.3, arc=28))
            self.tl.at(s + 0.3, lambda sc, i=i, kind=kind, a=amount: sc.carry(i, kind, a))

    def _hurt(self, i, t, dmg):
        if not dmg:
            return
        self.hp[i] -= dmg
        self.tl.add(Popup(f"-{dmg}", _head(i), t, 1.1, BAD, heart=True))
        self.tl.at(t, lambda sc, i=i, dmg=dmg: sc.add_health(i, -dmg))
        if self.hp[i] <= 0:
            self.fainted.add(i)
            self.tr(i).mood(t, INF, "fainted")
        else:
            self.tr(i).mood(t, t + 0.6, "hurt")

    # ---- raids: the raider visits the victim, who fights back or yields --------------------

    def raids(self, t0):
        raids = self.d.raids
        if not raids:
            return t0
        self.tl.phase(t0, "Raids")
        starts = _schedule(raids, lambda e: (e["attacker"], e["target"]), RAID, 0.35)
        k = min(1.0, RAID_MAX / (max(starts) + RAID))
        for e, s in zip(raids, starts):
            self._raid(e, t0 + s * k, k)
        return t0 + (max(starts) + RAID) * k + 0.2

    def _raid(self, e, s, k):
        tl = self.tl
        a, v = e["attacker"], e["target"]
        A, V = self.tr(a), self.tr(v)
        arrive = A.go(s, 0.65 * k, self.L.visit(self.sc.slots[v]))
        A.look(s, arrive + 1.6 * k, v)
        V.look(s + 0.3 * k, arrive + 1.6 * k, a)
        A.mood(s, arrive + 0.9 * k, "angry")
        tl.add(Emote("raid", _head(a), s, arrive + 0.4 * k, size=6))
        decide = arrive + 0.1 * k
        if e["defence"] == "yield":
            tl.add(Bubble(_head(v), decide, decide + 0.75 * k, ["flag"]))
            V.mood(decide, decide + 1.3 * k, "sad")
            hit = decide + 0.45 * k
        else:
            tl.add(Bubble(_head(v), decide, decide + 0.55 * k, ["fight"]))
            tl.add(Dust(lerp(self.home(v), A.end, 0.5), decide + 0.3 * k, decide + 1.1 * k))
            hit = decide + 1.1 * k
            self._hurt(e["dmg_to"], hit, self.cfg.raid.fight_dmg)
        winner, loser = (a, v) if e["attacker_won"] else (v, a)
        dest = _carry(a) if winner == a else _pile(winner)
        for r, n in e["loot"].items():
            self._transfer(r, n, loser, winner, _pile(loser), dest, hit, hit + 0.45 * k, to_carry=winner == a)
        if winner not in self.fainted:
            self.tr(winner).mood(hit, hit + 0.8 * k, "happy")
        if loser not in self.fainted:
            self.tr(loser).mood(hit, hit + 0.8 * k, "sad")
        if a not in self.fainted:
            back = A.go(hit + 0.55 * k, 0.55 * k, self.home(a))
            tl.at(back, lambda sc: sc.store(a))
        tl.at(hit, lambda sc: _add_raid(sc.stats, e))

    def _transfer(self, r, n, src_agent, dst_agent, src, dst, ta, tb, to_carry=False, arc=40, hearts=False):
        """n units of r leave src_agent's store at ta and reach dst_agent (store or carried) at tb."""
        if n <= 0:
            return
        shown = min(n, MAX_ITEMS)
        step = 0.06 if shown > 1 else 0.0
        for j in range(shown):
            amount = 1 if j < shown - 1 else n - (shown - 1)
            a = ta + j * step
            self.tl.add(Fly(r, src, dst, a, max(a + 0.2, tb + j * step), arc=arc, hearts=hearts))
            self.tl.at(a, lambda sc, i=src_agent, r=r, m=amount: sc.stock(i, r, -m))
            land = (lambda sc, i=dst_agent, r=r, m=amount: sc.carry(i, r, m)) if to_carry else \
                   (lambda sc, i=dst_agent, r=r, m=amount: sc.stock(i, r, m))
            self.tl.at(max(a + 0.2, tb + j * step), land)

    # ---- market: the needy proposer meets a partner at a stall ------------------------------

    def market(self, t0):
        trades = self.d.trades
        if not trades:
            return t0
        self.tl.phase(t0, "Market")
        starts, stalls = _schedule_stalls(trades, TRADE, 0.3, len(self.L.stalls))
        k = min(1.0, MARKET_MAX / (max(starts) + TRADE))
        for e, s, stall in zip(trades, starts, stalls):
            self._trade(e, t0 + s * k, stall, k)
        return t0 + (max(starts) + TRADE) * k + 0.2

    def _trade(self, e, s, stall, k):
        tl = self.tl
        p, q, r = e["proposer"], e["responder"], e["resource"]
        o, give, get, ok = OTHER[r], e["give"], e["get"], e["accepted"]
        P, Q = self.tr(p), self.tr(q)
        sp, sq = self.L.stall(stall)
        tl.at(s, lambda sc: (sc.take(p, o, give), sc.take(q, r, get)))
        meet = max(P.go(s, 0.6 * k, sp), Q.go(s, 0.6 * k, sq))
        P.look(meet, meet + 1.6 * k, q)
        Q.look(meet, meet + 1.6 * k, p)
        tl.add(Bubble(_head(p), meet, meet + 1.0 * k, [("n", give), o, ("n", "for", MUTED), ("n", get), r],
                      edge=OFFER_EDGE.get(e["offer"], MUTED)))
        answer = meet + 0.6 * k
        tl.add(Bubble(_head(q, lift=26), answer, answer + 0.65 * k, ["check" if ok else "cross"]))
        leave = answer + 0.6 * k
        if ok:
            swap = answer + 0.4 * k
            for kind, n, frm, to in ((o, give, p, q), (r, get, q, p)):
                tl.at(swap, lambda sc, i=frm, kind=kind, n=n: sc.carry(i, kind, -n))
                for j in range(min(n, MAX_ITEMS)):
                    tl.add(Fly(kind, _carry(frm), _carry(to), swap + j * 0.05, swap + 0.35 * k + j * 0.05, arc=22))
                tl.at(swap + 0.35 * k, lambda sc, i=to, kind=kind, n=n: sc.carry(i, kind, n))
            for i in (p, q):
                self.tr(i).mood(swap + 0.35 * k, swap + 0.9 * k, "happy")
            leave = swap + 0.5 * k
        else:
            P.mood(answer, answer + 0.8 * k, "sad")
        for i in (p, q):
            back = self.tr(i).go(leave, 0.55 * k, self.home(i))
            tl.at(back, lambda sc, i=i: sc.store(i))
        tl.at(answer, lambda sc: _add_trade(sc.stats, ok))

    # ---- gifts: someone with nothing asks a neighbour with plenty ------------------------------

    def gifts(self, t0):
        gifts = self.d.gifts
        if not gifts:
            return t0
        self.tl.phase(t0, "Gifts")
        starts = _schedule(gifts, lambda e: (e["donor"], e["recipient"]), GIFT, 0.12)
        k = min(1.0, GIFTS_MAX / (max(starts) + GIFT))
        for e, s in zip(gifts, starts):
            self._gift(e, t0 + s * k, k)
        return t0 + (max(starts) + GIFT) * k + 0.15

    def _gift(self, e, s, k):
        tl = self.tl
        dn, rc, r = e["donor"], e["recipient"], e["resource"]
        tl.add(Bubble(_head(rc), s, s + 0.55 * k, [r, ("n", "?")]))
        self.tr(rc).look(s, s + 1.0 * k, dn)
        self.tr(dn).look(s + 0.2 * k, s + 1.0 * k, rc)
        if e["given"]:
            a, b = s + 0.35 * k, s + 0.95 * k
            self._transfer(r, 1, dn, rc, _pile(dn), _pile(rc), a, b, arc=70, hearts=True)
            tl.add(Emote("heart", _head(rc), b, b + 0.7 * k))
            self.tr(rc).mood(b, b + 0.6 * k, "happy")
            self.tr(dn).mood(a, b + 0.3 * k, "happy")
        else:
            tl.add(Bubble(_head(dn), s + 0.3 * k, s + 0.85 * k, ["cross"]))
            self.tr(rc).mood(s + 0.5 * k, s + 1.0 * k, "sad")
        tl.at(s + 0.5 * k, lambda sc: _add_gift(sc.stats, e["given"]))

    # ---- night: eat 1 food, burn 1 wood, or lose health; the weakest die ------------------------

    def night(self, t0):
        tl, cfg = self.tl, self.cfg
        tl.phase(t0, "Night")
        tl.dark += [(t0, 0.0), (t0 + 0.6, 1.0)]
        if self.ground:
            self.ground.fade_at = t0
        eat, end = t0 + 0.5, t0 + NIGHT
        need = {r.value: n for r, n in cfg.agent.need.items()}
        hungry = cold = 0
        for e in self.d.nights:
            i = e["agent"]
            if i not in self.hp:
                continue
            hungry += not e["ate"]
            cold += not e["burned"]
            if i not in self.fainted:
                if e["ate"]:
                    tl.add(Fly("food", _pile(i), _head(i), eat, eat + 0.35, arc=10, shrink=True))
                    tl.at(eat, lambda sc, i=i: sc.stock(i, "food", -need["food"]))
                if e["burned"]:
                    fire = self.L.fire(self.sc.slots[i])
                    tl.add(Fly("wood", _pile(i), fire, eat + 0.08, eat + 0.4, arc=14, shrink=True))
                    tl.at(eat + 0.08, lambda sc, i=i: sc.stock(i, "wood", -need["wood"]))
                    tl.add(Campfire(fire, eat + 0.35, end))
                missing = [kind for kind, ok in (("hungry", e["ate"]), ("cold", e["burned"])) if not ok]
                for j, kind in enumerate(missing):
                    tl.add(Emote(kind, _head(i), eat + 0.3, end, size=6, dx=(j - (len(missing) - 1) / 2) * 15))
                if missing:
                    self.tr(i).mood(eat + 0.3, end, "sad")
            delta = e["health_after"] - self.hp[i]
            if delta < 0 and i not in self.fainted:
                tl.add(Popup(f"{delta}", _head(i), eat + 0.45, 1.0, BAD, heart=True))
            tl.at(eat + 0.45, lambda sc, i=i, h=e["health_after"]: sc.set_health(i, h))
            self.hp[i] = e["health_after"]
        for e in self.d.deaths:
            i, td = e["agent"], t0 + 1.05
            tl.add(Ghost(_feet(i), td, td + 1.3, self.L.blob_r))
            tl.at(td, lambda sc, i=i, c=e["cause"]: sc.die(i, c))
        tl.at(eat + 0.45, lambda sc: _set_night(sc.stats, hungry, cold))
        return end


# ---- generation changes ------------------------------------------------------------------

def build_gen_start(sc, item: GenStart, parents_at: dict, gap: int) -> Timeline:
    """The survivors of the previous generation split into the new villagers, who walk to their homes."""
    tl = Timeline("gen_start")
    tl.phase(0.0, "New generation")
    tl.dark = [(0.0, sc.dark), (0.6, 0.0)]
    rng = random.Random(item.gen)
    if item.gen == 0:
        sub = "the first villagers, with random genes"
    elif gap > 1:
        sub = f"{gap - 1} generations later"
    else:
        sub = "children of the survivors"
    tl.add(Banner(f"Generation {item.gen + 1}", sub, 0.1, 3.0, pos=(620, 384)))
    split = 0.9 if parents_at else 0.3
    for i, pos in parents_at.items():
        tr = tl.track(i, pos)
        tr.mood(0.0, split, "happy")
        tr.scales = [(0.0, 1.0), (split, 1.0), (split + 0.2, 1.3), (split + 0.4, 0.0)]
    ids = sorted(item.agents)
    for k, i in enumerate(ids):
        home = sc.layout.door(sc.slots[i])
        parent = next((p for p in item.agents[i].parents if p in parents_at), None)
        start = home if parent is None else (parents_at[parent][0] + rng.uniform(-9, 9),
                                             parents_at[parent][1] + rng.uniform(-5, 5))
        tr = tl.track(i, start)
        pop = split + 0.25 + 0.7 * k / max(1, len(ids))
        tr.scales = [(0.0, 0.0), (pop, 0.0), (pop + 0.18, 1.25), (pop + 0.32, 1.0)]
        if parent is not None:
            tr.go(pop + 0.4, 0.9, home)
    return tl.seal(3.2)


def build_gen_end(sc, item: GenEnd) -> Timeline:
    """Selection: the survivors celebrate; they will be the parents of the next generation."""
    tl = Timeline("gen_end")
    tl.phase(0.0, "Selection")
    tl.dark = [(0.0, sc.dark), (0.6, 0.0)]
    rng = random.Random(item.gen)
    alive = [i for i, s in sc.shown.items() if s.alive and not s.leaving]
    tl.add(Banner(f"Generation {item.gen + 1} is over",
                  f"{len(alive)} of {sc.cfg.population.N} survived: their genes go on", 0.2, 2.9, pos=(620, 384)))
    for i in alive:
        tr = tl.track(i, sc.layout.door(sc.slots[i]))
        tr.mood(0.5, 2.7, "happy")
        for k in range(2):
            tr.go(0.6 + k * 0.55 + rng.uniform(0, 0.15), 0.4, tr.end, hops=1)
    return tl.seal(3.0)


# ---- effects specific to the village -----------------------------------------------------------

class Ground(Effect):
    """Food and wood lying in the fields and forests until someone picks them up; they shrink away at night."""
    layer = "ground"

    def __init__(self, items, t0, t1):
        super().__init__(t0, t1)
        self.items, self.fade_at = items, INF

    def draw(self, cv, t, scene):
        size = 4.0 * ease((t - self.t0) / 0.5) * (1 - clamp01((t - self.fade_at) / 0.6))
        if size <= 0.2:
            return
        for kind, (x, y), taken in self.items:
            if taken is None or t < taken:
                sprites.item(cv, kind, x, y, size)


class Beast(Effect):
    """The hunt's prey: grazes, struggles, then falls (success) or runs off (failure)."""
    layer = "ground"

    def __init__(self, pos, t0, t1, t_result, success):
        super().__init__(t0, t1)
        self.pos, self.t_result, self.success = pos, t_result, success

    def draw(self, cv, t, scene):
        x, y = self.pos
        state = "idle"
        if t >= self.t_result:
            if self.success:
                state = "down"
            else:
                u = (t - self.t_result) / 0.9
                if u >= 1:
                    return
                state, x, y = "flee", x + 330 * ease(u), y - 40 * ease(u)
        elif t > self.t_result - 0.9:
            state, x = "struggle", x + 3 * math.sin(t * 45)
        sprites.beast(cv, x, y, t, state)


class Campfire(Effect):
    layer = "glow"

    def __init__(self, pos, t0, t1):
        super().__init__(t0, t1)
        self.pos = pos

    def draw(self, cv, t, scene):
        sprites.campfire(cv, *self.pos, t, ease((t - self.t0) / 0.3))


# ---- helpers ----------------------------------------------------------------------------

def _schedule(events, who, length, gap) -> list[float]:
    """Start times: in log order, at least `gap` apart, never two at once for the same agent."""
    free, last, starts = defaultdict(float), -gap, []
    for e in events:
        s = max([last + gap] + [free[i] for i in who(e)])
        for i in who(e):
            free[i] = s + length
        starts.append(s)
        last = s
    return starts


def _schedule_stalls(trades, length, gap, n_stalls) -> tuple[list[float], list[int]]:
    free, stall_free, last = defaultdict(float), [0.0] * n_stalls, -gap
    starts, stalls = [], []
    for e in trades:
        who = (e["proposer"], e["responder"])
        k = min(range(n_stalls), key=lambda j: stall_free[j])
        s = max(last + gap, stall_free[k], *(free[i] for i in who))
        for i in who:
            free[i] = s + length
        stall_free[k] = s + length - 0.4
        starts.append(s)
        stalls.append(k)
        last = s
    return starts, stalls


def _show_plans(stats, d: DayLog):
    stats.plans.clear()
    stats.plans.update(p.activity for p in d.plans.values())
    stats.far = sum(p.route == "far" for p in d.plans.values())
    stats.revealed = True


def _set_supply(stats, r, supply, demand):
    stats.supply[r], stats.demand[r] = supply, demand


def _add_raid(stats, e):
    stats.raids += 1
    stats.fights += e["defence"] == "fight"
    stats.raids_won += e["attacker_won"]


def _add_trade(stats, accepted):
    stats.offers += 1
    stats.accepted += accepted


def _add_gift(stats, given):
    stats.gift_asks += 1
    stats.gifts += given


def _set_night(stats, hungry, cold):
    stats.night, stats.hungry, stats.cold = True, hungry, cold
