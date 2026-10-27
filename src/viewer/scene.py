# The village as shown at the current animation time: who is alive, what each agent stores and carries, and how
# each blob looks (genes -> colours and accessories). The running timeline moves it; `draw` paints the map.
import math
from collections import Counter
from dataclasses import dataclass, field

import pygame

from viewer import sprites
from viewer.canvas import Canvas
from viewer.choreography import build_day, build_gen_end, build_gen_start
from viewer.layout import H, MAP_W, Layout
from viewer.replay import AgentInfo, DayLog, GenEnd, GenStart
from viewer.style import GENES, INK, NIGHT, WHITE, level, mix, ramp
from viewer.timeline import HOP_HEIGHT, Timeline

FACES = {"sleep": "sleep", "happy": "happy", "sad": "sad", "hurt": "hurt", "fainted": "fainted", "angry": "angry",
         "lazy": "lazy"}


@dataclass
class Shown:
    health: int
    food: int
    wood: int
    alive: bool = True
    carry_food: int = 0
    carry_wood: int = 0
    leaving: bool = False  # previous generation, during the hand-over to the next one


@dataclass
class Look:
    body: tuple
    belly: tuple
    halo: int
    brows: int
    phase: float


@dataclass
class DayStats:
    plans: Counter = field(default_factory=Counter)
    far: int = 0
    revealed: bool = False
    hunt: dict | None = None
    raids: int = 0
    fights: int = 0
    raids_won: int = 0
    offers: int = 0
    accepted: int = 0
    gift_asks: int = 0
    gifts: int = 0
    night: bool = False
    hungry: int = 0
    cold: int = 0
    died: int = 0
    supply: dict = field(default_factory=dict)
    demand: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)


@dataclass
class History:
    alive: list[int] = field(default_factory=list)  # alive at the start, then at the end of each day
    deaths: Counter = field(default_factory=Counter)
    start_counts: dict = field(default_factory=dict)  # gene -> agents per level at the start of the generation


class Scene:
    def __init__(self, cfg, color_by="aggression"):
        self.cfg = cfg
        self.layout = Layout(cfg.population.N)
        self.color_by = color_by
        self.agents: dict[int, AgentInfo] = {}
        self.looks: dict[int, Look] = {}
        self.slots: dict[int, int] = {}
        self.shown: dict[int, Shown] = {}
        self.graves: dict[int, float] = {}
        self.timeline: Timeline | None = None
        self.item = None
        self.t = 0.0
        self.clock = 0.0
        self.dark = 0.0
        self.gen: int | None = None
        self.day = 0
        self.last_played_gen: int | None = None
        self.stats = DayStats()
        self.history = History()
        self.means: dict[int, dict[str, float]] = {}
        self._bg: tuple[float, pygame.Surface] | None = None

    # ---- items from the event log ---------------------------------------------------

    def start_generation(self, item: GenStart, animate=True) -> None:
        parents_at = {i: self.layout.door(self.slots[i]) for i, s in self.shown.items()
                      if s.alive and not s.leaving} if animate else {}
        gap = item.gen - self.last_played_gen if self.last_played_gen is not None else 1
        old = self.shown
        self.shown = {}
        for i in parents_at:
            self.shown[i] = old[i]
            old[i].leaving, old[i].carry_food, old[i].carry_wood = True, 0, 0
        self.agents = {i: a for i, a in self.agents.items() if i in parents_at}
        self.slots = {i: k for i, k in self.slots.items() if i in parents_at}
        for k, i in enumerate(sorted(item.agents)):
            self.agents[i] = item.agents[i]
            self.slots[i] = k
            v = item.state[i]
            self.shown[i] = Shown(v.health, v.food, v.wood, v.alive)
        self.looks = {i: self._look(a) for i, a in self.agents.items()}
        self.gen, self.day, self.graves = item.gen, 0, {}
        self.stats = DayStats()
        self.history = History(alive=[self.n_alive()], start_counts=self.level_counts())
        self.last_played_gen = item.gen
        if animate:
            self._play(build_gen_start(self, item, parents_at, gap), item)
        else:
            self._drop_leaving()

    def play_day(self, item: DayLog) -> None:
        self.day = item.day
        self.stats = DayStats(season=item.season)
        self._play(build_day(self, item), item)

    def skip_day(self, item: DayLog) -> None:
        self.day = item.day
        self.stats = DayStats(season=item.season)
        self._finish_day(item)

    def end_generation(self, item: GenEnd, animate=True) -> None:
        if animate:
            self._play(build_gen_end(self, item), item)

    def end_timeline(self) -> None:
        tl, item = self.timeline, self.item
        tl.advance(math.inf, self)
        self.dark = tl.darkness(tl.duration)
        self.timeline, self.item, self.t = None, None, 0.0
        if tl.kind == "day":
            self._finish_day(item)
        elif tl.kind == "gen_start":
            self._drop_leaving()

    def _play(self, timeline: Timeline, item) -> None:
        self.timeline, self.item, self.t = timeline, item, 0.0

    def _finish_day(self, d: DayLog) -> None:
        causes = {e["agent"]: e["cause"] for e in d.deaths}
        for i, v in d.end.items():
            s = self.shown.get(i)
            if s is None:
                continue
            if not v.alive and s.alive:
                self.die(i, causes.get(i, "starvation"))
            s.health, s.food, s.wood, s.alive = v.health, v.food, v.wood, v.alive
            s.carry_food = s.carry_wood = 0
        self.history.alive.append(self.n_alive())

    def _drop_leaving(self) -> None:
        for i in [i for i, s in self.shown.items() if s.leaving]:
            del self.shown[i], self.agents[i], self.looks[i], self.slots[i]

    # ---- state changes (called by timelines) ---------------------------------------------

    def carry(self, i: int, r: str, n: int) -> None:
        s = self.shown[i]
        setattr(s, f"carry_{r}", max(0, getattr(s, f"carry_{r}") + n))

    def stock(self, i: int, r: str, n: int) -> None:
        s = self.shown[i]
        setattr(s, r, max(0, getattr(s, r) + n))

    def take(self, i: int, r: str, n: int) -> None:
        """Takes n of resource r from the agent's store to carry it."""
        self.stock(i, r, -n)
        self.carry(i, r, n)

    def store(self, i: int) -> None:
        s = self.shown[i]
        s.food, s.wood = s.food + s.carry_food, s.wood + s.carry_wood
        s.carry_food = s.carry_wood = 0

    def add_health(self, i: int, n: int) -> None:
        self.shown[i].health += n

    def set_health(self, i: int, h: int) -> None:
        self.shown[i].health = h

    def die(self, i: int, cause: str) -> None:
        s = self.shown.get(i)
        if s is None or not s.alive:
            return
        s.alive = False
        s.food = s.wood = s.carry_food = s.carry_wood = 0
        self.graves[self.slots[i]] = self.clock
        self.history.deaths[cause] += 1
        self.stats.died += 1

    # ---- queries -------------------------------------------------------------------

    def n_alive(self) -> int:
        return sum(s.alive and not s.leaving for s in self.shown.values())

    def level_counts(self) -> dict[str, list[int]]:
        counts = {g: [0] * 5 for g in GENES}
        for i, s in self.shown.items():
            if s.alive and not s.leaving:
                for g in GENES:
                    counts[g][level(self.agents[i].genome[g])] += 1
        return counts

    def living_means(self) -> dict[str, float] | None:
        alive = [self.agents[i].genome for i, s in self.shown.items() if s.alive and not s.leaving]
        return {g: sum(a[g] for a in alive) / len(alive) for g in GENES} if alive else None

    def set_color_by(self, gene: str) -> None:
        self.color_by = gene
        self.looks = {i: self._look(a) for i, a in self.agents.items()}

    def _look(self, info: AgentInfo) -> Look:
        g = info.genome
        return Look(ramp(self.color_by, g[self.color_by]), ramp("specialization", g["specialization"]),
                    level(g["altruism"]), level(g["risk"]), (info.id * 0.618034) % 1.0)

    def pose(self, i: int, t: float):
        """(feet position, hop phase, heading, scale, track)."""
        tr = self.timeline.tracks.get(i) if self.timeline else None
        if tr is None:
            return self.layout.door(self.slots[i]), 0.0, None, 1.0, None
        pos, hop, heading = tr.sample(t)
        return pos, hop, heading, tr.scale_at(t), tr

    def radius(self, i: int) -> float:
        h = self.shown[i].health / self.cfg.agent.health_max
        return self.layout.blob_r * (0.78 + 0.22 * min(1.0, max(0.0, h)))

    def feet(self, i: int, t: float):
        return self.pose(i, t)[0]

    def _top(self, i: int, t: float):
        (x, y), hop, _, scale, _ = self.pose(i, t)
        return x, y - hop * HOP_HEIGHT - 2 * self.radius(i) * scale - (9 if self.looks[i].halo else 3)

    def carry_point(self, i: int, t: float):
        x, y = self._top(i, t)
        s = self.shown[i]
        n = min(6, s.carry_food + s.carry_wood)
        return x + (n % 3 - 1) * 8, y - 6 - (n // 3) * 7

    def head(self, i: int, t: float):
        """Anchor above the blob and whatever it carries (bubbles, popups)."""
        x, y = self._top(i, t)
        s = self.shown.get(i)
        n = min(6, s.carry_food + s.carry_wood) if s else 0
        return x, y - (3 + 7 * ((n + 2) // 3) if n else 0)

    def pile_point(self, i: int):
        return self.layout.pile(self.slots[i])

    def agent_at(self, p, radius=22.0) -> int | None:
        best, best_d = None, radius
        for i, s in self.shown.items():
            if not s.alive or s.leaving:
                continue
            (x, y), hop, _, scale, _ = self.pose(i, self.t)
            d = math.dist(p, (x, y - hop * HOP_HEIGHT - self.radius(i) * scale))
            if d < best_d:
                best, best_d = i, d
        return best

    # ---- drawing -------------------------------------------------------------------

    def draw(self, cv: Canvas, hover: int | None = None) -> None:
        t, layout = self.t, self.layout
        self._background(cv)
        for slot, t0 in self.graves.items():
            sprites.grave(cv, *layout.door(slot), alpha=int(255 * min(1.0, (self.clock - t0) / 0.6)))
        for i, s in self.shown.items():
            if s.alive and not s.leaving:
                sprites.pile(cv, *layout.pile(self.slots[i]), s.food, s.wood)
        effects = [e for e in self.timeline.effects if e.t0 <= t < e.t1] if self.timeline else []
        for e in effects:
            if e.layer == "ground":
                e.draw(cv, t, self)
        actors = []
        for i, s in self.shown.items():
            if s.alive:
                pos, hop, heading, scale, tr = self.pose(i, t)
                if scale > 0.02:
                    actors.append((pos[1], i, pos, hop, heading, scale, tr))
        actors.sort(key=lambda a: (a[0], a[1]))
        for _, i, pos, hop, _, scale, _ in actors:
            sprites.shadow(cv, *pos, self.radius(i) * scale, hop * HOP_HEIGHT)
        for _, i, pos, hop, heading, scale, tr in actors:
            self._draw_blob(cv, i, pos, hop, heading, scale, tr, i == hover)
        dark = self.timeline.darkness(t) if self.timeline else self.dark
        if dark > 0.01:
            tint = mix(WHITE, NIGHT, 0.6 * dark)
            cv.surface.fill(tint, (0, 0, math.ceil(MAP_W * cv.k), math.ceil(H * cv.k)), special_flags=pygame.BLEND_MULT)
        for layer in ("glow", "over"):
            for e in effects:
                if e.layer == layer:
                    e.draw(cv, t, self)

    def _draw_blob(self, cv, i, pos, hop, heading, scale, tr, highlight) -> None:
        look, s, t = self.looks[i], self.shown[i], self.t
        mood = tr.mood_at(t) if tr else None
        r = self.radius(i) * scale
        lift = hop * HOP_HEIGHT
        ph = look.phase * math.tau
        if mood == "fainted":
            sx, sy = 1.3, 0.56
        elif heading is not None:
            sy = 0.9 + 0.2 * hop
            sx = 1 / math.sqrt(sy)
        elif mood == "work":
            b = abs(math.sin(self.clock * 10 + ph))
            sy, lift = 0.94 + 0.08 * b, lift + 2.5 * b
            sx = 1 / math.sqrt(sy)
        else:
            sy = 1 + 0.03 * math.sin(self.clock * 2.3 + ph)
            sx = 1 / math.sqrt(sy)
        gaze = self._gaze(i, tr, heading, t, ph)
        blink = (self.clock + look.phase * 5) % 3.9 < 0.11
        body = mix(look.body, (176, 176, 180), 0.55) if mood == "fainted" else look.body
        x, y = pos
        sprites.blob(cv, x, y, r, body, belly=look.belly, halo=look.halo, brows=look.brows, look=gaze, sx=sx, sy=sy,
                     lift=lift, face=FACES.get(mood, "normal"), blink=blink, ring=INK if highlight else None)
        n = s.carry_food + s.carry_wood
        if n and not s.leaving:
            top = y - lift - 2 * r * sy - (9 if look.halo else 3)
            kinds = ["food"] * s.carry_food + ["wood"] * s.carry_wood
            for k, kind in enumerate(kinds[:6]):
                sprites.item(cv, kind, x + (k % 3 - 1) * 8, top - 6 - (k // 3) * 7, 3.6)
            if n > 6:
                cv.text(f"+{n - 6}", (x + 15, top - 12), 11, INK, bold=True, anchor="midleft")

    def _gaze(self, i, tr, heading, t, ph):
        if heading is not None:
            return heading[0], heading[1] * 0.6
        target = tr.look_at(t) if tr else None
        if target is not None:
            tx, ty = target if isinstance(target, tuple) else self.feet(target, t)
            x, y = self.feet(i, t)
            d = math.hypot(tx - x, ty - y)
            if d > 1:
                return (tx - x) / d, (ty - y) / d * 0.6
        return 0.55 * math.sin(self.clock * 0.5 + ph * 3), 0.25 * math.sin(self.clock * 0.37 + ph)

    def _background(self, cv: Canvas) -> None:
        if self._bg is None or self._bg[0] != cv.k:
            surf = pygame.Surface((math.ceil(MAP_W * cv.k), math.ceil(H * cv.k)))
            sprites.background(Canvas(surf, cv.k), self.layout)
            self._bg = (cv.k, surf)
        cv.surface.blit(self._bg[1], (0, 0))
