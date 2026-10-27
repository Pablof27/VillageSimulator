# Animation engine: per-agent motion tracks, timed effects and scheduled state changes.
# Times are seconds of animation at 1x speed, measured from the start of the timeline.
import math
from dataclasses import dataclass

from viewer import sprites
from viewer.layout import MAP_W
from viewer.style import INK, PAPER, WHITE, shade

INF = math.inf
HOP_HEIGHT = 9.0


def clamp01(u: float) -> float:
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def ease(u: float) -> float:
    u = clamp01(u)
    return u * u * (3 - 2 * u)


def lerp(a, b, u: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def point(p, t, scene) -> tuple[float, float]:
    """A fixed point, or a callable `(t, scene) -> point` that follows an agent."""
    return p(t, scene) if callable(p) else p


def keyframes(keys: list[tuple[float, float]], t: float, default: float) -> float:
    if not keys:
        return default
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t < t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0) if t1 > t0 else 1.0)
    return keys[-1][1]


@dataclass(slots=True)
class Move:
    t0: float
    t1: float
    p0: tuple[float, float]
    p1: tuple[float, float]
    hops: int


class Track:
    """Where one blob is, where it looks, how it feels and how big it is over a timeline."""

    def __init__(self, pos):
        self.start = self.end = pos
        self.free = 0.0
        self.moves: list[Move] = []
        self.moods: list[tuple[float, float, str]] = []
        self.looks: list[tuple[float, float, object]] = []
        self.scales: list[tuple[float, float]] = []

    def go(self, t0: float, dur: float, dest, hop=True, hops: int | None = None) -> float:
        """Moves to `dest` (hopping) from max(t0, end of the previous move); returns the arrival time."""
        t0 = max(t0, self.free)
        if hops is None:
            dist = math.dist(self.end, dest)
            hops = max(1, round(dist / 50)) if hop and dist > 3 else 0
        self.moves.append(Move(t0, t0 + dur, self.end, dest, hops))
        self.end, self.free = dest, t0 + dur
        return self.free

    def mood(self, t0: float, t1: float, kind: str) -> None:
        self.moods.append((t0, t1, kind))

    def look(self, t0: float, t1: float, target) -> None:
        """Look at an agent (id) or a point."""
        self.looks.append((t0, t1, target))

    def sample(self, t: float):
        """(position, hop phase 0-1, unit heading or None)."""
        pos, hop, heading = self.start, 0.0, None
        for m in self.moves:
            if t < m.t0:
                break
            if t >= m.t1:
                pos = m.p1
                continue
            u = (t - m.t0) / (m.t1 - m.t0)
            pos = lerp(m.p0, m.p1, ease(u))
            hop = abs(math.sin(math.pi * m.hops * u)) if m.hops else 0.0
            d = math.dist(m.p0, m.p1)
            heading = ((m.p1[0] - m.p0[0]) / d, (m.p1[1] - m.p0[1]) / d) if d > 0.5 else (0.0, 0.0)
            break
        return pos, hop, heading

    def mood_at(self, t: float) -> str | None:
        """The most recently started mood; fainting overrides everything."""
        best = None
        for t0, t1, kind in self.moods:
            if t0 <= t < t1:
                if kind == "fainted":
                    return kind
                if best is None or t0 >= best[0]:
                    best = (t0, kind)
        return best[1] if best else None

    def look_at(self, t: float):
        """An agent id or a point to look at, if any."""
        for t0, t1, target in reversed(self.looks):
            if t0 <= t < t1:
                return target
        return None

    def scale_at(self, t: float) -> float:
        return keyframes(self.scales, t, 1.0)


class Effect:
    layer = "over"  # ground (under the blobs) | glow (after the night shade) | over (on top)

    def __init__(self, t0: float, t1: float):
        self.t0, self.t1 = t0, t1

    def draw(self, cv, t: float, scene) -> None:
        raise NotImplementedError


class Timeline:
    def __init__(self, kind: str):
        self.kind = kind  # day | gen_start | gen_end
        self.tracks: dict[int, Track] = {}
        self.effects: list[Effect] = []
        self.phases: list[tuple[float, str]] = []
        self.dark: list[tuple[float, float]] = []
        self.duration = 0.0
        self._changes: list[tuple[float, int, object]] = []
        self._done = 0

    def track(self, agent: int, pos) -> Track:
        self.tracks[agent] = Track(pos)
        return self.tracks[agent]

    def add(self, effect: Effect) -> Effect:
        self.effects.append(effect)
        return effect

    def at(self, t: float, change) -> None:
        """Schedules `change(scene)` at time t."""
        self._changes.append((t, len(self._changes), change))

    def phase(self, t: float, name: str) -> None:
        self.phases.append((t, name))

    def seal(self, min_duration=0.0) -> "Timeline":
        self._changes.sort(key=lambda c: (c[0], c[1]))
        ends = [e.t1 for e in self.effects if e.t1 < INF] + [tr.free for tr in self.tracks.values()]
        ends += [c[0] for c in self._changes] + [min_duration]
        self.duration = max(ends)
        return self

    def advance(self, t: float, scene) -> None:
        changes = self._changes
        while self._done < len(changes) and changes[self._done][0] <= t:
            changes[self._done][2](scene)
            self._done += 1

    def phase_at(self, t: float) -> str | None:
        name = None
        for t0, n in self.phases:
            if t >= t0:
                name = n
        return name

    def darkness(self, t: float) -> float:
        return keyframes(self.dark, t, 0.0)


# ---- generic effects ---------------------------------------------------------------

class Fly(Effect):
    """A resource item flying along an arc between two (possibly moving) points."""

    def __init__(self, kind, src, dst, t0, t1, arc=34.0, size=4.5, shrink=False, hearts=False):
        super().__init__(t0, t1)
        self.kind, self.src, self.dst, self.arc, self.size = kind, src, dst, arc, size
        self.shrink, self.hearts = shrink, hearts

    def draw(self, cv, t, scene):
        u = clamp01((t - self.t0) / (self.t1 - self.t0))
        e = ease(u)
        a, b = point(self.src, t, scene), point(self.dst, t, scene)
        x, y = lerp(a, b, e)
        y -= self.arc * 4 * e * (1 - e)
        if self.hearts and 0.1 < u < 0.95:
            sprites.heart(cv, x - 10 * (1 - u), y + 6, 3.2, alpha=int(200 * (1 - u)))
        sprites.item(cv, self.kind, x, y, self.size * (1 - 0.7 * u if self.shrink else 1))


class Popup(Effect):
    """Text rising from a point and fading out, e.g. '-2' over a hurt blob."""

    def __init__(self, text, at, t0, dur=1.1, color=INK, size=16.0, heart=False):
        super().__init__(t0, t0 + dur)
        self.text, self.at, self.color, self.size, self.heart = text, at, color, size, heart

    def draw(self, cv, t, scene):
        u = clamp01((t - self.t0) / (self.t1 - self.t0))
        x, y = point(self.at, self.t0, scene)
        y -= 6 + 24 * ease(min(1.0, u * 1.6))
        alpha = 255 if u < 0.65 else int(255 * (1 - u) / 0.35)
        w, _ = cv.measure(self.text, self.size, bold=True)
        if self.heart:
            x -= 6
            sprites.heart(cv, x + w / 2 + 8, y + 1, 4.4)
        outlined(cv, self.text, (x, y), self.size, self.color, alpha)


class Bubble(Effect):
    """A speech bubble over an agent with a row of parts: icon names, ('text', s, colour) or ('n', count)."""

    def __init__(self, at, t0, t1, parts, edge=(84, 90, 106), size=7.5):
        super().__init__(t0, t1)
        self.at, self.parts, self.edge, self.size = at, parts, edge, size

    def draw(self, cv, t, scene):
        pop = min(ease((t - self.t0) / 0.14), ease((self.t1 - t) / 0.1))
        if pop <= 0.05:
            return
        s = self.size * (0.6 + 0.4 * pop)
        widths = [_part_width(cv, p, s) for p in self.parts]
        w, h = sum(widths) + 0.9 * s * (len(widths) + 1), 2.8 * s
        x, y = point(self.at, t, scene)
        x0, y0 = sprites.bubble(cv, x, y, w, h, edge=self.edge)
        cx = x0 + 0.9 * s
        for p, pw in zip(self.parts, widths):
            _draw_part(cv, p, cx + pw / 2, y0 + h / 2, s)
            cx += pw + 0.9 * s


def _part_width(cv, p, s) -> float:
    if isinstance(p, tuple):
        return cv.measure(str(p[1]), 1.9 * s, bold=True)[0]
    return {"wood": 2.0, "gather_wood": 2.0, "raid": 2.6, "far": 1.9}.get(p, 1.8) * s


def _draw_part(cv, p, x, y, s):
    if isinstance(p, tuple):
        color = p[2] if len(p) > 2 else INK
        cv.text(str(p[1]), (x, y), 1.9 * s, color, bold=True, anchor="center")
    else:
        sprites.icon(cv, p, x, y, s)


class Emote(Effect):
    """A small icon bobbing over an agent (zzz, sweat, heart, ...)."""

    def __init__(self, kind, at, t0, t1, size=6.5, dx=0.0):
        super().__init__(t0, t1)
        self.kind, self.at, self.size, self.dx = kind, at, size, dx

    def draw(self, cv, t, scene):
        pop = min(ease((t - self.t0) / 0.15), ease((self.t1 - t) / 0.15))
        if pop <= 0.05:
            return
        x, y = point(self.at, t, scene)
        sprites.icon(cv, self.kind, x + self.dx, y - 6 + 2 * math.sin(t * 6), self.size * pop)


class Banner(Effect):
    """A big title over the map (day number, generation changes)."""

    def __init__(self, title, subtitle, t0, t1, pos=(MAP_W / 2, 62), size=40.0):
        super().__init__(t0, t1)
        self.title, self.subtitle, self.pos, self.size = title, subtitle, pos, size

    def draw(self, cv, t, scene):
        alpha = int(255 * min(ease((t - self.t0) / 0.25), ease((self.t1 - t) / 0.35)))
        if alpha <= 0:
            return
        x, y = self.pos
        w, h = cv.measure(self.title, self.size, bold=True)
        if self.subtitle:
            w = max(w, cv.measure(self.subtitle, 0.45 * self.size)[0])
        box_h = h + (0.62 * self.size if self.subtitle else 0) + 14
        cv.rect(PAPER, (x - w / 2 - 22, y - 8, w + 44, box_h), radius=16, alpha=int(alpha * 0.88))
        cv.text(self.title, (x, y - 2), self.size, INK, bold=True, anchor="midtop", alpha=alpha)
        if self.subtitle:
            cv.text(self.subtitle, (x, y + h - 4), 0.45 * self.size, shade(INK, 1.5), anchor="midtop", alpha=alpha)


class Label(Effect):
    """A short caption on a pill at a fixed point (e.g. the hunt result)."""

    def __init__(self, text, pos, t0, t1, color=INK, size=15.0, fill=PAPER):
        super().__init__(t0, t1)
        self.text, self.pos, self.color, self.size, self.fill = text, pos, color, size, fill

    def draw(self, cv, t, scene):
        alpha = int(255 * min(ease((t - self.t0) / 0.2), ease((self.t1 - t) / 0.3)))
        if alpha <= 0:
            return
        w, h = cv.measure(self.text, self.size, bold=True)
        x, y = self.pos
        cv.rect(self.fill, (x - w / 2 - 10, y - h / 2 - 3, w + 20, h + 6), radius=h / 2 + 3, alpha=int(alpha * 0.92))
        cv.text(self.text, (x, y), self.size, self.color, bold=True, anchor="center", alpha=alpha)


class Flash(Effect):
    """An expanding ring (injury, success)."""

    def __init__(self, at, t0, t1, color, radius=26.0):
        super().__init__(t0, t1)
        self.at, self.color, self.radius = at, color, radius

    def draw(self, cv, t, scene):
        u = clamp01((t - self.t0) / (self.t1 - self.t0))
        x, y = point(self.at, t, scene)
        cv.circle(self.color, (x, y), 6 + self.radius * ease(u), width=3, alpha=int(230 * (1 - u)))


class Dust(Effect):
    """A cartoon fight cloud."""

    def __init__(self, pos, t0, t1):
        super().__init__(t0, t1)
        self.pos = pos

    def draw(self, cv, t, scene):
        u = clamp01((t - self.t0) / (self.t1 - self.t0))
        x, y = self.pos
        grow = min(1.0, u * 5) * (1 - max(0.0, u - 0.8) * 5)
        for k in range(7):
            a = k * 0.9 + t * 9
            r = (11 + 4 * math.sin(t * 13 + k)) * grow
            cv.circle((214, 210, 204), (x + 16 * math.cos(a) * grow, y - 16 + 9 * math.sin(a) * grow), r)
        for k in range(3):
            a = t * 7 + k * 2.1
            sprites.burst(cv, x + 22 * math.cos(a) * grow, y - 18 + 14 * math.sin(a) * grow, 6 * grow,
                          (255, 220, 90), (240, 120, 50), spikes=5, turn=t * 5)


class Ghost(Effect):
    def __init__(self, at, t0, t1, radius):
        super().__init__(t0, t1)
        self.at, self.radius = at, radius

    def draw(self, cv, t, scene):
        u = clamp01((t - self.t0) / (self.t1 - self.t0))
        x, y = point(self.at, self.t0, scene)
        sprites.ghost(cv, x + 6 * math.sin(u * 9), y - 70 * ease(u), self.radius, int(225 * (1 - u ** 2)))


def outlined(cv, text, pos, size, color, alpha=255, anchor="center"):
    x, y = pos
    for dx, dy in ((-1.3, 0), (1.3, 0), (0, -1.3), (0, 1.3)):
        cv.text(text, (x + dx, y + dy), size, WHITE, bold=True, anchor=anchor, alpha=alpha)
    cv.text(text, (x, y), size, color, bold=True, anchor=anchor, alpha=alpha)
