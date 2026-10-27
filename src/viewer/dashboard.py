# The right-hand panel: header, gene bar charts, today's choices and outcomes, resources and scarcity, and the
# alive / average-gene charts. Also the blob legend under the map, the hover card and the help overlay.
from dataclasses import dataclass

from viewer import sprites
from viewer.canvas import Canvas
from viewer.layout import H, MAP_W, W
from viewer.replay import DayLog
from viewer.style import (ACCESSORY, ACTIVITIES, ACTIVITY_COLOR, ACTIVITY_LABEL, BAD, DEATH_COLOR, DEATH_LABEL, ENDS,
                          FAINT, GENES, GOOD, INK, LEVELS, MUTED, PANEL, PAPER, PHASES, TITLE, WARN, WHITE, level,
                          ramp, shade)

X0 = MAP_W + 14
CW = W - MAP_W - 28


@dataclass
class Status:
    run: str
    mode: str  # live | replay | video
    speed: float = 1.0
    paused: bool = False
    waiting: bool = False
    skipping: bool = False


class Dashboard:
    def __init__(self):
        self._smooth: dict = {}

    def _ease(self, key, target: float, dt: float, rate=7.0) -> float:
        v = self._smooth.get(key, target)
        v += (target - v) * min(1.0, dt * rate)
        self._smooth[key] = v
        return v

    def draw(self, cv: Canvas, sc, status: Status, dt: float) -> None:
        cv.rect(PANEL, (MAP_W, 0, W - MAP_W, H))
        y = self._header(cv, sc, status, 14) + 12
        y = self._genes(cv, sc, dt, y) + 12
        y = self._today(cv, sc, dt, y) + 12
        y = self._resources(cv, sc, dt, y) + 12
        self._charts(cv, sc, y)

    # ---- cards --------------------------------------------------------------------

    def _header(self, cv, sc, status, y) -> float:
        h = 104
        _card(cv, y, h)
        cv.text("LAYA VILLAGE", (X0 + 18, y + 13), 13, MUTED, bold=True)
        cv.text(status.run, (X0 + 22 + cv.measure("LAYA VILLAGE", 13, bold=True)[0], y + 13), 13, MUTED)
        _badge(cv, status, X0 + CW - 18, y + 20)
        cfg = sc.cfg
        if sc.gen is None:
            cv.text("Waiting for the simulation...", (X0 + 18, y + 34), 26, INK, bold=True)
        else:
            w, _ = cv.text(f"Generation {sc.gen + 1}", (X0 + 18, y + 33), 30, INK, bold=True)
            cv.text(f"of {cfg.population.G}", (X0 + 26 + w, y + 46), 16, MUTED)
            if sc.day:
                w, _ = cv.text(f"of {cfg.population.D}", (X0 + CW - 18, y + 46), 16, MUTED, anchor="topright")
                cv.text(f"Day {sc.day}", (X0 + CW - 26 - w, y + 33), 30, INK, bold=True, anchor="topright")
        phase = sc.timeline.phase_at(sc.t) if sc.timeline else None
        names = list(PHASES) + ([phase] if phase and phase not in PHASES else [])
        x, cy = X0 + 18, y + 84
        for name in names:
            w, th = cv.measure(name, 12.5, bold=True)
            on = name == phase
            cv.rect(INK if on else shade(PANEL, 1.06), (x, cy - th / 2 - 3, w + 16, th + 6), radius=th / 2 + 3)
            cv.text(name, (x + 8 + w / 2, cy), 12.5, WHITE if on else MUTED, bold=True, anchor="center")
            x += w + 22
        return y + h

    def _genes(self, cv, sc, dt, y) -> float:
        h = 290
        _card(cv, y, h, "Genes of the living", "bars: alive now   outlines: start of generation")
        now = sc.level_counts()
        start = sc.history.start_counts or now
        top = max([sc.cfg.population.N * 0.5] + [max(c) for c in now.values()] + [max(c) for c in start.values()])
        means = sc.living_means()
        cw, ch = (CW - 36 - 24) / 2, 112
        for k, g in enumerate(GENES):
            x, cy = X0 + 18 + (k % 2) * (cw + 24), y + 46 + (k // 2) * (ch + 12)
            self._gene_cell(cv, sc, g, now[g], start[g], top, means, x, cy, cw, dt)
        return y + h

    def _gene_cell(self, cv, sc, g, now, start, top, means, x, y, w, dt):
        tw, _ = cv.text(TITLE[g], (x, y), 15, INK, bold=True)
        channel = "body colour" if g == sc.color_by else ACCESSORY.get(g)
        if channel:
            cw, chh = cv.measure(channel, 11.5)
            cv.rect(shade(PANEL, 1.06), (x + tw + 8, y + 2, cw + 12, chh + 2), radius=(chh + 2) / 2)
            cv.text(channel, (x + tw + 14, y + 3), 11.5, MUTED)
        if means:
            cv.text(f"avg {means[g]:.2f}", (x + w, y + 3), 12.5, MUTED, anchor="topright")
        top_y, bar_h, gap = y + 26, 60, 10
        bw = (w - 4 * gap) / 5
        for lv in range(5):
            bx = x + lv * (bw + gap)
            col = ramp(g, (lv + 0.5) / 5)
            hh = (bar_h - 14) * self._ease((g, lv), now[lv], dt) / top
            sh = (bar_h - 14) * start[lv] / top
            if hh > 0.3:
                cv.rect(col, (bx, top_y + bar_h - hh, bw, hh), radius=min(4, hh / 2))
            if start[lv]:
                cv.rect(shade(col, 0.62), (bx - 1, top_y + bar_h - sh - 1, bw + 2, sh + 2), radius=4, width=1.4)
            cv.text(str(now[lv]), (bx + bw / 2, top_y + bar_h - max(hh, sh) - 2), 12, INK, bold=True,
                    anchor="midbottom")
        cv.line(FAINT, (x, top_y + bar_h + 1), (x + w, top_y + bar_h + 1), 1)
        lo, hi = ENDS[g]
        cv.text(lo, (x, top_y + bar_h + 5), 12, MUTED)
        cv.text(hi, (x + w, top_y + bar_h + 5), 12, MUTED, anchor="topright")

    def _today(self, cv, sc, dt, y) -> float:
        h = 222
        st = sc.stats
        _card(cv, y, h, "Today", f"{sc.n_alive()} villagers alive" if sc.gen is not None else None)
        alive = max(1, sc.n_alive())
        x, ry = X0 + 18, y + 44
        for a in ACTIVITIES:
            n = st.plans.get(a, 0) if st.revealed else 0
            sprites.icon(cv, a, x + 9, ry + 9, 7)
            cv.text(ACTIVITY_LABEL[a], (x + 26, ry), 14, INK)
            bx, bw = x + 112, 140
            cv.rect(shade(PANEL, 1.05), (bx, ry + 3, bw, 13), radius=6.5)
            v = self._ease(("act", a), n, dt)
            if v > 0.05:
                cv.rect(ACTIVITY_COLOR[a], (bx, ry + 3, max(13, bw * v / alive), 13), radius=6.5)
            cv.text(str(n), (bx + bw + 8, ry), 14, INK, bold=True)
            ry += 27
        if st.revealed:
            cv.text(f"{st.far} took the far, risky route", (x, ry + 6), 12.5, MUTED)
        x, ry = X0 + 334, y + 44
        for kind, title, text, color in _outcomes(sc):
            sprites.icon(cv, kind, x + 8, ry + 9, 7)
            cv.text(title, (x + 24, ry), 13.5, INK, bold=True)
            cv.text(text, (x + 24, ry + 16), 12.5, color)
            ry += 35
        return y + h

    def _resources(self, cv, sc, dt, y) -> float:
        h = 150
        st = sc.stats
        _card(cv, y, h, "Resources and scarcity", "daily supply vs. what the gatherers would take")
        alive = [s for s in sc.shown.values() if s.alive and not s.leaving]
        for k, r in enumerate(("food", "wood")):
            x, ry = X0 + 18, y + 46 + k * 50
            sprites.icon(cv, r, x + 9, ry + 9, 8)
            cv.text(r.capitalize(), (x + 26, ry), 15, INK, bold=True)
            season = st.season.get(r, 1.0)
            label, color = ("plentiful", GOOD) if season > 1.01 else ("scarce", BAD) if season < 0.99 else ("normal season", MUTED)
            cv.text(label, (x + 26, ry + 20), 12.5, color)
            bx, bw = X0 + 160, 240
            cv.rect(shade(PANEL, 1.05), (bx, ry + 3, bw, 14), radius=7)
            supply, demand = st.supply.get(r), st.demand.get(r)
            mark = bx + bw / 1.5
            if supply:
                v = self._ease(("demand", r), demand / supply, dt)
                fill = min(v, 1.5) / 1.5 * bw
                if fill > 1:
                    cv.rect(ACTIVITY_COLOR[f"gather_{r}"], (bx, ry + 3, min(fill, mark - bx), 14), radius=7)
                if fill > mark - bx:
                    cv.rect(BAD, (mark, ry + 3, fill - (mark - bx), 14), radius=7)
                note = (f"crowded: yields cut to {supply / demand:.0%}" if demand > supply
                        else f"plenty: {demand:.0f} of {supply:.0f} taken")
                cv.text(note, (bx, ry + 21), 12.5, BAD if demand > supply else MUTED)
            else:
                cv.text("waiting for the gatherers", (bx, ry + 21), 12.5, MUTED)
            cv.line(INK, (mark, ry - 1), (mark, ry + 21), 2)
            total = sum(getattr(s, r) + getattr(s, f"carry_{r}") for s in alive)
            none = sum(getattr(s, r) + getattr(s, f"carry_{r}") == 0 for s in alive)
            cv.text(str(total), (X0 + CW - 92, ry - 3), 22, INK, bold=True, anchor="topright")
            cv.text("stored", (X0 + CW - 86, ry + 4), 12.5, MUTED)
            cv.text(f"{none} have none", (X0 + CW - 18, ry + 22), 12.5, BAD if none else MUTED, anchor="topright")
        return y + h

    def _charts(self, cv, sc, y):
        h = H - 14 - y
        _card(cv, y, h)
        cw = (CW - 36 - 28) / 2
        _alive_chart(cv, sc, X0 + 18, y + 12, cw, h - 22)
        _evolution_chart(cv, sc, X0 + 18 + cw + 28, y + 12, cw, h - 22)


# ---- pieces -----------------------------------------------------------------------

def _card(cv, y, h, title=None, note=None):
    cv.rect(shade(PANEL, 0.92), (X0, y + 2, CW, h), radius=14)
    cv.rect(PAPER, (X0, y, CW, h), radius=14)
    if title:
        cv.text(title, (X0 + 18, y + 12), 17, INK, bold=True)
    if note:
        cv.text(note, (X0 + CW - 18, y + 16), 12.5, MUTED, anchor="topright")


def _badge(cv, status: Status, right, cy):
    if status.mode == "video":
        return
    if status.paused:
        text, color = "PAUSED", WARN
    elif status.skipping:
        text, color = "FAST-FORWARD", WARN
    elif status.mode == "live":
        text, color = ("LIVE  waiting for the next day" if status.waiting else "LIVE"), BAD
    else:
        text, color = "REPLAY", MUTED
    text += f"   x{status.speed:g}"
    w, th = cv.measure(text, 12.5, bold=True)
    x0 = right - w - 30
    cv.rect(shade(color, 1.75), (x0, cy - th / 2 - 4, w + 30, th + 8), radius=th / 2 + 4)
    cv.circle(color, (x0 + 12, cy), 4.5)
    cv.text(text, (x0 + 22, cy), 12.5, shade(color, 0.7), bold=True, anchor="midleft")


def _outcomes(sc):
    st = sc.stats
    went = st.plans.get("hunt", 0) if st.revealed else None
    h = st.hunt
    if h is None:
        hunt = ("nobody went hunting", MUTED) if went == 0 else ("...", MUTED)
    elif h["cancelled"]:
        hunt = ("called off: too few hunters", MUTED)
    elif h["success"]:
        hunt = (f"{len(h['contributors'])}/{len(h['party'])} worked hard: +{h['payout']} food each", GOOD)
    else:
        hunt = (f"{len(h['contributors'])}/{len(h['party'])} worked hard, {h['threshold']} needed: failed", BAD)
    raids = (f"{st.raids} raids, {st.fights} fought back, raider won {st.raids_won}", INK) if st.raids \
        else ("no raids yet", MUTED)
    trades = (f"{st.offers} offers, {st.accepted} accepted", INK) if st.offers else ("no trades yet", MUTED)
    gifts = (f"{st.gift_asks} asked for help, {st.gifts} helped", INK) if st.gift_asks else ("no requests yet", MUTED)
    night = (f"{st.hungry} hungry, {st.cold} cold, {st.died} died", BAD if st.died else INK) if st.night \
        else ("...", MUTED)
    return [("hunt", "Hunt", *hunt), ("raid", "Raids", *raids), ("check", "Market", *trades),
            ("heart", "Gifts", *gifts), ("rest", "Night", *night)]


def _alive_chart(cv, sc, x, y, w, h):
    cv.text("Alive this generation", (x, y), 14, INK, bold=True)
    n, days = sc.cfg.population.N, sc.cfg.population.D
    px, py, pw, ph = x + 22, y + 32, w - 26, h - 76
    cv.line(FAINT, (px, py + ph), (px + pw, py + ph), 1)
    cv.line(FAINT, (px, py), (px, py + ph), 1)
    cv.text(str(n), (px - 4, py), 11, MUTED, anchor="topright")
    cv.text("0", (px - 4, py + ph), 11, MUTED, anchor="bottomright")
    cv.text("day 1", (px, py + ph + 3), 11, MUTED)
    cv.text(f"day {days}", (px + pw, py + ph + 3), 11, MUTED, anchor="topright")
    values = list(sc.history.alive)
    if sc.timeline is not None and sc.timeline.kind == "day":
        values.append(sc.n_alive())
    if len(values) >= 1 and sc.gen is not None:
        pts = [(px + pw * d / days, py + ph * (1 - v / n)) for d, v in enumerate(values)]
        if len(pts) > 1:
            cv.polygon(shade(GOOD, 1.7), [(pts[0][0], py + ph)] + pts + [(pts[-1][0], py + ph)])
            cv.polyline(shade(GOOD, 0.8), pts, 2.2)
        cv.circle(shade(GOOD, 0.8), pts[-1], 3.5)
        lx = pts[-1][0] + 6 if pts[-1][0] < px + pw - 30 else pts[-1][0] - 6
        cv.text(str(values[-1]), (lx, max(pts[-1][1] - 2, py + 14)), 12, INK, bold=True,
                anchor="bottomleft" if lx > pts[-1][0] else "bottomright")
    lx, ly = x, py + ph + 22
    for cause, label in DEATH_LABEL.items():
        k = sc.history.deaths.get(cause, 0)
        cv.circle(DEATH_COLOR[cause], (lx + 5, ly + 8), 4.5)
        tw, _ = cv.text(f"{label} {k}", (lx + 13, ly), 12, INK if k else MUTED)
        lx += tw + 22
        if lx > x + w - 60:
            lx, ly = x, ly + 18


def _evolution_chart(cv, sc, x, y, w, h):
    cv.text("Average genes per generation", (x, y), 14, INK, bold=True)
    gens = sc.cfg.population.G
    px, py, pw, ph = x + 22, y + 32, w - 26, h - 76
    cv.line(FAINT, (px, py + ph), (px + pw, py + ph), 1)
    cv.line(FAINT, (px, py), (px, py + ph), 1)
    cv.line(shade(FAINT, 1.12), (px, py + ph / 2), (px + pw, py + ph / 2), 1)
    cv.text("1", (px - 4, py), 11, MUTED, anchor="topright")
    cv.text("0", (px - 4, py + ph), 11, MUTED, anchor="bottomright")
    cv.text("gen 1", (px, py + ph + 3), 11, MUTED)
    cv.text(f"gen {gens}", (px + pw, py + ph + 3), 11, MUTED, anchor="topright")

    def at(gen, v):
        return px + pw * gen / max(1, gens - 1), py + ph * (1 - v)

    means = {g: m for g, m in sc.means.items() if sc.gen is None or g <= sc.gen}
    living = sc.living_means()
    for gene in GENES:
        color = ramp(gene, 1.0)
        pts = [at(g, means[g][gene]) for g in sorted(means)]
        if len(pts) > 1:
            cv.polyline(color, pts, 2.2)
        elif pts:
            cv.circle(color, pts[0], 2.5)
        if living and sc.gen in means:
            a, b = at(sc.gen, means[sc.gen][gene]), at(sc.gen, living[gene])
            cv.line(color, a, b, 1.5)
            cv.circle(color, b, 4)
            cv.circle(WHITE, b, 1.8)
    lx, ly = x, py + ph + 22
    for gene in GENES:
        cv.circle(ramp(gene, 1.0), (lx + 5, ly + 8), 4.5)
        tw, _ = cv.text(TITLE[gene], (lx + 13, ly), 12, INK)
        lx += tw + 22
        if lx > x + w - 70:
            lx, ly = x, ly + 18


def draw_legend(cv: Canvas, sc):
    """'How to read a blob', under the map."""
    cv.rect(PAPER, (14, 994, MAP_W - 28, 74), radius=14, alpha=238)
    cv.text("HOW TO READ A BLOB", (32, 1004), 12, MUTED, bold=True)
    mid = ramp(sc.color_by, 0.5)
    belly = ramp("specialization", 0.5)
    items = (
        (f"Body colour: {TITLE[sc.color_by].lower()}", "{} to {}".format(*ENDS[sc.color_by]),
         [dict(body=ramp(sc.color_by, 0.02)), dict(body=ramp(sc.color_by, 0.98))]),
        ("Halo: altruism", "miser to devoted helper", [dict(halo=0), dict(halo=4)]),
        ("Eyebrows: risk", "cautious to reckless", [dict(brows=0), dict(brows=4)]),
        ("Belly: specialization", "woodcutter to forager",
         [dict(belly=ramp("specialization", 0.0)), dict(belly=ramp("specialization", 1.0))]),
    )
    x = 30
    for title, sub, pair in items:
        for k, opts in enumerate(pair):
            bx = x + 12 + k * 34
            kw = {"body": mid, "belly": belly, "halo": 0, "brows": 2} | opts
            sprites.blob(cv, bx, 1058, 11, kw.pop("body"), **kw)
        cv.text(">", (x + 29, 1046), 13, MUTED, bold=True, anchor="center")
        cv.text(title, (x + 74, 1024), 13.5, INK, bold=True)
        cv.text(sub, (x + 74, 1043), 12.5, MUTED)
        x += 296


def draw_tooltip(cv: Canvas, sc, i: int, pos):
    a, s = sc.agents[i], sc.shown[i]
    w, h = 300, 196
    x = min(pos[0] + 18, MAP_W - w - 10)
    y = min(pos[1] + 18, H - h - 10)
    cv.rect(shade(PANEL, 0.8), (x, y + 2, w, h), radius=12)
    cv.rect(PAPER, (x, y, w, h), radius=12)
    cv.text(f"Villager {i}", (x + 14, y + 10), 16, INK, bold=True)
    if a.parents:
        cv.text("child of " + " and ".join(map(str, a.parents)), (x + w - 14, y + 13), 12, MUTED, anchor="topright")
    ry = y + 38
    for g in GENES:
        v = a.genome[g]
        cv.text(TITLE[g], (x + 14, ry), 12.5, INK)
        cv.rect(shade(PANEL, 1.05), (x + 112, ry + 3, 70, 10), radius=5)
        cv.rect(ramp(g, v), (x + 112, ry + 3, max(10, 70 * v), 10), radius=5)
        cv.text(LEVELS[g][level(v)], (x + 190, ry), 12, MUTED)
        ry += 22
    cv.text(f"Health {s.health}/{sc.cfg.agent.health_max}    Food {s.food + s.carry_food}    "
            f"Wood {s.wood + s.carry_wood}", (x + 14, ry + 6), 12.5, INK, bold=True)
    plan = sc.item.plans.get(i) if isinstance(sc.item, DayLog) else None
    if plan:
        what = ACTIVITY_LABEL[plan.activity].lower()
        if plan.route:
            what += f" ({plan.route})"
        if plan.target is not None:
            what += f" villager {plan.target}"
        cv.text(f"Today: {what}", (x + 14, ry + 28), 12.5, MUTED)


def draw_help(cv: Canvas):
    keys = (("space", "pause / resume"), ("+  -", "faster / slower"), ("N", "skip to the next day"),
            ("G", "skip to the next generation"), ("L", "jump to the latest day (live)"),
            ("1 2 3 4", "body colour: aggression, altruism, risk, specialization"),
            ("mouse", "hover a blob to see its genes"), ("F", "full screen on / off (or resize the window)"),
            ("H", "show / hide this help"), ("Q  esc", "quit (esc leaves full screen first)"))
    w, h = 560, 52 + 26 * len(keys)
    x, y = (MAP_W - w) / 2, (H - h) / 2
    cv.rect(shade(PANEL, 0.8), (x, y + 3, w, h), radius=16)
    cv.rect(PAPER, (x, y, w, h), radius=16)
    cv.text("Controls", (x + 22, y + 14), 18, INK, bold=True)
    for k, (key, what) in enumerate(keys):
        cv.text(key, (x + 22, y + 48 + 26 * k), 14, INK, bold=True)
        cv.text(what, (x + 130, y + 48 + 26 * k), 14, MUTED)
