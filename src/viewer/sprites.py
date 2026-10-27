# Primer-style sprites: blobs, resources, icons, homes and the static village background.
import math
import random

from viewer.canvas import Canvas
from viewer.layout import MAP_W, H, Layout
from viewer.style import (BAD, BLACK, FOOD, FOOD_DARK, FOREST_FAR, GOLD, GOOD, GROUND, GROVE, HEART, INK, MEADOW,
                          MEADOW_FAR, MUTED, PATH, PLAZA, SAVANNA, TREE, TREE_FAR, WARN, WHITE, WOOD, WOOD_DARK,
                          WOOD_LIGHT, mix, shade)

BROW = (52, 44, 52)
EYE_DARK = (34, 32, 44)
HALO_ALPHA = (0, 105, 160, 215, 255)


# ---- blobs ----------------------------------------------------------------------

def shadow(cv: Canvas, x, y, r, lift=0.0):
    cv.ellipse(BLACK, (x, y), r * 0.92 * (1 - min(lift, 24) / 48), r * 0.3, alpha=42)


def blob(cv: Canvas, x, y, r, body, *, belly=None, halo=0, brows=2, look=(0.0, 0.0), sx=1.0, sy=1.0, lift=0.0,
         face="normal", blink=False, ring=None):
    """A blob standing on (x, y). `halo` and `brows` are gene levels 0-4 (altruism, risk)."""
    bx, by = r * sx, r * sy
    cy = y - lift - by
    if ring:
        cv.ellipse(ring, (x, cy), bx + 5, by + 5, width=2.2)
    cv.ellipse(shade(body, 0.7), (x, cy), bx + 1.4, by + 1.4)
    cv.ellipse(body, (x, cy), bx, by)
    if belly:
        cv.ellipse(belly, (x, cy + 0.47 * by), 0.6 * bx, 0.4 * by)
    cv.ellipse(WHITE, (x - 0.42 * bx, cy - 0.5 * by), 0.22 * bx, 0.14 * by, alpha=100)
    lx, ly = look
    ex, ey, er = 0.37 * bx, cy - 0.16 * by, 0.27 * r
    if face == "angry":
        brows = 4
    elif face == "sad":
        brows, ly = 0, 0.7
    for side in (-1, 1):
        cx = x + side * ex + lx * 0.1 * r
        _eye(cv, cx, ey + ly * 0.06 * r, er, side, face, blink, (lx, ly), body)
        if face != "fainted":
            tilt = (brows - 2) / 2 * 0.17 * r
            by0 = ey - er - 0.2 * r
            cv.line(BROW, (cx - side * 0.24 * r, by0 + tilt), (cx + side * 0.27 * r, by0 - tilt), 0.13 * r)
    _mouth(cv, x + lx * 0.1 * r, cy + 0.3 * by, r, face)
    if halo:
        hy = cy - by - 0.36 * r
        if halo == 4:
            cv.ellipse(GOLD, (x, hy), 0.66 * r, 0.24 * r, width=0.3 * r, alpha=55)
        cv.ellipse(GOLD, (x, hy), 0.54 * r, 0.17 * r, width=0.14 * r, alpha=HALO_ALPHA[halo])


def _eye(cv, cx, cy, er, side, face, blink, look, body):
    if face == "fainted":
        d = 0.62 * er
        cv.line(EYE_DARK, (cx - d, cy - d), (cx + d, cy + d), 0.45 * er)
        cv.line(EYE_DARK, (cx - d, cy + d), (cx + d, cy - d), 0.45 * er)
    elif face == "happy":
        cv.polyline(EYE_DARK, [(cx - 0.8 * er, cy + 0.25 * er), (cx, cy - 0.5 * er), (cx + 0.8 * er, cy + 0.25 * er)],
                    0.5 * er)
    elif face == "hurt":
        s = -side
        cv.polyline(EYE_DARK, [(cx - s * 0.6 * er, cy - 0.6 * er), (cx + s * 0.55 * er, cy),
                               (cx - s * 0.6 * er, cy + 0.6 * er)], 0.45 * er)
    elif face == "sleep" or blink:
        cv.polyline(EYE_DARK, [(cx - 0.8 * er, cy), (cx, cy + 0.38 * er), (cx + 0.8 * er, cy)], 0.42 * er)
    else:
        cv.circle(WHITE, (cx, cy), er)
        cv.circle(EYE_DARK, (cx + look[0] * 0.42 * er, cy + look[1] * 0.42 * er), 0.55 * er)
        cv.circle(WHITE, (cx + look[0] * 0.42 * er - 0.2 * er, cy + look[1] * 0.42 * er - 0.22 * er), 0.17 * er)
        if face == "lazy":
            cv.ellipse(body, (cx, cy - 0.55 * er), er + 0.6, 0.62 * er)


def _mouth(cv, x, y, r, face):
    w = 0.1 * r
    if face == "happy":
        cv.polygon(EYE_DARK, [(x - 0.26 * r, y - 0.04 * r), (x + 0.26 * r, y - 0.04 * r),
                              (x + 0.16 * r, y + 0.16 * r), (x, y + 0.22 * r), (x - 0.16 * r, y + 0.16 * r)])
    elif face == "sad":
        cv.polyline(EYE_DARK, [(x - 0.17 * r, y + 0.09 * r), (x, y), (x + 0.17 * r, y + 0.09 * r)], w)
    elif face in ("hurt", "fainted", "sleep"):
        cv.circle(EYE_DARK, (x, y + 0.04 * r), 0.09 * r)
    elif face in ("angry", "lazy"):
        cv.line(EYE_DARK, (x - 0.15 * r, y + 0.04 * r), (x + 0.15 * r, y + 0.02 * r), w)
    else:
        cv.polyline(EYE_DARK, [(x - 0.17 * r, y), (x, y + 0.08 * r), (x + 0.17 * r, y)], w)


def ghost(cv: Canvas, x, y, r, alpha):
    """A departing soul rising from (x, y)."""
    layer = cv.layer(x - 2 * r, y - 3 * r, 4 * r, 4 * r)
    body = (246, 248, 255)
    layer.ellipse((170, 176, 200), (x, y - 1.3 * r), 0.95 * r + 1.2, 1.05 * r + 1.2)
    layer.ellipse(body, (x, y - 1.3 * r), 0.95 * r, 1.05 * r)
    for k in (-1, 0, 1):
        layer.circle(body, (x + k * 0.58 * r, y - 0.38 * r), 0.36 * r)
    for side in (-1, 1):
        layer.ellipse(EYE_DARK, (x + side * 0.34 * r, y - 1.45 * r), 0.13 * r, 0.2 * r)
    cv.paste(layer, alpha)


# ---- resources and icons ----------------------------------------------------------

def food(cv: Canvas, x, y, s=4.5, alpha=255):
    cv.circle(FOOD_DARK, (x, y), s + 0.9, alpha=alpha)
    cv.circle(FOOD, (x, y), s, alpha=alpha)
    cv.circle(WHITE, (x - 0.35 * s, y - 0.38 * s), 0.3 * s, alpha=min(alpha, 170))
    cv.ellipse(FOOD_DARK, (x + 0.42 * s, y - 1.08 * s), 0.5 * s, 0.24 * s, alpha=alpha)


def wood(cv: Canvas, x, y, s=4.5, alpha=255):
    w, h = 2.5 * s, 1.15 * s
    cv.rect(WOOD_DARK, (x - w / 2 - 0.9, y - h / 2 - 0.9, w + 1.8, h + 1.8), radius=h / 2 + 0.9, alpha=alpha)
    cv.rect(WOOD, (x - w / 2, y - h / 2, w, h), radius=h / 2, alpha=alpha)
    cv.ellipse(WOOD_LIGHT, (x + w / 2 - 0.45 * h, y), 0.34 * h, 0.4 * h, alpha=alpha)


def item(cv: Canvas, kind: str, x, y, s=4.5, alpha=255):
    (food if kind == "food" else wood)(cv, x, y, s, alpha)


def heart(cv: Canvas, x, y, s, color=HEART, alpha=255):
    cv.circle(color, (x - 0.5 * s, y - 0.25 * s), 0.56 * s, alpha=alpha)
    cv.circle(color, (x + 0.5 * s, y - 0.25 * s), 0.56 * s, alpha=alpha)
    cv.polygon(color, [(x - 1.03 * s, y - 0.08 * s), (x + 1.03 * s, y - 0.08 * s), (x, y + 1.05 * s)], alpha=alpha)


def icon(cv: Canvas, kind: str, x, y, s=7.0):
    """A small pictogram centred on (x, y)."""
    if kind in ("food", "gather_food"):
        food(cv, x, y + 0.15 * s, 0.75 * s)
    elif kind in ("wood", "gather_wood"):
        wood(cv, x, y, 0.62 * s)
    elif kind == "hunt":
        cv.line(WOOD_DARK, (x - 0.95 * s, y + 0.95 * s), (x + 0.55 * s, y - 0.55 * s), 0.26 * s)
        cv.polygon((128, 134, 146), [(x + 1.15 * s, y - 1.15 * s), (x + 0.3 * s, y - 0.8 * s),
                                     (x + 0.8 * s, y - 0.3 * s)])
    elif kind == "raid":
        cv.rect(EYE_DARK, (x - 1.25 * s, y - 0.5 * s, 2.5 * s, s), radius=0.45 * s)
        for side in (-1, 1):
            cv.ellipse(WHITE, (x + side * 0.55 * s, y), 0.32 * s, 0.24 * s)
        cv.line(EYE_DARK, (x + 1.1 * s, y - 0.1 * s), (x + 1.5 * s, y + 0.5 * s), 0.22 * s)
    elif kind == "rest":
        cv.text("z", (x - 0.45 * s, y + 0.4 * s), 1.3 * s, (110, 120, 200), bold=True, anchor="center")
        cv.text("z", (x + 0.45 * s, y - 0.3 * s), 1.0 * s, (110, 120, 200), bold=True, anchor="center")
    elif kind == "far":
        cv.polygon(shade(WARN, 0.8), [(x, y - 1.05 * s), (x + 1.05 * s, y + 0.8 * s), (x - 1.05 * s, y + 0.8 * s)])
        cv.polygon((255, 205, 70), [(x, y - 0.75 * s), (x + 0.78 * s, y + 0.62 * s), (x - 0.78 * s, y + 0.62 * s)])
        cv.line(EYE_DARK, (x, y - 0.25 * s), (x, y + 0.15 * s), 0.18 * s)
        cv.circle(EYE_DARK, (x, y + 0.42 * s), 0.1 * s)
    elif kind == "check":
        cv.polyline(GOOD, [(x - 0.8 * s, y), (x - 0.2 * s, y + 0.6 * s), (x + 0.9 * s, y - 0.7 * s)], 0.34 * s)
    elif kind == "cross":
        cv.line(BAD, (x - 0.7 * s, y - 0.7 * s), (x + 0.7 * s, y + 0.7 * s), 0.34 * s)
        cv.line(BAD, (x - 0.7 * s, y + 0.7 * s), (x + 0.7 * s, y - 0.7 * s), 0.34 * s)
    elif kind == "flag":
        cv.line(WOOD_DARK, (x - 0.6 * s, y + 1.0 * s), (x - 0.6 * s, y - 1.0 * s), 0.18 * s)
        cv.polygon((150, 150, 160), [(x - 0.55 * s, y - 1.05 * s), (x + 1.05 * s, y - 0.65 * s),
                                     (x - 0.55 * s, y - 0.1 * s)])
        cv.polygon(WHITE, [(x - 0.5 * s, y - 0.92 * s), (x + 0.8 * s, y - 0.62 * s), (x - 0.5 * s, y - 0.25 * s)])
    elif kind == "fight":
        burst(cv, x, y, 1.15 * s, (255, 196, 60), (232, 96, 50))
    elif kind == "heart":
        heart(cv, x, y, 0.75 * s)
    elif kind == "cold":
        for a in (0, math.pi / 3, 2 * math.pi / 3):
            dx, dy = math.cos(a) * s, math.sin(a) * s
            cv.line((90, 160, 230), (x - dx, y - dy), (x + dx, y + dy), 0.2 * s)
    elif kind == "hungry":
        food(cv, x, y, 0.6 * s)
        cv.line(BAD, (x - 0.9 * s, y + 0.9 * s), (x + 0.9 * s, y - 0.9 * s), 0.22 * s)
    elif kind == "lazy":
        for k in (-1, 0, 1):
            cv.circle(MUTED, (x + k * 0.6 * s, y + 0.2 * s), 0.2 * s)
    elif kind == "sweat":
        cv.circle((110, 180, 240), (x, y + 0.25 * s), 0.42 * s)
        cv.polygon((110, 180, 240), [(x - 0.4 * s, y + 0.15 * s), (x, y - 0.75 * s), (x + 0.4 * s, y + 0.15 * s)])
    elif kind == "angry":
        for a in (0.25, 1.35, 2.45, 3.55):
            dx, dy = math.cos(a * math.pi / 2), math.sin(a * math.pi / 2)
            cv.line(BAD, (x + dx * 0.25 * s, y + dy * 0.25 * s), (x + dx * 0.9 * s, y + dy * 0.9 * s), 0.25 * s)


def burst(cv: Canvas, x, y, r, inner, outer, spikes=9, turn=0.0):
    pts = []
    for k in range(spikes * 2):
        a = turn + k * math.pi / spikes
        rr = r if k % 2 == 0 else r * 0.55
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    cv.polygon(outer, pts)
    cv.polygon(inner, [(x + (px - x) * 0.62, y + (py - y) * 0.62) for px, py in pts])


def bubble(cv: Canvas, x, y, w, h, fill=WHITE, edge=(84, 90, 106)):
    """Speech bubble with its tail on (x, y); returns the top-left corner of its content box."""
    x0, y0 = x - w / 2, y - 6 - h
    cv.rect(edge, (x0 - 1.2, y0 - 1.2, w + 2.4, h + 2.4), radius=h / 2 + 1.2)
    cv.polygon(edge, [(x - 4.8, y0 + h - 0.5), (x + 4.8, y0 + h - 0.5), (x, y + 1.5)])
    cv.rect(fill, (x0, y0, w, h), radius=h / 2)
    cv.polygon(fill, [(x - 3.4, y0 + h - 1.8), (x + 3.4, y0 + h - 1.8), (x, y - 0.6)])
    return x0, y0


# ---- homes ----------------------------------------------------------------------

def hut(cv: Canvas, x, y, s=1.0):
    wall, roof = (226, 196, 150), (190, 96, 70)
    cv.ellipse(BLACK, (x, y + 7 * s), 14 * s, 3.5 * s, alpha=30)
    cv.rect(shade(wall, 0.74), (x - 11 * s, y - 9 * s, 22 * s, 17 * s), radius=2.5 * s)
    cv.rect(wall, (x - 10 * s, y - 8 * s, 20 * s, 15 * s), radius=2 * s)
    cv.polygon(shade(roof, 0.74), [(x - 15 * s, y - 6 * s), (x, y - 20.5 * s), (x + 15 * s, y - 6 * s)])
    cv.polygon(roof, [(x - 12.8 * s, y - 7.2 * s), (x, y - 18.6 * s), (x + 12.8 * s, y - 7.2 * s)])
    cv.rect((126, 88, 62), (x - 3.2 * s, y - 1.5 * s, 6.4 * s, 8.5 * s), radius=2.5 * s)


def grave(cv: Canvas, x, y, s=1.0, alpha=255):
    stone = (172, 174, 184)
    cv.ellipse(BLACK, (x, y + 1), 8 * s, 2.6 * s, alpha=alpha * 40 // 255)
    cv.rect(shade(stone, 0.72), (x - 6.6 * s, y - 15.5 * s, 13.2 * s, 16.5 * s), radius=6.6 * s, alpha=alpha)
    cv.rect(stone, (x - 5.6 * s, y - 14.5 * s, 11.2 * s, 15 * s), radius=5.6 * s, alpha=alpha)
    cv.line((122, 124, 136), (x, y - 11.5 * s), (x, y - 3.5 * s), 1.6 * s, alpha=alpha)
    cv.line((122, 124, 136), (x - 3.2 * s, y - 8.6 * s), (x + 3.2 * s, y - 8.6 * s), 1.6 * s, alpha=alpha)


def pile(cv: Canvas, x, y, food_n: int, wood_n: int, s=3.1):
    """An agent's stock: logs stacked on the left, fruit on the right (6 of each drawn, then a count)."""
    for k in range(min(max(wood_n, 0), 6)):
        row, col = divmod(k, 2)
        wood(cv, x - 13 + col * 8.6 + row * 2, y - row * 4.6, s * 0.95)
    for k in range(min(max(food_n, 0), 6)):
        row, col = divmod(k, 2)
        food(cv, x + 5 + col * 7.4 + row * 1.5, y - row * 6.2, s)
    for n, dx in ((wood_n, -9), (food_n, 9)):
        if n > 6:
            cv.text(str(n), (x + dx, y - 17), 11, INK, bold=True, anchor="midbottom")


def campfire(cv: Canvas, x, y, t, strength=1.0):
    cv.circle((255, 170, 70), (x, y - 5), 20 * strength, alpha=int(40 * strength))
    cv.line(WOOD_DARK, (x - 6, y + 2), (x + 6, y - 1), 2.6)
    cv.line(WOOD_DARK, (x - 6, y - 1), (x + 6, y + 2), 2.6)
    f = strength * (1 + 0.16 * math.sin(t * 19 + x) + 0.08 * math.sin(t * 31 + y))
    cv.polygon((240, 110, 40), [(x - 5, y), (x - 1, y - 14 * f), (x + 5, y)])
    cv.polygon((255, 210, 80), [(x - 2.6, y), (x, y - 8 * f), (x + 2.6, y)])


def beast(cv: Canvas, x, y, t, state="idle", face=1, alpha=255):
    """The hunt's prey, a big woolly blob with tusks standing on (x, y)."""
    down = state == "down"
    fur = (150, 118, 92) if not down else (150, 140, 132)
    run = state == "flee"
    cv.ellipse(BLACK, (x, y + 2), 44, 9, alpha=alpha * 40 // 255)
    if not down:
        for k, dx in enumerate((-24, -10, 10, 24)):
            swing = 4 * math.sin(t * 22 + k * 1.7) if run else 0
            cv.rect(shade(fur, 0.6), (x + dx - 4 + swing, y - 14, 8, 15), radius=3, alpha=alpha)
    ry = 18 if down else 27 + 1.2 * math.sin(t * 3)
    cy = y - (ry + 4 if not down else ry)
    cv.ellipse(shade(fur, 0.7), (x, cy), 42, ry + 1.5, alpha=alpha)
    cv.ellipse(fur, (x, cy), 40.5, ry, alpha=alpha)
    cv.ellipse(shade(fur, 1.18), (x - 12, cy - 0.45 * ry), 18, 0.32 * ry, alpha=alpha)
    hx, hy = x + face * 34, cy - 2 + (8 if down else 0)
    cv.circle(shade(fur, 0.7), (hx, hy), 17.5, alpha=alpha)
    cv.circle(fur, (hx, hy), 16, alpha=alpha)
    tusk = [(hx + face * 4, hy + 8), (hx + face * 22, hy + 16), (hx + face * 26, hy + 6), (hx + face * 20, hy + 12)]
    cv.polygon((250, 246, 232), tusk, alpha=alpha)
    ex, ey = hx + face * 6, hy - 5
    if down:
        cv.line(EYE_DARK, (ex - 3.5, ey - 3.5), (ex + 3.5, ey + 3.5), 2.2, alpha=alpha)
        cv.line(EYE_DARK, (ex - 3.5, ey + 3.5), (ex + 3.5, ey - 3.5), 2.2, alpha=alpha)
    else:
        cv.circle(WHITE, (ex, ey), 5.2, alpha=alpha)
        cv.circle(EYE_DARK, (ex + face * 1.5, ey + (1.5 if state == "struggle" else 0)), 2.7, alpha=alpha)
        cv.line(EYE_DARK, (ex - 5, ey - 7 + (2 if state == "struggle" else 0)), (ex + 5, ey - 7), 2, alpha=alpha)


# ---- background ------------------------------------------------------------------

def background(cv: Canvas, layout: Layout):
    """The static part of the map: ground, paths, fields, forests, hunting grounds, market square and huts."""
    rng = random.Random(7)
    cv.rect(GROUND, (0, 0, MAP_W, H))
    for _ in range(260):
        x, y = rng.uniform(0, MAP_W), rng.uniform(0, H)
        cv.ellipse(shade(GROUND, 0.965), (x, y), rng.uniform(3, 9), rng.uniform(1.5, 3.5))
    zones = layout.zones
    px, py = layout.plaza
    for z in zones.values():
        cv.line(PATH, layout.plaza, z.center, 22)
    for z in zones.values():
        _zone(cv, z, rng)
    _plaza(cv, layout)
    for slot in range(layout.n):
        hut(cv, *layout.hut(slot))
    for z in zones.values():
        title = z.label + (" (far, risky)" if z.far else "")
        x, y = z.center[0], z.center[1] - z.ry - 10
        w, h = cv.measure(title, 15, bold=True)
        pad = 24 if z.far else 0
        cv.rect(shade(GROUND, 1.25), (x - w / 2 - 10 - pad, y - h / 2 - 3, w + 20 + pad, h + 6), radius=h / 2 + 3,
                alpha=225)
        cv.text(title, (x + pad / 2, y), 15, shade(INK, 1.25), bold=True, anchor="center")
        if z.far:
            icon(cv, "far", x - w / 2 - 6, y, 7.5)


def _zone(cv: Canvas, z, rng):
    kind = z.key.split("_")[0]
    (cx, cy), rx, ry = z.center, z.rx, z.ry
    fill = {"food": MEADOW_FAR if z.far else MEADOW, "wood": FOREST_FAR if z.far else GROVE, "hunt": SAVANNA}[kind]
    cv.ellipse(shade(fill, 0.9), (cx, cy), rx + 5, ry + 5)
    cv.ellipse(fill, (cx, cy), rx, ry)
    if kind == "wood":
        trees = sorted(((cx + (rx + rng.uniform(-6, 14)) * math.cos(a), cy + (ry + rng.uniform(-6, 14)) * math.sin(a))
                        for a in (k * math.tau / 26 + rng.uniform(-0.08, 0.08) for k in range(26))),
                       key=lambda p: p[1])
        for tx, ty in trees:
            _tree(cv, tx, ty, rng.uniform(0.85, 1.2), TREE_FAR if z.far else TREE)
    elif kind == "food":
        for k in range(22):
            a = k * math.tau / 22 + rng.uniform(-0.1, 0.1)
            bx, by = cx + (rx + rng.uniform(-4, 8)) * math.cos(a), cy + (ry + rng.uniform(-4, 8)) * math.sin(a)
            _bush(cv, bx, by, rng.uniform(0.8, 1.15), shade(fill, 0.78), rng)
    else:
        for _ in range(46):
            a, rr = rng.uniform(0, math.tau), math.sqrt(rng.random())
            gx, gy = cx + rr * rx * math.cos(a), cy + rr * ry * math.sin(a)
            g = shade(SAVANNA, 0.78)
            cv.polyline(g, [(gx - 4, gy - 6), (gx, gy), (gx + 4, gy - 6)], 1.4)
        for dx, dy, s in ((-150, -40, 1.0), (165, 30, 0.8), (-120, 62, 0.7)):
            cv.ellipse((182, 176, 166), (cx + dx, cy + dy), 13 * s, 8 * s)
            cv.ellipse((200, 195, 186), (cx + dx - 2, cy + dy - 2), 9 * s, 5 * s)


def _tree(cv: Canvas, x, y, s, leaf):
    cv.ellipse(BLACK, (x, y + 3), 15 * s, 4.5 * s, alpha=34)
    cv.rect(WOOD_DARK, (x - 3 * s, y - 12 * s, 6 * s, 15 * s), radius=2 * s)
    for dx, dy, r in ((-7, -20, 11), (7, -21, 11), (0, -30, 12)):
        cv.circle(shade(leaf, 0.8), (x + dx * s, y + dy * s), (r + 1.5) * s)
    for dx, dy, r in ((-7, -20, 11), (7, -21, 11), (0, -30, 12)):
        cv.circle(leaf, (x + dx * s, y + dy * s), r * s)
    cv.circle(shade(leaf, 1.25), (x - 4 * s, y - 33 * s), 4.5 * s)


def _bush(cv: Canvas, x, y, s, leaf, rng):
    for dx, dy, r in ((-6, -2, 7), (6, -2, 7), (0, -7, 8)):
        cv.circle(shade(leaf, 0.85), (x + dx * s, y + dy * s), (r + 1.2) * s)
    for dx, dy, r in ((-6, -2, 7), (6, -2, 7), (0, -7, 8)):
        cv.circle(leaf, (x + dx * s, y + dy * s), r * s)
    for _ in range(3):
        cv.circle(mix(FOOD, WHITE, 0.15), (x + rng.uniform(-8, 8) * s, y + rng.uniform(-10, 0) * s), 1.8 * s)


def _plaza(cv: Canvas, layout: Layout):
    px, py = layout.plaza
    cv.ellipse(shade(PLAZA, 0.86), (px, py), 104, 70)
    cv.ellipse(PLAZA, (px, py), 100, 66)
    cv.ellipse(shade(PLAZA, 0.95), (px, py), 28, 18)
    cv.text("MARKET", (px, py), 13, shade(PLAZA, 0.62), bold=True, anchor="center")
    awnings = ((232, 96, 84), (84, 150, 220), (246, 190, 60), (110, 186, 110), (196, 120, 206), (240, 150, 80))
    for (sx, sy), col in zip(layout.stalls, awnings):
        cv.rect((170, 130, 92), (sx - 11, sy - 6, 22, 9), radius=2)
        for k in range(4):
            cv.rect(col if k % 2 == 0 else WHITE, (sx - 12 + k * 6, sy - 13, 6, 7))
        cv.rect(shade(col, 0.7), (sx - 12, sy - 14, 24, 2), radius=1)
