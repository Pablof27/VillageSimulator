# Drawing in the viewer's virtual 1920x1080 coordinates on a (possibly supersampled) pygame surface.
import math
from contextlib import contextmanager

import pygame

from viewer.style import font_path


class Canvas:
    """`k` = surface pixels per virtual unit; shapes with alpha < 255 are drawn on a temporary RGBA layer."""

    def __init__(self, surface: pygame.Surface, k: float, origin=(0.0, 0.0)):
        self.surface, self.k, self.origin = surface, k, origin
        self._fonts: dict = {}
        self._texts: dict = {}

    def xy(self, p) -> tuple[float, float]:
        return (p[0] - self.origin[0]) * self.k, (p[1] - self.origin[1]) * self.k

    def _w(self, width: float) -> int:
        return 0 if width <= 0 else max(1, round(width * self.k))

    @contextmanager
    def _layer(self, alpha, x0, y0, x1, y1):
        if alpha >= 255:
            yield self.surface, 0, 0, 255
            return
        ox, oy = math.floor(x0) - 2, math.floor(y0) - 2
        layer = pygame.Surface((math.ceil(x1) - ox + 3, math.ceil(y1) - oy + 3), pygame.SRCALPHA)
        yield layer, ox, oy, int(alpha)
        self.surface.blit(layer, (ox, oy))

    # ---- shapes ---------------------------------------------------------------

    def circle(self, color, c, r, width=0.0, alpha=255):
        if alpha <= 0 or r <= 0:
            return
        (x, y), rr = self.xy(c), r * self.k
        with self._layer(alpha, x - rr, y - rr, x + rr, y + rr) as (s, dx, dy, a):
            pygame.draw.circle(s, (*color[:3], a), (x - dx, y - dy), rr, self._w(width))

    def ellipse(self, color, c, rx, ry, width=0.0, alpha=255):
        if alpha <= 0 or rx <= 0 or ry <= 0:
            return
        (x, y), ax, ay = self.xy(c), rx * self.k, ry * self.k
        with self._layer(alpha, x - ax, y - ay, x + ax, y + ay) as (s, dx, dy, a):
            pygame.draw.ellipse(s, (*color[:3], a), (round(x - ax - dx), round(y - ay - dy), round(2 * ax),
                                                     round(2 * ay)), self._w(width))

    def rect(self, color, r, radius=0.0, width=0.0, alpha=255):
        if alpha <= 0 or r[2] <= 0 or r[3] <= 0:
            return
        (x, y), w, h = self.xy(r[:2]), r[2] * self.k, r[3] * self.k
        with self._layer(alpha, x, y, x + w, y + h) as (s, dx, dy, a):
            pygame.draw.rect(s, (*color[:3], a), (round(x - dx), round(y - dy), round(w), round(h)),
                             self._w(width), border_radius=round(radius * self.k))

    def polygon(self, color, points, width=0.0, alpha=255):
        if alpha <= 0:
            return
        pts = [self.xy(p) for p in points]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        with self._layer(alpha, min(xs), min(ys), max(xs), max(ys)) as (s, dx, dy, a):
            pygame.draw.polygon(s, (*color[:3], a), [(px - dx, py - dy) for px, py in pts], self._w(width))

    def line(self, color, p0, p1, width=1.0, alpha=255):
        self.polyline(color, (p0, p1), width, alpha)

    def polyline(self, color, points, width=1.0, alpha=255):
        """Thick polyline with round joins and caps."""
        if alpha <= 0 or len(points) < 2:
            return
        pts = [self.xy(p) for p in points]
        w = max(1, round(width * self.k))
        r = w / 2
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        with self._layer(alpha, min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r) as (s, dx, dy, a):
            col = (*color[:3], a)
            q = [(px - dx, py - dy) for px, py in pts]
            for p, n in zip(q, q[1:]):
                pygame.draw.line(s, col, p, n, w)
            if w > 2:
                for p in q:
                    pygame.draw.circle(s, col, p, r)

    # ---- text -----------------------------------------------------------------

    def font(self, size: float, bold=False) -> pygame.font.Font:
        key = (max(4, round(size * self.k)), bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.Font(font_path(bold), key[0])
        return self._fonts[key]

    def text(self, s: str, pos, size: float, color, bold=False, anchor="topleft", alpha=255) -> tuple[float, float]:
        """Draws `s` with `anchor` (a pygame.Rect attribute) at `pos`; returns its virtual (width, height)."""
        key = (s, size, bold, tuple(color[:3]))
        surf = self._texts.get(key)
        if surf is None:
            if len(self._texts) > 4000:
                self._texts.clear()
            surf = self._texts[key] = self.font(size, bold).render(s, True, color[:3])
        if alpha > 0:
            x, y = self.xy(pos)
            rect = surf.get_rect(**{anchor: (round(x), round(y))})
            surf.set_alpha(int(min(alpha, 255)))
            self.surface.blit(surf, rect)
        return surf.get_width() / self.k, surf.get_height() / self.k

    def measure(self, s: str, size: float, bold=False) -> tuple[float, float]:
        w, h = self.font(size, bold).size(s)
        return w / self.k, h / self.k

    # ---- layers ---------------------------------------------------------------

    def layer(self, x0: float, y0: float, w: float, h: float) -> "Canvas":
        """A transparent canvas over the virtual rect (x0, y0, w, h), to be `paste`d with an overall alpha."""
        surf = pygame.Surface((math.ceil(w * self.k) + 2, math.ceil(h * self.k) + 2), pygame.SRCALPHA)
        sub = Canvas(surf, self.k, (x0, y0))
        sub._fonts, sub._texts = self._fonts, self._texts
        return sub

    def paste(self, sub: "Canvas", alpha=255) -> None:
        sub.surface.set_alpha(int(alpha))
        self.surface.blit(sub.surface, self.xy(sub.origin))
