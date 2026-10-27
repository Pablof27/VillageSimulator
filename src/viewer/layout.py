# Map geometry in the virtual 1920x1080 frame: homes around the market square, fields and forests on the
# sides (near = safe, far = risky), the hunting grounds at the top. The dashboard uses x >= MAP_W.
import math
from dataclasses import dataclass

W, H = 1920, 1080
MAP_W = 1240
GOLDEN = math.pi * (3 - math.sqrt(5))


@dataclass(frozen=True)
class Zone:
    key: str
    label: str
    center: tuple[float, float]
    rx: float
    ry: float
    far: bool = False

    def spot(self, k: int, n: int, fill=0.72, turn=0.0) -> tuple[float, float]:
        """The k-th of n points spread evenly over the inner `fill` of the ellipse (sunflower pattern)."""
        r = fill * math.sqrt((k + 0.5) / max(n, 1))
        a = k * GOLDEN + turn
        return self.center[0] + r * self.rx * math.cos(a), self.center[1] + r * self.ry * math.sin(a)


class Layout:
    def __init__(self, n_agents: int):
        self.n = n_agents
        self.plaza = (620.0, 700.0)
        self.zones = {z.key: z for z in (
            Zone("food_far", "Wild meadow", (162, 318), 128, 150, far=True),
            Zone("food_near", "Fields", (162, 790), 128, 160),
            Zone("wood_far", "Deep forest", (1078, 318), 128, 150, far=True),
            Zone("wood_near", "Grove", (1078, 790), 128, 160),
            Zone("hunt", "Hunting grounds", (620, 250), 215, 118),
        )}
        self.blob_r = 15.0 if n_agents <= 30 else 13.0 if n_agents <= 60 else 11.0
        self._homes = _rings(n_agents, self.plaza)
        self.stalls = [(self.plaza[0] + 64 * math.cos(a), self.plaza[1] + 40 * math.sin(a) + 4)
                       for a in (math.pi * (i / 3 + 1 / 6) for i in range(6))]

    def hut(self, slot: int) -> tuple[float, float]:
        return self._homes[slot % len(self._homes)][0]

    def door(self, slot: int) -> tuple[float, float]:
        """Where the agent stands when it is at home."""
        (x, y), (ix, iy) = self._homes[slot % len(self._homes)]
        return x + ix * 20, y + iy * 20 + 9

    def pile(self, slot: int) -> tuple[float, float]:
        """Where the agent's stock is stored (behind its hut)."""
        (x, y), (ix, iy) = self._homes[slot % len(self._homes)]
        return x - ix * 21, y - iy * 21 - 4

    def visit(self, slot: int) -> tuple[float, float]:
        """Where a visitor (raider) stands when calling at this home."""
        (x, y), (ix, iy) = self._homes[slot % len(self._homes)]
        return x + ix * 46 + iy * 8, y + iy * 46 + 9 - ix * 8

    def fire(self, slot: int) -> tuple[float, float]:
        """The campfire next to the door, lit at night with the wood the agent burns."""
        (x, y), (ix, iy) = self._homes[slot % len(self._homes)]
        return x + ix * 18 - iy * 17, y + iy * 18 + 10 + ix * 17

    def stall(self, k: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two trading spots (proposer, responder) of market stall k."""
        x, y = self.stalls[k % len(self.stalls)]
        return (x - 15, y + 9), (x + 15, y + 9)

    def hunt_spot(self, k: int, n: int) -> tuple[float, float]:
        cx, cy = self.zones["hunt"].center
        a = math.pi / 2 + (k + 0.5) / max(n, 1) * math.tau
        return cx + 120 * math.cos(a), cy + 18 + 62 * math.sin(a)


def _rings(n: int, center) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(hut position, unit vector towards the square) per home slot, on one or two elliptical rings."""
    if n <= 30:
        rings = [(n, 292, 200, 0.0)]
    else:
        inner = int(n * 0.42)
        rings = [(inner, 208, 136, 0.0), (n - inner, 304, 214, 0.5)]
    homes = []
    for count, rx, ry, phase in rings:
        for k in range(count):
            a = (k + phase) / count * math.tau - math.pi / 2
            x, y = center[0] + rx * math.cos(a), center[1] + ry * math.sin(a)
            dx, dy = center[0] - x, center[1] - y
            d = math.hypot(dx, dy) or 1.0
            homes.append(((x, y), (dx / d, dy / d)))
    return homes
