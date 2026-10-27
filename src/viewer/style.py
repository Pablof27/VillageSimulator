# Colours, gene colour ramps and fonts of the village viewer.
import pygame

GENES = ("aggression", "altruism", "risk", "specialization")
TITLE = {"aggression": "Aggression", "altruism": "Altruism", "risk": "Risk", "specialization": "Specialization"}
ENDS = {"aggression": ("gentle", "raider"), "altruism": ("miser", "helper"),
        "risk": ("cautious", "reckless"), "specialization": ("woodcutter", "forager")}
LEVELS = {
    "aggression": ("gentle farmer", "quiet villager", "ordinary villager", "tough brawler", "feared raider"),
    "altruism": ("miser", "self-reliant", "helps if convenient", "kind neighbour", "devoted helper"),
    "risk": ("extremely cautious", "plays it safe", "moderate risks", "bold", "reckless gambler"),
    "specialization": ("master woodcutter", "good woodcutter", "generalist", "good forager", "master forager"),
}
# Genes with a fixed accessory on the blob; the body colour shows the selected gene (aggression by default).
ACCESSORY = {"altruism": "halo", "risk": "eyebrows", "specialization": "belly"}

_RAMPS = {
    "aggression": ((66, 133, 226), (152, 104, 204), (230, 70, 66)),
    "altruism": ((148, 150, 162), (208, 184, 112), (247, 190, 36)),
    "risk": ((46, 178, 166), (112, 148, 200), (180, 70, 198)),
    "specialization": ((139, 90, 50), (186, 166, 86), (78, 168, 76)),
}

INK = (43, 49, 64)
MUTED = (122, 128, 142)
FAINT = (196, 198, 204)
PAPER = (252, 251, 247)
PANEL = (231, 227, 216)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GROUND = (237, 232, 216)
PATH = (226, 213, 186)
MEADOW = (213, 233, 176)
MEADOW_FAR = (190, 214, 150)
GROVE = (204, 226, 186)
FOREST_FAR = (172, 202, 160)
SAVANNA = (240, 226, 176)
PLAZA = (225, 216, 197)
TREE = (98, 156, 88)
TREE_FAR = (72, 126, 70)
FOOD = (76, 182, 80)
FOOD_DARK = (36, 116, 50)
WOOD = (156, 104, 58)
WOOD_DARK = (102, 66, 36)
WOOD_LIGHT = (222, 182, 124)
HEART = (232, 72, 96)
GOLD = (247, 197, 49)
GOOD = (60, 168, 92)
BAD = (222, 66, 66)
WARN = (236, 150, 40)
NIGHT = (92, 104, 160)

ACTIVITIES = ("gather_food", "gather_wood", "hunt", "raid", "rest")
ACTIVITY_LABEL = {"gather_food": "Forage", "gather_wood": "Chop wood", "hunt": "Hunt", "raid": "Raid", "rest": "Rest"}
ACTIVITY_COLOR = {"gather_food": FOOD, "gather_wood": WOOD, "hunt": (196, 140, 64), "raid": BAD, "rest": (130, 140, 200)}
DEATH_COLOR = {"starvation": (150, 120, 90), "injury": WARN, "fight": BAD, "hunt_exhaustion": (190, 150, 60)}
DEATH_LABEL = {"starvation": "starved/froze", "injury": "injury", "fight": "fight", "hunt_exhaustion": "exhaustion"}
PHASES = ("Morning", "Work", "Raids", "Market", "Gifts", "Night")

_FONTS = {False: ("sfnsrounded", "avenirnext", "helveticaneue", "arial"),
          True: ("arialrounded", "sfnsrounded", "avenirnextdemibold", "arialbold")}
_font_paths: dict[bool, str | None] = {}


def level(x: float) -> int:
    return min(4, int(5 * x))


def mix(a, b, u: float) -> tuple[int, int, int]:
    return tuple(round(a[j] + (b[j] - a[j]) * u) for j in range(3))


def shade(c, f: float) -> tuple[int, int, int]:
    """f < 1 darkens, f > 1 lightens towards white."""
    return tuple(round(v * f) for v in c[:3]) if f <= 1 else mix(c, WHITE, min(1.0, f - 1))


def ramp(gene: str, x: float) -> tuple[int, int, int]:
    stops = _RAMPS[gene]
    x = min(1.0, max(0.0, x)) * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    return mix(stops[i], stops[i + 1], x - i)


def font_path(bold: bool) -> str | None:
    if bold not in _font_paths:
        _font_paths[bold] = next((p for name in _FONTS[bold] if (p := pygame.font.match_font(name))), None)
    return _font_paths[bold]
