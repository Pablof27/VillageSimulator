from enum import Enum
from math import floor
from random import random, uniform
from resources import Resource

class Gene(Enum):
    AGGRESSION = "aggression"
    ALTRUISM = "altruism"
    RISK = "risk"
    SPECIALIZATION = "specialization"

type Genome = dict[Gene, float]

PERSONA = {
    Gene.AGGRESSION: (
        "You are a gentle farmer who abhors violence and never takes what is not yours.",
        "You are a quiet villager who dislikes conflict.",
        "You are an ordinary villager who takes care of your own.",
        "You are a tough brawler who takes what you need.",
        "You are a feared raider who lives by taking from others.",
    ),
    Gene.ALTRUISM: (
        "You are a miser who never shares anything with anyone.",
        "You are a self-reliant villager who seldom helps others.",
        "You are an ordinary villager who helps when it is convenient.",
        "You are a kind neighbour who likes to help others.",
        "You are a devoted helper who always shares with those in need.",
    ),
    Gene.RISK: (
        "You are extremely cautious and avoid every risk.",
        "You prefer safe, predictable choices.",
        "You take moderate risks when the reward is worth it.",
        "You are bold and enjoy taking chances.",
        "You are a reckless gambler who loves danger.",
    ),
    Gene.SPECIALIZATION: (
        "You are a master woodcutter but a poor forager.",
        "You are a good woodcutter and a mediocre forager.",
        "You are equally able at foraging and woodcutting.",
        "You are a good forager and a mediocre woodcutter.",
        "You are a master forager but a poor woodcutter.",
    ),
}

def level(g: float) -> int:
    return min(4, floor(5*g))

def persona(genome: Genome, genes: tuple[Gene, ...], mode: str = "gated") -> str:
    if mode == "none":
        return ""
    if mode == "full":
        genes = tuple(Gene)
    return " ".join(PERSONA[g][level(genome[g])] for g in genes)

def gather_mult(genome: Genome, r: Resource) -> float:
    return 0.5 + genome[Gene.SPECIALIZATION] if r == Resource.FOOD else 1.5 - genome[Gene.SPECIALIZATION]

def crossover(g1: Genome, g2: Genome) -> Genome:
    child = {}
    for gene in Gene:
        child[gene] = (g1[gene] + g2[gene]) / 2
    return child

# TODO: Revisar rango y distribución
def mutate(g: Genome, rate: float) -> Genome:
    child = g.copy()
    for gene in Gene:
        if random() < rate:
            child[gene] = min(1.0, max(0.0, child[gene] + uniform(-0.1, 0.1)))
    return child