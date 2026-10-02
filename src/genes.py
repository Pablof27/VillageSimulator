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

def level(g: float) -> int:
    return min(4, floor(5*g))

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