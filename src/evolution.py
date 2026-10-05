# Evolution (SIMULATION_DESIGN.md §7): fitness, selection, crossover/mutation and the next population.
from dataclasses import dataclass

from config import Config
from genes import Gene, Genome, crossover, mutate
from world import Agent


@dataclass
class Offspring:
    genome: Genome
    parents: tuple[int, ...]


def fitness(agent, cfg) -> float:
    base = agent.days_alive / cfg.population.D
    health_factor = 0.5 * agent.health / cfg.agent.health_max
    wealth_factor = 0.5 * min(1, agent.wealth / cfg.evolution.wealth_ref)
    return base + agent.alive * (health_factor + wealth_factor)


def init_population(cfg: Config, rng) -> list[Offspring]:
    init = cfg.population.init_genes
    if init == "uniform":
        return [Offspring({g: rng.random() for g in Gene}, ()) for _ in range(cfg.population.N)]
    return [Offspring(dict(zip(Gene, init)), ()) for _ in range(cfg.population.N)]


def reproduce(agents: list[Agent], cfg: Config, rng) -> list[Offspring]:
    evo = cfg.evolution
    scores = {a.id: fitness(a, cfg) for a in agents}
    ranked = sorted(select_parents(agents, scores), key=lambda a: scores[a.id], reverse=True)

    population = [Offspring(dict(a.genome), (a.id,)) for a in ranked[:evo.elites]]
    while len(population) < cfg.population.N:
        mother = _tournament(ranked, scores, evo.tournament_k, rng)
        father = _tournament([a for a in ranked if a is not mother] or ranked, scores, evo.tournament_k, rng)
        child = mutate(crossover(mother.genome, father.genome, rng), evo.p_mut, evo.sigma_mut, rng)
        population.append(Offspring(child, (mother.id, father.id)))
    return population


def select_parents(agents: list[Agent], scores: dict[int, float]) -> list[Agent]:
    survivors = [a for a in agents if a.alive]
    if len(survivors) >= 2:
        return survivors
    return sorted(agents, key=lambda a: scores[a.id], reverse=True)[:2]


def spawn(offspring: list[Offspring], cfg: Config, first_id: int) -> list[Agent]:
    return [Agent(first_id + i, o.genome, cfg.agent.health_init, dict(cfg.agent.stock_init), parents=o.parents)
            for i, o in enumerate(offspring)]


def _tournament(candidates: list[Agent], scores: dict[int, float], k: int, rng) -> Agent:
    return max((rng.choice(candidates) for _ in range(k)), key=lambda a: scores[a.id])