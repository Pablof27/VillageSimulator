from genes import Genome

def fitness(agent, cfg) -> float:
    base = agent.days_alive / cfg.population.D
    health_factor = 0.5 * agent.health / cfg.agent.max_health
    wealth_factor = 0.5 * min(1, agent.wealth() / cfg.evolution.wealth_ref)
    return base + agent.alivea * (health_factor + wealth_factor)


def reproduce(village, cfg, rng) -> list[Genome]:
    pass