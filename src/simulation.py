# Generation loop (SIMULATION_DESIGN.md §7.1): spawn -> D days -> metrics -> reproduce, G times.
import random
from dataclasses import dataclass, field
from pathlib import Path

from brain import make_brain
from config import Config
from evolution import init_population, reproduce, spawn
from logEvents import RunLog
from phases import run_day
from resources import Resource
from world import Agent, Day, Hunt, Village


@dataclass
class RunResult:
    run_dir: Path
    generations: list[dict] = field(default_factory=list)


def run(cfg: Config, run_dir: str | Path | None = None, brain=None, progress=print) -> RunResult:
    rng = random.Random(cfg.seed)
    brain = brain or make_brain(cfg.brain, rng)
    offspring = init_population(cfg, rng)
    next_id = 0

    with RunLog(cfg, run_dir) as log:
        result = RunResult(log.run_dir)
        for gen in range(cfg.population.G):
            agents = spawn(offspring, cfg, next_id)
            next_id += len(agents)
            log.start_generation(gen)
            run_generation(cfg, agents, gen, brain, rng, log)
            row = log.end_generation(agents)
            result.generations.append(row)
            progress(_summary(row, cfg))
            offspring = reproduce(agents, cfg, rng)
    return result


def run_generation(cfg: Config, agents: list[Agent], gen: int, brain, rng, log) -> Village:
    village = Village(agents, 0, season_for(cfg, gen, 1), Day(Hunt(False), {r: 1.0 for r in Resource}, 0), [])
    for day in range(1, cfg.population.D + 1):
        village.season = season_for(cfg, gen, day)
        run_day(cfg, village, brain, rng, log)
        if village.n_alive < 2:
            break
    return village


def season_for(cfg: Config, gen: int, day: int) -> dict[Resource, float]:
    """Daily resource multipliers (§3.1); `famine_food_second_half` is experiment E3."""
    season = cfg.season
    base = {Resource.FOOD: season.food, Resource.WOOD: season.wood}
    if season.schedule == "stationary":
        return base
    if season.schedule == "famine_food_second_half":
        famine = day > cfg.population.D // 2
        return {**base, Resource.FOOD: 0.5 if famine else season.food}
    raise NotImplementedError(f"season.schedule {season.schedule!r} is not implemented yet")


def _summary(row: dict, cfg: Config) -> str:
    genes = " ".join(f"{g}={row[f'mean_{g}']:.2f}" for g in ("agg", "alt", "risk", "spec"))
    return (f"gen {row['gen'] + 1:>3}/{cfg.population.G}  survivors {row['survivors']:>2}/{cfg.population.N}  "
            f"{genes}  raid={row['raid_rate']:.2f} contrib={row['contribution_rate']:.2f} "
            f"trade={row['trade_rate']:.2f}  {row['wall_seconds']:.0f}s")
