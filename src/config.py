# Run configuration (SIMULATION_DESIGN.md §3.2, Appendix D): nested dataclasses mirroring the YAML sections.
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml

from resources import Resource

OFFER_NAMES = ("generous", "fair", "greedy")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _one_of(value, options: tuple, name: str) -> None:
    _check(value in options, f"{name} must be one of {options}, got {value!r}")


def _by_resource(d: dict, name: str) -> dict:
    try:
        out = {Resource(k): v for k, v in d.items()}
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    _check(set(out) == set(Resource), f"{name} must have exactly the keys food and wood")
    _check(all(v >= 0 for v in out.values()), f"{name} values must be >= 0")
    return out


@dataclass
class PopulationConfig:
    N: int = 24
    D: int = 20
    G: int = 40
    init_genes: Literal["uniform"] | list[float] = "uniform"

    def __post_init__(self):
        _check(self.N >= 2, "population.N must be >= 2")
        _check(self.D >= 1 and self.G >= 1, "population.D and population.G must be >= 1")
        if self.init_genes != "uniform":
            _check(isinstance(self.init_genes, list) and len(self.init_genes) == 4
                   and all(0 <= g <= 1 for g in self.init_genes),
                   "population.init_genes must be 'uniform' or a list of 4 floats in [0, 1]")


@dataclass
class AgentConfig:
    health_max: int = 10
    health_init: int = 10
    stock_init: dict[Resource, int] = field(default_factory=lambda: {"food": 2, "wood": 2})
    need: dict[Resource, int] = field(default_factory=lambda: {"food": 1, "wood": 1})
    miss_penalty: int = 2
    regen: int = 1
    rest_regen: int = 2

    def __post_init__(self):
        self.stock_init = _by_resource(self.stock_init, "agent.stock_init")
        self.need = _by_resource(self.need, "agent.need")
        _check(0 < self.health_init <= self.health_max, "agent.health_init must be in (0, health_max]")
        _check(min(self.miss_penalty, self.regen, self.rest_regen) >= 0, "agent penalties/regen must be >= 0")


@dataclass
class GatheringConfig:
    base_yield: float = 2.0
    far_multiplier: float = 1.5
    far_injury_p: float = 0.25
    far_injury_dmg: int = 2
    supply_per_capita: float = 1.2

    def __post_init__(self):
        _check(self.base_yield > 0 and self.supply_per_capita > 0, "gathering yields must be > 0")
        _check(self.far_multiplier >= 1, "gathering.far_multiplier must be >= 1")
        # Must be > 0: the route prompt renders it as "1 in {1/p}".
        _check(0 < self.far_injury_p <= 1, "gathering.far_injury_p must be in (0, 1]")
        _check(self.far_injury_dmg >= 0, "gathering.far_injury_dmg must be >= 0")


@dataclass
class HuntConfig:
    min_party: int = 2
    threshold_frac: float = 0.6
    effort_cost: int = 1
    payout: int = 4
    supply_per_capita: float = 2.0
    cancel_food: int = 1

    def __post_init__(self):
        _check(self.min_party >= 1, "hunt.min_party must be >= 1")
        _check(0 < self.threshold_frac <= 1, "hunt.threshold_frac must be in (0, 1]")
        _check(min(self.effort_cost, self.payout, self.supply_per_capita, self.cancel_food) >= 0,
               "hunt costs and payouts must be >= 0")


@dataclass
class RaidConfig:
    loot_frac: float = 0.5
    fight_dmg: int = 2
    max_candidates: int = 4

    def __post_init__(self):
        _check(0 <= self.loot_frac <= 1, "raid.loot_frac must be in [0, 1]")
        _check(self.fight_dmg >= 0, "raid.fight_dmg must be >= 0")
        # Candidates + "nobody" must respect the 6-option limit (§6.3).
        _check(1 <= self.max_candidates <= 5, "raid.max_candidates must be in [1, 5]")


@dataclass
class MarketConfig:
    enabled: bool = True
    spare_min: int = 2
    offers: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"generous": (2, 1), "fair": (1, 1), "greedy": (1, 2)})

    def __post_init__(self):
        self.offers = {k: tuple(v) for k, v in self.offers.items()}
        _check(bool(self.offers) and set(self.offers) <= set(OFFER_NAMES),
               f"market.offers keys must be a non-empty subset of {OFFER_NAMES}")
        _check(all(len(v) == 2 and min(v) >= 1 for v in self.offers.values()),
               "market.offers values must be [give, get] with both >= 1")
        _check(self.spare_min >= 1, "market.spare_min must be >= 1")


@dataclass
class GiftsConfig:
    enabled: bool = True
    spare_min: int = 3

    def __post_init__(self):
        _check(self.spare_min >= 1, "gifts.spare_min must be >= 1")


@dataclass
class SeasonConfig:
    schedule: Literal["stationary", "famine_food_second_half", "custom"] = "stationary"
    food: float = 1.0
    wood: float = 1.0

    def __post_init__(self):
        _one_of(self.schedule, ("stationary", "famine_food_second_half", "custom"), "season.schedule")
        _check(self.food > 0 and self.wood > 0, "season multipliers must be > 0")


@dataclass
class MemoryConfig:
    days: int = 5
    render: int = 2
    rep_threshold: int = 2

    def __post_init__(self):
        _check(self.days >= 0 and self.render >= 0, "memory.days and memory.render must be >= 0")
        _check(self.rep_threshold >= 1, "memory.rep_threshold must be >= 1")


@dataclass
class PromptsConfig:
    persona_mode: Literal["gated", "full", "none"] = "gated"
    render_reputation: bool = True
    render_memory: bool = True

    def __post_init__(self):
        _one_of(self.persona_mode, ("gated", "full", "none"), "prompts.persona_mode")


@dataclass
class BrainConfig:
    kind: Literal["laya", "mock"] = "laya"
    model: str = "aac6fef/laya-mlx"
    dtype: str = "float16"
    device: str = "gpu"
    batch_size: int = 8
    compile: bool = False
    cache_prompts: bool = False
    tau: float = 0.5
    cache: bool = False

    def __post_init__(self):
        _one_of(self.kind, ("laya", "mock"), "brain.kind")
        _check(self.batch_size >= 1, "brain.batch_size must be >= 1")
        _check(self.tau >= 0, "brain.tau must be >= 0 (0 = argmax)")


@dataclass
class EvolutionConfig:
    elites: int = 2
    tournament_k: int = 3
    p_mut: float = 0.15
    sigma_mut: float = 0.1
    wealth_ref: float = 10

    def __post_init__(self):
        _check(self.elites >= 0 and self.tournament_k >= 1, "evolution.elites >= 0 and tournament_k >= 1")
        _check(0 <= self.p_mut <= 1, "evolution.p_mut must be in [0, 1]")
        _check(self.sigma_mut >= 0 and self.wealth_ref > 0, "evolution.sigma_mut >= 0 and wealth_ref > 0")


@dataclass
class LogConfig:
    prompts: bool = True
    out_dir: str = "runs"


@dataclass
class Config:
    run_name: str = "baseline"
    seed: int = 0
    population: PopulationConfig = field(default_factory=PopulationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    gathering: GatheringConfig = field(default_factory=GatheringConfig)
    hunt: HuntConfig = field(default_factory=HuntConfig)
    raid: RaidConfig = field(default_factory=RaidConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    gifts: GiftsConfig = field(default_factory=GiftsConfig)
    season: SeasonConfig = field(default_factory=SeasonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def __post_init__(self):
        _check(self.evolution.elites <= self.population.N, "evolution.elites must be <= population.N")

    @classmethod
    def from_dict(cls, data: dict | None) -> "Config":
        """Build from a (possibly partial) dict; missing keys keep their defaults, unknown keys raise."""
        data = dict(data or {})
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            section = _SECTIONS.get(f.name)
            if section is None:
                kwargs[f.name] = value
                continue
            try:
                kwargs[f.name] = section(**(value or {}))
            except TypeError as e:
                raise ValueError(f"{f.name}: {e}") from None
        _check(not data, f"unknown config keys: {sorted(data)}")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Plain YAML/JSON-serializable dict (enum keys -> strings, tuples -> lists)."""
        return _plain(asdict(self))


_SECTIONS = {
    "population": PopulationConfig,
    "agent": AgentConfig,
    "gathering": GatheringConfig,
    "hunt": HuntConfig,
    "raid": RaidConfig,
    "market": MarketConfig,
    "gifts": GiftsConfig,
    "season": SeasonConfig,
    "memory": MemoryConfig,
    "prompts": PromptsConfig,
    "brain": BrainConfig,
    "evolution": EvolutionConfig,
    "log": LogConfig,
}


def _plain(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {_plain(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.from_dict(yaml.safe_load(f))
