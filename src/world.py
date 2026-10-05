from genes import Genome, Gene
from enum import Enum
from typing import Literal
from dataclasses import dataclass, field
from logEvents import LogEvent
from resources import Resource

class MemoryEventKind(Enum):
    ROBBED_ME = "robbed_me"
    I_ROBBED = "i_robbed"
    GAVE_ME = "gave_me"
    I_GAVE = "i_gave"
    TRADED_FAIR = "traded_fair"
    TRADED_GENEROUS_TO_ME = "traded_generous_to_me"
    TRADED_GREEDY_TO_ME = "traded_greedy_to_me"
    REFUSED_ME = "refused_me"
    I_REFUSED = "i_refused"
    SHIRKED_IN_MY_HUNT = "shirked_in_my_hunt"
    WORKED_IN_MY_HUNT = "worked_in_my_hunt"

@dataclass
class MemoryEvent:
    day: int
    other_id: int
    kind: MemoryEventKind
    details: Resource | None

@dataclass
class Record:
    raids: int = 0
    gifts: int = 0
    contributed: int = 0
    free_rode: int = 0

@dataclass
class Agent:
    id: int
    genome: Genome
    health: int
    stock: dict[Resource, int]
    alive: bool = True
    days_alive: int = 0
    record: Record = field(default_factory=Record)
    memory: list[MemoryEvent] = field(default_factory=list)
    activity: Literal["gather_food", "gather_wood", "rest", "raid"] | None = None
    route: Literal["near", "far"] | None = None
    target: int | None = None
    traded_today: bool = False
    asked_gift_today: set[str] = field(default_factory=set)
    last_damage: str | None = None
    parents: tuple[int, ...] = ()

    @property
    def wealth(self) -> int:
        return sum(self.stock.values())

@dataclass
class Hunt:
    took_place: bool
    success: bool | None = None
    contributors: int = 0
    party_size: int = 0


@dataclass
class Day:
    hunt: Hunt
    congestion: dict[Resource, float]
    raids: int

@dataclass
class Village:
    agents: list[Agent]
    day_count: int
    season: dict[Resource, float]
    yesterday: Day
    log: list[LogEvent]

    @property
    def n_alive(self) -> int:
        return sum(a.alive for a in self.agents)

    def agent(self, agent_id: int) -> Agent:
        return next(a for a in self.agents if a.id == agent_id)


def living_others(village, agent: Agent) -> list[Agent]:
    return [a for a in village.agents if a.alive and a.health > 0 and a.id != agent.id]

def mult_r(agent: Agent, r: Resource) -> float:
    g_spec = agent.genome[Gene.SPECIALIZATION]
    return 0.5 + g_spec if r == Resource.FOOD else 1.5 - g_spec