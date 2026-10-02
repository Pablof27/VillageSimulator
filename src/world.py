from genes import Genome
from enum import Enum
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
    activity: str | None = None
    route: str
    target: int | None = None
    traded_today: bool = False
    asked_gift_today: set[str] = field(default_factory=set)

@dataclass
class Hunt:
    took_place: bool
    success: bool | None
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