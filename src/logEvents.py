from enum import Enum
from typing import Literal
from dataclasses import dataclass, asdict
import json

class LogEventType(Enum):
    decision = "decision"
    gather = "gather"
    hunt = "hunt"
    raid = "raid"
    trade = "trade"
    gift = "gift"
    night = "night"
    death = "death"
    generation = "generation"

@dataclass
class LogEvent:
    run: str
    gen: int
    day: int
    phase: Literal["morning", "gathering", "hunting", "raid", "trade", "gift", "night"]
    event_type: LogEventType
    payload: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
    