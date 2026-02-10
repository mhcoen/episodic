"""
Router Types.

Sum-type router result for unified routing decisions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from episodic.utility.types import UtilityQuery
from episodic.query.types import ResolvedQuery


class RouteTarget(Enum):
    """Where to route the user input."""
    PREEMPT = auto()    # Immediate action (STOP/REPEAT)
    UTILITY = auto()    # Utility command
    MQL = auto()        # Memory query
    LLM = auto()        # Fall through to LLM


@dataclass(frozen=True)
class RouterResult:
    """
    Sum type for routing decisions.
    Exactly ONE of utility_query or mql_result is set (or neither for LLM).
    """
    target: RouteTarget
    utility_query: Optional[UtilityQuery] = None
    mql_result: Optional[ResolvedQuery] = None
    confidence: float = 0.0
    reason: str = ""

    # Audit fields
    original_text: str = ""
    voice_parse_attempted: bool = False
    mql_parse_attempted: bool = False

    def __post_init__(self) -> None:
        # Invariant: at most one payload
        if self.utility_query and self.mql_result:
            raise ValueError("RouterResult cannot have both utility_query and mql_result")
