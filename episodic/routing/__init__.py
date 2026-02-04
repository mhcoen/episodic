"""
Unified Router for Episodic.

Routes user input to the appropriate handler: utility commands, MQL queries, or LLM.
"""

from .types import RouteTarget, RouterResult
from .router import route

__all__ = ["route", "RouteTarget", "RouterResult"]
