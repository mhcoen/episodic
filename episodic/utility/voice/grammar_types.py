"""
Grammar Type Definitions.

Shared dataclasses used by grammar.py and grammar_calendar_email.py.
Extracted to break circular imports.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .confidence import ParseFeatures


@dataclass
class GrammarMatch:
    """Result of a grammar rule match."""
    command: str
    category: str
    args: Dict[str, Any]
    features: ParseFeatures
    consumed_tokens: int


@dataclass
class GrammarRule:
    """A grammar rule definition."""
    name: str
    category: str
    command: str
    patterns: List[List[str]]  # List of token kind patterns
    required_args: List[str]
    optional_args: List[str]
    is_exact_template: bool = False
    arg_extractor: Optional[Callable] = field(default=None, repr=False)
