"""Core data types for the MCP intent extraction pipeline."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ArgDefinition:
    """An argument within an intent definition."""
    type: str              # "string", "boolean", "list"
    description: str
    required: bool = False
    examples: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentDefinition:
    """An intent that the extraction LLM can recognize."""
    intent_id: str         # Stable ID: "cal_query" (never changes)
    name: str              # Display name: "calendar.query"
    description: str
    action_class: str      # "read", "draft", "write"
    args: Dict[str, ArgDefinition] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionResult:
    """Result from the extraction LLM call."""
    intent: Optional[str]          # Intent name or None
    args: Dict[str, Any]           # Extracted args (empty if null intent)
    confidence: float              # Logged only, not used for dispatch
    followup_suggestion: Optional[str]  # Secondary request if compound utterance
    raw_json: str                  # The raw JSON string from the LLM


@dataclass(frozen=True)
class PluginExtractionContribution:
    """Extraction-pipeline contributions from a plugin."""
    gate_keywords: List[str]
    gate_phrases: List[List[str]]
    intents: List["IntentDefinition"]
    contacts: Dict[str, str] = field(default_factory=dict)
    context_provider: Optional[Callable] = field(default=None, hash=False)


@dataclass(frozen=True)
class DispatchabilityVerdict:
    """Core-computed dispatchability decision."""
    dispatchable: bool
    intent: Optional[str]
    args: Dict[str, Any]
    action_class: Optional[str]
    missing_required_args: List[str]   # Non-empty means ask follow-up
    is_unknown_command: bool           # router.unknown_command
    unknown_command_hint: Optional[str]
    followup_suggestion: Optional[str]
    error: Optional[str]               # JSON parse fail, unregistered intent, etc.
