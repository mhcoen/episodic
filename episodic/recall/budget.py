"""
Intent to budget mapping.

Maps parser output (query_form, has_broadness_cue, speaker) to retrieval budget.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .expansion import Tier


class IntentClass(Enum):
    """High-level intent classes for recall."""
    CONVERSATION_RECALL_LOCATE = auto()   # "when we discussed X"
    CONVERSATION_RECALL_SUMMARIZE = auto()  # "what did we discuss about X"
    EXISTENCE_CHECK = auto()               # "have we talked about X"
    STATEMENT_RECALL = auto()              # "what did you say about X"
    BROWSE = auto()                        # explicit browse mode


@dataclass
class RecallBudget:
    """Budget allocation for recall retrieval."""
    intent: IntentClass
    max_topics: int
    max_statements: int
    topic_tier: Tier
    
    # Horizon control (affected by broadness_cue)
    broad_horizon: bool  # True = search full history; False = recent bias
    overfetch_multiplier: float  # How much to overfetch from semantic search
    
    # Output emphasis
    emphasize_timestamps: bool  # For locate intent
    emphasize_summary: bool     # For summarize intent


# Default budgets by intent
_BUDGETS = {
    IntentClass.CONVERSATION_RECALL_LOCATE: RecallBudget(
        intent=IntentClass.CONVERSATION_RECALL_LOCATE,
        max_topics=2,
        max_statements=1,
        topic_tier=Tier.B,
        broad_horizon=False,
        overfetch_multiplier=3.0,
        emphasize_timestamps=True,
        emphasize_summary=False,
    ),
    IntentClass.CONVERSATION_RECALL_SUMMARIZE: RecallBudget(
        intent=IntentClass.CONVERSATION_RECALL_SUMMARIZE,
        max_topics=2,
        max_statements=1,
        topic_tier=Tier.B,
        broad_horizon=False,
        overfetch_multiplier=3.0,
        emphasize_timestamps=False,
        emphasize_summary=True,
    ),
    IntentClass.EXISTENCE_CHECK: RecallBudget(
        intent=IntentClass.EXISTENCE_CHECK,
        max_topics=1,
        max_statements=2,
        topic_tier=Tier.A,
        broad_horizon=False,  # Will be overridden by broadness_cue
        overfetch_multiplier=2.0,
        emphasize_timestamps=True,
        emphasize_summary=False,
    ),
    IntentClass.STATEMENT_RECALL: RecallBudget(
        intent=IntentClass.STATEMENT_RECALL,
        max_topics=1,  # Only if strong clustering
        max_statements=3,
        topic_tier=Tier.A,
        broad_horizon=False,
        overfetch_multiplier=2.0,
        emphasize_timestamps=True,
        emphasize_summary=False,
    ),
    IntentClass.BROWSE: RecallBudget(
        intent=IntentClass.BROWSE,
        max_topics=3,
        max_statements=0,
        topic_tier=Tier.B,
        broad_horizon=False,
        overfetch_multiplier=1.0,
        emphasize_timestamps=False,
        emphasize_summary=False,
    ),
}


def map_parser_output_to_budget(
    query_form: Optional[str],
    has_broadness_cue: bool,
    speaker: Optional[str],
    mode: Optional[str] = None
) -> RecallBudget:
    """
    Map parser output to recall budget.
    
    Args:
        query_form: From DiscussionQuery.query_form 
            ("when_we", "what_we", "have_we", "did_speaker")
        has_broadness_cue: True if "ever/before/previously" present
        speaker: From ResolvedQuery.speaker ("user", "assistant", None)
        mode: From ResolvedQuery.mode ("browse", "answer", "summarize")
    
    Returns:
        RecallBudget with appropriate settings
    """
    # Determine intent class
    intent = _classify_intent(query_form, speaker, mode)
    
    # Get base budget
    budget = _BUDGETS[intent]
    
    # Apply modifiers
    budget = _apply_broadness_modifier(budget, has_broadness_cue)
    budget = _apply_speaker_modifier(budget, speaker)
    
    return budget


def _classify_intent(
    query_form: Optional[str],
    speaker: Optional[str],
    mode: Optional[str]
) -> IntentClass:
    """Classify intent from parser output."""
    
    # Explicit browse mode
    if mode == "browse" and query_form is None:
        return IntentClass.BROWSE
    
    # Discussion query forms
    if query_form == "when_we":
        return IntentClass.CONVERSATION_RECALL_LOCATE
    
    if query_form == "what_we":
        return IntentClass.CONVERSATION_RECALL_SUMMARIZE
    
    if query_form == "have_we":
        # Always existence check regardless of broadness cue
        return IntentClass.EXISTENCE_CHECK
    
    if query_form == "did_speaker":
        return IntentClass.STATEMENT_RECALL
    
    # Fallback based on mode
    if mode == "browse":
        return IntentClass.BROWSE
    if mode == "summarize":
        return IntentClass.CONVERSATION_RECALL_SUMMARIZE
    
    # Default to conversation recall
    return IntentClass.CONVERSATION_RECALL_LOCATE


def _apply_broadness_modifier(budget: RecallBudget, has_broadness_cue: bool) -> RecallBudget:
    """Apply broadness cue modifier (affects horizon, not intent)."""
    if not has_broadness_cue:
        return budget
    
    # Create modified copy
    return RecallBudget(
        intent=budget.intent,
        max_topics=budget.max_topics,
        max_statements=budget.max_statements,
        topic_tier=budget.topic_tier,
        broad_horizon=True,  # Override
        overfetch_multiplier=budget.overfetch_multiplier * 1.5,  # Increase overfetch
        emphasize_timestamps=budget.emphasize_timestamps,
        emphasize_summary=budget.emphasize_summary,
    )


def _apply_speaker_modifier(budget: RecallBudget, speaker: Optional[str]) -> RecallBudget:
    """
    Apply speaker filter modifier.
    
    When speaker != None (i.e., filtering to user or assistant only),
    semantic retrieval is disabled and lexical results don't cluster as well,
    so bias toward statement blocks.
    """
    if speaker is None:
        return budget
    
    # Reduce topic confidence when speaker-filtered
    # (lexical-only hits cluster less reliably)
    return RecallBudget(
        intent=budget.intent,
        max_topics=max(0, budget.max_topics - 1),  # Reduce topic budget
        max_statements=budget.max_statements + 1,   # Increase statement budget
        topic_tier=Tier.A if budget.topic_tier == Tier.B else budget.topic_tier,  # Downgrade tier
        broad_horizon=budget.broad_horizon,
        overfetch_multiplier=budget.overfetch_multiplier,
        emphasize_timestamps=budget.emphasize_timestamps,
        emphasize_summary=budget.emphasize_summary,
    )


def get_budget_description(budget: RecallBudget) -> str:
    """Human-readable budget description for debugging."""
    return (
        f"Intent: {budget.intent.name}, "
        f"Topics: {budget.max_topics} (Tier {budget.topic_tier.name}), "
        f"Statements: {budget.max_statements}, "
        f"Horizon: {'broad' if budget.broad_horizon else 'narrow'}"
    )
