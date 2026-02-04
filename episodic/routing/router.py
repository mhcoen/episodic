"""
Unified Router.

Routes user input to the appropriate handler based on:
1. Preempt check (STOP/REPEAT exact matches)
2. Memory pattern detector (past-tense → MQL)
3. Voice grammar parse (utility commands)
4. MQL parse (memory queries)
5. LLM fallback

CRITICAL: route() is side-effect-free. No DB writes, no scheduling during parse.
"""

import sqlite3
from datetime import datetime
from typing import Optional

from .types import RouteTarget, RouterResult
from episodic.utility.voice.preempt import PreemptRouter, RuntimeState
from episodic.utility.voice.memory_detector import MemoryQueryDetector
from episodic.utility.voice.pipeline import parse_utterance_full
from episodic.utility.voice.confidence import ConfidenceCalculator
from episodic.query.pipeline import parse_query
from episodic.query.types import FreeText


# Confidence thresholds by command class
THRESHOLDS = {
    "mutate": 0.80,
    "read": 0.55,
    "system": 0.70,
}


def route(
    text: str,
    state: Optional[RuntimeState] = None,
    conn: Optional[sqlite3.Connection] = None,
    now_utc: Optional[datetime] = None,
    user_tz: str = "America/Chicago"
) -> RouterResult:
    """
    Unified routing with explicit acceptance criteria.

    Order:
    1. Preempt check (STOP/REPEAT exact matches)
    2. Memory pattern detector (past-tense, "did we discuss", etc.)
       → If detected, route to MQL WITHOUT trying voice grammar
    3. Voice grammar parse
       → Accept only if confidence >= threshold for command class
       → Mutations require exact-template OR (args-complete AND domain-keyword AND no-fuzzy)
    4. MQL parse (on original text)
       → Accept if not FreeText
    5. Low-confidence utility with confirm
    6. LLM fallback

    CRITICAL: This function is side-effect-free. No DB writes, no scheduling.
    """
    if state is None:
        state = RuntimeState(timezone=user_tz)

    original = text
    voice_attempted = False
    mql_attempted = False

    preempt_router = PreemptRouter()
    memory_detector = MemoryQueryDetector()
    confidence_calc = ConfidenceCalculator()

    # 1. Preempt (exact matches only)
    preempt_query = preempt_router.preempt_check(text, state)
    if preempt_query:
        return RouterResult(
            target=RouteTarget.PREEMPT,
            utility_query=preempt_query,
            confidence=0.99,
            reason="preempt_exact",
            original_text=original,
        )

    # 2. Memory pattern detector BEFORE voice grammar
    # This ensures "what time did we discuss X" goes to MQL, not time_now
    tokens = text.lower().split()
    bypass, bypass_reason = memory_detector.should_bypass_utilities(tokens)
    if bypass:
        mql_attempted = True
        mql_result = parse_query(original, conn, now_utc, user_tz)
        # Check if MQL parsed successfully (not FreeText)
        if not _is_free_text(mql_result):
            return RouterResult(
                target=RouteTarget.MQL,
                mql_result=mql_result,
                confidence=0.90,
                reason=f"memory_pattern:{bypass_reason}",
                original_text=original,
                mql_parse_attempted=True,
            )

    # 3. Voice grammar parse (uses voice normalizer)
    voice_attempted = True
    voice_result = parse_utterance_full(text, state)

    if voice_result.query and not voice_result.bypassed:
        cmd_class = confidence_calc.classify_command(voice_result.query.command)
        threshold = THRESHOLDS.get(cmd_class, 0.70)

        # Acceptance criteria for mutations (hard gate)
        if cmd_class == "mutate" and voice_result.features:
            features = voice_result.features
            mutation_acceptable = (
                features.is_exact_template or
                (features.args_complete and features.has_domain_keyword and not features.fuzzy_match_used)
            )
            if not mutation_acceptable:
                # Force below threshold → will go to confirm or reject
                voice_result.query = voice_result.query._replace(
                    confidence=min(voice_result.query.confidence, threshold - 0.01)
                ) if hasattr(voice_result.query, '_replace') else voice_result.query

        if voice_result.query.confidence >= threshold:
            return RouterResult(
                target=RouteTarget.UTILITY,
                utility_query=voice_result.query,
                confidence=voice_result.query.confidence,
                reason="voice_accepted",
                original_text=original,
                voice_parse_attempted=True,
            )

    # 4. Try MQL (on original text, if not already tried)
    if not mql_attempted:
        mql_attempted = True
        mql_result = parse_query(original, conn, now_utc, user_tz)
        if not _is_free_text(mql_result):
            return RouterResult(
                target=RouteTarget.MQL,
                mql_result=mql_result,
                confidence=0.80,
                reason="mql_matched",
                original_text=original,
                voice_parse_attempted=voice_attempted,
                mql_parse_attempted=True,
            )

    # 5. Low-confidence utility (confirm mode)
    if voice_result.query and not voice_result.bypassed:
        if voice_result.query.confidence >= 0.50:
            return RouterResult(
                target=RouteTarget.UTILITY,
                utility_query=voice_result.query,
                confidence=voice_result.query.confidence,
                reason="voice_low_confidence_confirm",
                original_text=original,
                voice_parse_attempted=True,
                mql_parse_attempted=mql_attempted,
            )

    # 6. LLM fallback
    return RouterResult(
        target=RouteTarget.LLM,
        reason="no_grammar_match",
        original_text=original,
        voice_parse_attempted=voice_attempted,
        mql_parse_attempted=mql_attempted,
    )


def _is_free_text(mql_result) -> bool:
    """Check if MQL result is a FreeText fallback or weak match."""
    # ResolvedQuery doesn't directly tell us if it came from FreeText,
    # but we can check for meaningful MQL-specific content
    if mql_result is None:
        return True

    # Strong MQL signals: temporal constraint, explicit segment, deictic reference
    if mql_result.temporal:
        return False
    if mql_result.segment_explicit:
        return False
    if mql_result.deictic:
        return False
    if mql_result.has_broadness_cue:
        return False

    # If only target is set with no other MQL constraints, it's likely free text
    # (e.g., "tell me about timers" has target but no MQL structure)
    return True
