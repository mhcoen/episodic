"""
Voice Grammar Pipeline.

Main entry point for parsing voice utterances into UtilityQuery AST nodes.
"""

from dataclasses import dataclass
from typing import Optional

from ..types import UtilityQuery
from .normalizer import Normalizer
from .lexer import Lexer
from .memory_detector import MemoryQueryDetector
from .preempt import PreemptRouter, RuntimeState
from .single_word import route_single_word
from .grammar import GrammarParser
from .confidence import ConfidenceCalculator, ParseFeatures
from .fuzzy import FuzzyMatcher


@dataclass
class VoiceParseResult:
    """Result of voice grammar parsing."""
    query: Optional[UtilityQuery]
    features: Optional[ParseFeatures]
    normalized_text: str
    bypassed: bool = False
    bypass_reason: Optional[str] = None
    preempted: bool = False


def parse_utterance(
    text: str,
    state: Optional[RuntimeState] = None,
) -> Optional[UtilityQuery]:
    """
    Main entry point for voice grammar parsing.

    Pipeline:
    1. Preempt check (STOP, REPEAT)
    2. Normalization
    3. Memory/past-tense check → bypass
    4. Single-word routing
    5. Lexer (maximal munch)
    6. Grammar parser
    7. Confidence calculation
    8. Return UtilityQuery or None

    Args:
        text: Raw input text
        state: Runtime state for stateful commands (optional)

    Returns:
        UtilityQuery if parsing succeeded with acceptable confidence, None otherwise
    """
    if state is None:
        state = RuntimeState()

    result = parse_utterance_full(text, state)

    if result.query is None:
        return None

    # Return None if bypassed (memory pattern detected)
    if result.bypassed:
        return None

    return result.query


def parse_utterance_full(
    text: str,
    state: Optional[RuntimeState] = None,
) -> VoiceParseResult:
    """
    Full voice grammar parsing with detailed result.

    Returns VoiceParseResult with full details including features and bypass info.
    """
    if state is None:
        state = RuntimeState()

    # Initialize components
    normalizer = Normalizer()
    lexer = Lexer()
    memory_detector = MemoryQueryDetector()
    preempt_router = PreemptRouter()
    grammar_parser = GrammarParser()
    confidence_calc = ConfidenceCalculator()
    fuzzy_matcher = FuzzyMatcher()

    # 1. Preempt check (on original text)
    preempt_query = preempt_router.preempt_check(text, state)
    if preempt_query:
        return VoiceParseResult(
            query=preempt_query,
            features=None,
            normalized_text=text,
            preempted=True,
        )

    # 2. Normalize text
    normalized = normalizer.normalize(text)

    # 3. Memory/past-tense check (on normalized tokens)
    tokens_for_memory = normalized.split()
    bypass, bypass_reason = memory_detector.should_bypass_utilities(tokens_for_memory)
    if bypass:
        return VoiceParseResult(
            query=None,
            features=None,
            normalized_text=normalized,
            bypassed=True,
            bypass_reason=bypass_reason,
        )

    # 4. Single-word routing
    words = normalized.split()
    if len(words) == 1:
        single_word_query = route_single_word(words[0], text)
        if single_word_query:
            return VoiceParseResult(
                query=single_word_query,
                features=ParseFeatures(
                    is_exact_template=True,
                    has_domain_keyword=True,
                    args_complete=True,
                    word_count=1,
                ),
                normalized_text=normalized,
            )

    # 5. Lexer
    tokens = lexer.tokenize(normalized)

    # 6. Check for opinion/explanation patterns (disqualifiers)
    features = ParseFeatures(word_count=len(words))
    if _has_opinion_request(normalized):
        features.has_opinion_request = True
    if _has_explanation_request(normalized):
        features.has_explanation_request = True

    # 7. Grammar parser
    match = grammar_parser.parse(normalized, tokens)

    if match is None:
        # No grammar match - try to still extract something useful
        # Check if this looks like a media play command (common case)
        if len(tokens) >= 2 and tokens[0].kind == "ACTION_PLAY":
            # "play X" pattern
            query_text = " ".join(t.value for t in tokens[1:])
            features.has_action_marker = True
            features.has_domain_keyword = False  # No explicit domain keyword
            features.args_complete = bool(query_text)

            confidence = confidence_calc.calculate(features, "media_play")

            return VoiceParseResult(
                query=UtilityQuery(
                    category="media",
                    command="media_play",
                    args={"query": query_text},
                    confidence=confidence,
                    source="voice",
                    raw_input=text,
                ),
                features=features,
                normalized_text=normalized,
            )

        return VoiceParseResult(
            query=None,
            features=features,
            normalized_text=normalized,
        )

    # 8. Merge features from grammar match
    features = match.features
    features.word_count = len(words)

    if _has_opinion_request(normalized):
        features.has_opinion_request = True
    if _has_explanation_request(normalized):
        features.has_explanation_request = True
    if _has_conjunction(tokens):
        features.has_conjunction = True

    # 9. Try fuzzy matching for unresolved tokens
    fuzzy_used = False
    for token in tokens:
        if token.kind == "WORD":
            corrected, used = fuzzy_matcher.try_fuzzy_match(
                token.value,
                _guess_token_class(match.command),
                [t.value for t in tokens]
            )
            if used:
                fuzzy_used = True
                break

    features.fuzzy_match_used = fuzzy_used

    # 10. Calculate confidence
    confidence = confidence_calc.calculate(features, match.command)

    # 11. Build UtilityQuery
    query = UtilityQuery(
        category=match.category,
        command=match.command,
        args=match.args,
        confidence=confidence,
        source="voice",
        raw_input=text,
    )

    return VoiceParseResult(
        query=query,
        features=features,
        normalized_text=normalized,
    )


def _has_opinion_request(text: str) -> bool:
    """Check if text contains opinion request patterns."""
    patterns = [
        "what do you think",
        "do you think",
        "your opinion",
        "what would you",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in patterns)


def _has_explanation_request(text: str) -> bool:
    """Check if text contains explanation request patterns."""
    patterns = [
        "tell me about",
        "explain",
        "how does",
        "how do",
        "what is a",
        "what are",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in patterns)


def _has_conjunction(tokens) -> bool:
    """Check if tokens contain a conjunction."""
    return any(t.kind == "CONJ" for t in tokens)


def _guess_token_class(command: str) -> str:
    """Guess the token class for fuzzy matching based on command."""
    if command.startswith("media_"):
        return "station_name"
    elif command.startswith("timer_"):
        return "time_unit"
    elif "weather" in command:
        return "domain_keyword"
    return "domain_keyword"
