"""
Voice Grammar Tests.

Tests for the voice grammar parser including:
- Positive cases (from spec Section 10)
- Negative cases (must not mutate, LLM fallback)
- Trap idioms
- Mutation false positive rate
"""

import pytest
from typing import List, Tuple, Dict, Optional

from episodic.utility.voice import parse_utterance
from episodic.utility.voice.pipeline import parse_utterance_full, VoiceParseResult
from episodic.utility.voice.confidence import ConfidenceCalculator
from episodic.utility.voice.normalizer import Normalizer
from episodic.utility.voice.lexer import Lexer
from episodic.utility.voice.memory_detector import MemoryQueryDetector
from episodic.routing import route, RouteTarget


# Positive cases: (utterance, command, min_confidence, args_complete)
# Note: Confidence thresholds reflect the scoring system:
# - domain_keyword: +0.35
# - action/query marker: +0.25
# - args_complete: +0.25
# - Shorthand forms (no action marker) get lower confidence but still work
POSITIVE_CASES: Dict[str, List[Tuple[str, float, bool]]] = {
    "time_now": [
        ("what time is it", 0.90, True),
        ("time", 0.85, True),
        ("what is the time", 0.90, True),
    ],
    "date_today": [
        ("what is the date", 0.90, True),
        ("today", 0.80, True),
    ],
    "timer_set": [
        ("set a timer for 10 minutes", 0.80, True),
        ("timer 5 minutes", 0.55, True),  # Shorthand - lower confidence but valid
    ],
    "alarm_set": [
        ("set an alarm for 7am", 0.55, True),  # Args extracted but lower confidence
    ],
    "weather_now": [
        ("what is the weather", 0.80, True),
        ("weather", 0.80, True),
    ],
    "media_play": [
        ("play npr", 0.50, True),
        ("play some jazz", 0.50, True),
    ],
}

# Must NOT parse as mutation with confidence >= 0.80
NEGATIVE_MUST_NOT_MUTATE: List[str] = [
    "tell me about timers",
    "what do you think about alarms",
    "how does a timer work",
    "explain reminders to me",
    "I'm thinking about setting an alarm",
    "remind me why I'm here",
    "play along with me",
]

# Should not match any utility (fall to LLM)
NEGATIVE_LLM_FALLBACK: List[str] = [
    "tell me a story",
    "what's the capital of france",
    "how do I learn python",
    "why is the sky blue",
    "hello there",
]

# Trap idioms: (utterance, forbidden_command)
TRAP_IDIOMS: List[Tuple[str, str]] = [
    ("stop by the store", "media_stop"),
    ("remind me why", "remind_set"),
    ("play it by ear", "media_play"),
]

# Memory bypass cases - should route to MQL, not utility
MEMORY_BYPASS_CASES: List[str] = [
    "what time did the meeting start",
    "when did we discuss that",
    "did I mention the project",
    "what did we talk about yesterday",
]


class TestPositiveCases:
    """Test that valid utility commands are recognized."""

    @pytest.mark.parametrize("command,cases", list(POSITIVE_CASES.items()))
    def test_positive_cases(self, command: str, cases: List[Tuple[str, float, bool]]):
        for utterance, min_confidence, args_complete in cases:
            result = parse_utterance_full(utterance)
            assert result.query is not None, f"Failed to parse: {utterance}"
            assert result.query.command == command, \
                f"Wrong command for {utterance}: got {result.query.command}, expected {command}"
            assert result.query.confidence >= min_confidence, \
                f"Low confidence for {utterance}: {result.query.confidence} < {min_confidence}"


class TestNegativeCases:
    """Test that non-utility utterances are not misclassified."""

    def test_negative_must_not_mutate(self):
        """Utterances that must NOT trigger mutations above threshold."""
        calc = ConfidenceCalculator()

        for utterance in NEGATIVE_MUST_NOT_MUTATE:
            result = parse_utterance_full(utterance)

            if result.query is not None:
                command_class = calc.classify_command(result.query.command)
                if command_class == "mutate":
                    assert result.query.confidence < calc.THRESHOLDS["mutate"], \
                        f"DANGEROUS: Mutation FP: {utterance} -> {result.query.command} @ {result.query.confidence}"

    def test_negative_llm_fallback(self):
        """Utterances that should fall through to LLM."""
        for utterance in NEGATIVE_LLM_FALLBACK:
            result = route(utterance)
            assert result.target == RouteTarget.LLM, \
                f"Expected LLM fallback for '{utterance}', got {result.target}"


class TestTrapIdioms:
    """Test that trap idioms are not misclassified."""

    @pytest.mark.parametrize("utterance,forbidden_command", TRAP_IDIOMS)
    def test_trap_idiom(self, utterance: str, forbidden_command: str):
        calc = ConfidenceCalculator()
        result = parse_utterance_full(utterance)

        if result.query is not None and result.query.command == forbidden_command:
            threshold = calc.THRESHOLDS.get(
                calc.classify_command(result.query.command), 0.70
            )
            assert result.query.confidence < threshold, \
                f"Trap idiom matched: {utterance} -> {result.query.command} @ {result.query.confidence}"


class TestMemoryBypass:
    """Test that past-tense queries bypass utilities."""

    @pytest.mark.parametrize("utterance", MEMORY_BYPASS_CASES)
    def test_memory_pattern_bypass(self, utterance: str):
        result = parse_utterance_full(utterance)
        assert result.bypassed, \
            f"Expected memory bypass for: {utterance}"

    @pytest.mark.parametrize("utterance", MEMORY_BYPASS_CASES)
    def test_memory_route_not_utility(self, utterance: str):
        result = route(utterance)
        assert result.target != RouteTarget.UTILITY, \
            f"Memory utterance routed to utility: {utterance}"


class TestMutationFalsePositiveRate:
    """Test aggregate mutation false positive rate."""

    def test_mutation_fp_rate(self):
        """Aggregate FP rate for mutations must be < 5%."""
        calc = ConfidenceCalculator()
        all_negatives = NEGATIVE_MUST_NOT_MUTATE + NEGATIVE_LLM_FALLBACK

        false_positives = 0
        for utterance in all_negatives:
            result = parse_utterance_full(utterance)
            if result.query is not None:
                if calc.classify_command(result.query.command) == "mutate":
                    if result.query.confidence >= calc.THRESHOLDS["mutate"]:
                        false_positives += 1

        fp_rate = false_positives / len(all_negatives)
        assert fp_rate < 0.05, f"Mutation FP rate: {fp_rate:.2%}"


class TestNormalizer:
    """Test the normalizer component."""

    def test_contractions(self):
        n = Normalizer()
        assert "what is" in n.normalize("what's the time")
        assert "it is" in n.normalize("it's raining")

    def test_edge_fillers(self):
        n = Normalizer()
        assert n.normalize("um what time is it") == "what time is it"
        assert n.normalize("hey weather") == "weather"
        assert n.normalize("okay set a timer") == "set a timer"

    def test_number_words(self):
        n = Normalizer()
        assert "10" in n.normalize("set a timer for ten minutes")
        assert "25" in n.normalize("timer for twenty five minutes")

    def test_letter_sequences(self):
        n = Normalizer()
        assert "npr" in n.normalize("play n p r")
        assert "bbc" in n.normalize("play b b c")

    def test_edge_filler_boundary(self):
        """Edge fillers should not strip from inside words."""
        n = Normalizer()
        # "er" is a filler but should not strip from "weather"
        assert n.normalize("weather") == "weather"
        assert n.normalize("whether") == "whether"


class TestLexer:
    """Test the lexer component."""

    def test_multiword_tokens(self):
        lexer = Lexer()
        tokens = lexer.tokenize("what is the time")
        assert tokens[0].kind == "QUERY"
        assert tokens[0].value == "what is"

    def test_keyword_tokens(self):
        lexer = Lexer()
        tokens = lexer.tokenize("set a timer")
        assert tokens[0].kind == "ACTION_SET"
        assert tokens[2].kind == "KW_TIMER"

    def test_number_tokens(self):
        lexer = Lexer()
        tokens = lexer.tokenize("timer 10 minutes")
        assert tokens[1].kind == "NUMBER"


class TestMemoryDetector:
    """Test the memory query detector."""

    def test_past_tense_detection(self):
        detector = MemoryQueryDetector()

        # Should bypass
        bypass, reason = detector.should_bypass_utilities(["what", "time", "did", "the", "meeting", "start"])
        assert bypass
        assert "did" in reason

    def test_present_tense_no_bypass(self):
        detector = MemoryQueryDetector()

        # Should NOT bypass
        bypass, _ = detector.should_bypass_utilities(["what", "time", "is", "it"])
        assert not bypass


class TestRouting:
    """Test the unified router."""

    def test_utility_routing(self):
        result = route("what time is it")
        assert result.target == RouteTarget.UTILITY
        assert result.utility_query.command == "time_now"

    def test_memory_routing(self):
        result = route("what did we discuss yesterday")
        assert result.target == RouteTarget.MQL

    def test_llm_fallback(self):
        result = route("hello there")
        assert result.target == RouteTarget.LLM

    def test_preempt_routing(self):
        result = route("stop")
        assert result.target == RouteTarget.PREEMPT

    def test_trap_idiom_not_preempt(self):
        result = route("stop by the store")
        assert result.target != RouteTarget.PREEMPT
        assert result.target != RouteTarget.UTILITY
