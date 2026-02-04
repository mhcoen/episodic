"""
Comprehensive Memory Query Detector Tests.

Tests for episodic/utility/voice/memory_detector.py including:
- Each PAST_TOKEN individually
- Each PAST_SEQUENCE
- Each MEMORY_PATTERN
- Skippable token combinations
- Negative cases (should NOT bypass)
"""

import pytest
from episodic.utility.voice.memory_detector import MemoryQueryDetector


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def detector():
    return MemoryQueryDetector()


# =============================================================================
# PAST_TOKEN Tests
# =============================================================================

class TestMemoryDetectorPastTokens:
    """Test each individual past token triggers bypass."""

    @pytest.mark.parametrize("past_token", [
        "did",
        "was",
        "were",
        "had",
        "started",
        "ended",
        "finished",
        "earlier",
        "yesterday",
        "previously",
        "already",
    ])
    def test_past_token_triggers_bypass(self, detector, past_token):
        tokens = ["what", "time", past_token, "the", "meeting"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass, f"Expected bypass for past_token: {past_token}"
        assert past_token in reason

    @pytest.mark.parametrize("sentence", [
        "what time did the meeting start",
        "was the weather good",
        "were we discussing python",
        "had I set an alarm",
        "the timer started",
        "when the meeting ended",
        "we finished the discussion",
        "we talked earlier",
        "what did we discuss yesterday",
        "I previously mentioned",
        "I already asked about that",
    ])
    def test_past_token_in_context(self, detector, sentence):
        tokens = sentence.lower().split()
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass, f"Expected bypass for: {sentence}"


# =============================================================================
# PAST_SEQUENCE Tests
# =============================================================================

class TestMemoryDetectorPastSequences:
    """Test each past sequence triggers bypass."""

    @pytest.mark.parametrize("sequence,sentence", [
        (["last", "time"], "the last time we talked"),
        (["last", "week"], "what happened last week"),
        (["last", "month"], "last month we discussed"),
        (["last", "year"], "last year I asked"),
        (["used", "to"], "we used to talk about"),
        (["have", "been"], "have been working on"),
        (["has", "been"], "it has been discussed"),
    ])
    def test_past_sequence_triggers_bypass(self, detector, sequence, sentence):
        tokens = sentence.lower().split()
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass, f"Expected bypass for sequence {sequence} in: {sentence}"
        # Check reason contains the sequence indicator
        assert "sequence" in reason.lower() or any(s in reason for s in sequence)


# =============================================================================
# MEMORY_PATTERN Tests
# =============================================================================

class TestMemoryDetectorMemoryPatterns:
    """Test each memory pattern triggers bypass."""

    @pytest.mark.parametrize("pattern,sentence", [
        (["when", "did"], "when did we meet"),
        (["what", "time", "did"], "what time did the meeting start"),
        (["where", "did"], "where did I put that"),
        (["who", "did"], "who did we talk about"),
        (["how", "did"], "how did that go"),
        (["what", "did", "we"], "what did we discuss"),
        (["what", "did", "i"], "what did I say"),
        (["do", "you", "remember"], "do you remember the topic"),
        (["did", "we", "discuss"], "did we discuss python"),
        (["did", "i", "mention"], "did I mention the deadline"),
    ])
    def test_memory_pattern_triggers_bypass(self, detector, pattern, sentence):
        tokens = sentence.lower().split()
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass, f"Expected bypass for pattern {pattern} in: {sentence}"


# =============================================================================
# SKIPPABLE_TOKEN Tests
# =============================================================================

class TestMemoryDetectorSkippableTokens:
    """Test skippable tokens are properly ignored."""

    @pytest.mark.parametrize("prefix", [
        "um",
        "uh",
        "er",
        "ah",
        "like",
        "hey",
        "hi",
        "ok",
        "okay",
        "please",
        "can",
        "you",
        "could",
        "would",
        "will",
        "just",
    ])
    def test_skippable_prefix_doesnt_block_detection(self, detector, prefix):
        """Skippable tokens at start should not block memory pattern detection."""
        tokens = [prefix, "what", "did", "we", "discuss"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass, f"Expected bypass with skippable prefix: {prefix}"

    def test_multiple_skippable_tokens(self, detector):
        """Multiple skippable tokens should all be skipped."""
        tokens = ["um", "uh", "like", "what", "did", "we", "discuss"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass

    def test_skippable_followed_by_present_tense(self, detector):
        """Skippable tokens followed by present-tense should not bypass."""
        tokens = ["um", "what", "time", "is", "it"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert not bypass


# =============================================================================
# Negative Cases (Should NOT Bypass)
# =============================================================================

class TestMemoryDetectorNegativeCases:
    """Test cases that should NOT trigger bypass."""

    @pytest.mark.parametrize("sentence", [
        "what time is it",
        "set a timer for 5 minutes",
        "what is the weather",
        "play some music",
        "tell me the news",
        "alarm for 7am",
        "remind me to call",
        "note buy milk",
        "stop",
        "repeat",
        "cancel the timer",
    ])
    def test_present_tense_not_bypassed(self, detector, sentence):
        tokens = sentence.lower().split()
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert not bypass, f"Expected NO bypass for: {sentence}"

    @pytest.mark.parametrize("sentence", [
        "time",
        "weather",
        "news",
        "timer",
        "alarm",
        "timers",
        "alarms",
    ])
    def test_single_word_not_bypassed(self, detector, sentence):
        tokens = sentence.lower().split()
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert not bypass, f"Expected NO bypass for single word: {sentence}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestMemoryDetectorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tokens(self, detector):
        bypass, reason = detector.should_bypass_utilities([])
        assert not bypass

    def test_only_skippable_tokens(self, detector):
        tokens = ["um", "uh", "like"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert not bypass

    def test_case_insensitivity(self, detector):
        """Detection should be case-insensitive."""
        tokens = ["WHAT", "DID", "WE", "DISCUSS"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass

    def test_sequence_not_at_start(self, detector):
        """Sequences anywhere in utterance should be detected."""
        tokens = ["tell", "me", "about", "last", "week"]
        bypass, reason = detector.should_bypass_utilities(tokens)
        assert bypass

    def test_pattern_must_be_at_content_start(self, detector):
        """Memory patterns must be at start of content tokens."""
        # "when did" at start
        bypass1, _ = detector.should_bypass_utilities(["when", "did", "we", "meet"])
        assert bypass1

        # "when did" not at start (after non-skippable)
        bypass2, _ = detector.should_bypass_utilities(["tell", "me", "when", "did", "we", "meet"])
        # This should still bypass due to "did" being a PAST_TOKEN
        assert bypass2
