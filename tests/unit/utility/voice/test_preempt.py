"""
Comprehensive Preempt Router Tests.

Tests for episodic/utility/voice/preempt.py including:
- Each STOP_EXACT trigger
- Each STOP_BLACKLIST_PREFIX
- Each REPEAT_EXACT trigger
- RuntimeState interactions
- Stop resolution priority
"""

import pytest
from episodic.utility.voice.preempt import PreemptRouter, RuntimeState


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def router():
    return PreemptRouter()


@pytest.fixture
def default_state():
    return RuntimeState()


@pytest.fixture
def tts_speaking_state():
    return RuntimeState(tts_speaking=True)


@pytest.fixture
def media_playing_state():
    return RuntimeState(media_playing=True)


@pytest.fixture
def pending_mutation_state():
    return RuntimeState(last_pending_mutation="timer_set")


# =============================================================================
# STOP_EXACT Tests
# =============================================================================

class TestPreemptStopExact:
    """Test each STOP_EXACT trigger."""

    @pytest.mark.parametrize("trigger", [
        "stop",
        "silence",
        "shut up",
        "enough",
        "quiet",
        "stop it",
        "stop that",
        "stop playing",
        "stop talking",
    ])
    def test_stop_exact_trigger(self, router, default_state, trigger):
        result = router.preempt_check(trigger, default_state)
        assert result is not None, f"Expected preempt for: {trigger}"
        # All stop triggers should route to some stop-related command
        assert result.source == "preempt"

    @pytest.mark.parametrize("trigger", [
        "STOP",
        "Stop",
        "SILENCE",
        "Shut Up",
    ])
    def test_stop_case_insensitive(self, router, default_state, trigger):
        """Stop triggers should be case-insensitive."""
        result = router.preempt_check(trigger, default_state)
        assert result is not None


# =============================================================================
# STOP_BLACKLIST_PREFIX Tests
# =============================================================================

class TestPreemptStopBlacklist:
    """Test each STOP_BLACKLIST_PREFIX is NOT triggered."""

    @pytest.mark.parametrize("phrase", [
        "stop by the store",
        "stop by later",
        "stop for coffee",
        "stop for a moment",
        "stop at the corner",
        "stop at 5pm",
        "stop and think",
        "stop and smell the roses",
    ])
    def test_blacklist_not_triggered(self, router, default_state, phrase):
        result = router.preempt_check(phrase, default_state)
        assert result is None, f"Expected NO preempt for blacklisted phrase: {phrase}"


class TestPreemptStopPrefixEdgeCases:
    """Test edge cases for stop prefix handling."""

    def test_stop_with_short_suffix(self, router, default_state):
        """'stop X' with 1-2 words should trigger (if not blacklisted)."""
        result = router.preempt_check("stop now", default_state)
        assert result is not None

    def test_stop_with_long_suffix(self, router, default_state):
        """'stop X Y Z W' (4+ words) should not trigger."""
        result = router.preempt_check("stop what you are doing now", default_state)
        assert result is None


# =============================================================================
# REPEAT_EXACT Tests
# =============================================================================

class TestPreemptRepeatExact:
    """Test each REPEAT_EXACT trigger."""

    @pytest.mark.parametrize("trigger", [
        "repeat",
        "say that again",
        "again",
        "pardon",
        "huh",
        "come again",
        "what did you say",
        "i did not hear that",
        "i did not catch that",
    ])
    def test_repeat_exact_trigger(self, router, default_state, trigger):
        result = router.preempt_check(trigger, default_state)
        assert result is not None, f"Expected preempt for: {trigger}"
        assert result.command == "repeat"
        assert result.confidence == 0.95

    @pytest.mark.parametrize("trigger", [
        "REPEAT",
        "Repeat",
        "Say That Again",
        "PARDON",
    ])
    def test_repeat_case_insensitive(self, router, default_state, trigger):
        result = router.preempt_check(trigger, default_state)
        assert result is not None
        assert result.command == "repeat"


# =============================================================================
# RuntimeState Interaction Tests
# =============================================================================

class TestPreemptStateInteraction:
    """Test RuntimeState affects stop resolution."""

    def test_tts_speaking_gets_stop_tts(self, router, tts_speaking_state):
        """When TTS is speaking, 'stop' should resolve to stop_tts."""
        result = router.preempt_check("stop", tts_speaking_state)
        assert result is not None
        assert result.command == "stop_tts"
        assert result.confidence == 0.99

    def test_media_playing_gets_media_stop(self, router, media_playing_state):
        """When media is playing, 'stop' should resolve to media_stop."""
        result = router.preempt_check("stop", media_playing_state)
        assert result is not None
        assert result.command == "media_stop"
        assert result.confidence == 0.99

    def test_pending_mutation_gets_cancel(self, router, pending_mutation_state):
        """When there's a pending mutation, 'stop' should cancel it."""
        result = router.preempt_check("stop", pending_mutation_state)
        assert result is not None
        assert result.command == "cancel"
        assert result.args.get("target") == "timer_set"

    def test_nothing_active_gets_noop(self, router, default_state):
        """When nothing is active, 'stop' should be noop."""
        result = router.preempt_check("stop", default_state)
        assert result is not None
        assert result.command == "noop"
        assert result.args.get("reason") == "nothing_active"


class TestPreemptStatePriority:
    """Test priority when multiple states are active."""

    def test_tts_priority_over_media(self, router):
        """TTS speaking should take priority over media playing."""
        state = RuntimeState(tts_speaking=True, media_playing=True)
        result = router.preempt_check("stop", state)
        assert result.command == "stop_tts"

    def test_tts_priority_over_pending(self, router):
        """TTS speaking should take priority over pending mutation."""
        state = RuntimeState(tts_speaking=True, last_pending_mutation="timer_set")
        result = router.preempt_check("stop", state)
        assert result.command == "stop_tts"

    def test_media_priority_over_pending(self, router):
        """Media playing should take priority over pending mutation."""
        state = RuntimeState(media_playing=True, last_pending_mutation="timer_set")
        result = router.preempt_check("stop", state)
        assert result.command == "media_stop"


# =============================================================================
# Non-Preempt Cases
# =============================================================================

class TestPreemptNonTriggers:
    """Test phrases that should NOT trigger preempt."""

    @pytest.mark.parametrize("phrase", [
        "what time is it",
        "set a timer",
        "play some music",
        "cancel the alarm",
        "what",
        "when",
        "tell me about stop signs",
        "repeat after me please",  # Not exact match
        "stopping now",            # Not exact match
    ])
    def test_non_triggers(self, router, default_state, phrase):
        result = router.preempt_check(phrase, default_state)
        assert result is None, f"Expected NO preempt for: {phrase}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestPreemptEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, router, default_state):
        result = router.preempt_check("", default_state)
        assert result is None

    def test_whitespace_only(self, router, default_state):
        result = router.preempt_check("   ", default_state)
        assert result is None

    def test_stop_with_leading_whitespace(self, router, default_state):
        """Leading/trailing whitespace should be stripped."""
        result = router.preempt_check("  stop  ", default_state)
        assert result is not None

    def test_repeat_preserves_raw_input(self, router, default_state):
        """raw_input should preserve original text."""
        result = router.preempt_check("repeat", default_state)
        assert result.raw_input == "repeat"


# =============================================================================
# Confidence Tests
# =============================================================================

class TestPreemptConfidence:
    """Test confidence values for preempt results."""

    def test_stop_tts_confidence(self, router, tts_speaking_state):
        result = router.preempt_check("stop", tts_speaking_state)
        assert result.confidence == 0.99

    def test_media_stop_confidence(self, router, media_playing_state):
        result = router.preempt_check("stop", media_playing_state)
        assert result.confidence == 0.99

    def test_cancel_pending_confidence(self, router, pending_mutation_state):
        result = router.preempt_check("stop", pending_mutation_state)
        assert result.confidence == 0.95

    def test_noop_confidence(self, router, default_state):
        result = router.preempt_check("stop", default_state)
        assert result.confidence == 0.90

    def test_repeat_confidence(self, router, default_state):
        result = router.preempt_check("repeat", default_state)
        assert result.confidence == 0.95
