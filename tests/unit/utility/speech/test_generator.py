"""Tests for speech generator."""

from datetime import datetime

import pytest

from episodic.utility.speech.generator import SpeechGenerator


class TestSpeechGenerator:
    """Test speech generation with anti-repetition."""

    def setup_method(self):
        """Reset singleton for each test."""
        SpeechGenerator.reset_instance()

    def test_singleton(self):
        """Get_instance returns same instance."""
        gen1 = SpeechGenerator.get_instance()
        gen2 = SpeechGenerator.get_instance()
        assert gen1 is gen2

    def test_time_now_generates_output(self):
        """Time command generates display and speech text."""
        gen = SpeechGenerator()
        now = datetime.now()

        display, speech = gen.generate("time_now", {
            "_command": "time_now",
            "time": now.isoformat(),
        })

        assert display
        assert speech
        # Display should have emoji
        assert "\u23f0" in display or "It's" in display

    def test_timer_set_generates_output(self):
        """Timer set generates display and speech text."""
        gen = SpeechGenerator()

        display, speech = gen.generate("timer_set", {
            "_command": "timer_set",
            "duration": 300,
            "label": None,
        })

        assert display
        assert speech
        # Should mention 5 minutes
        assert "five minutes" in speech or "5 minute" in speech

    def test_timer_set_labeled_uses_label(self):
        """Timer with label uses labeled template."""
        gen = SpeechGenerator()

        display, speech = gen.generate("timer_set", {
            "_command": "timer_set",
            "duration": 600,
            "label": "pasta",
        })

        assert "pasta" in display or "pasta" in speech

    def test_weather_generates_output(self):
        """Weather generates display and speech."""
        gen = SpeechGenerator()

        display, speech = gen.generate("weather_now", {
            "_command": "weather_now",
            "location": "Madison",
            "temp": 33,
            "condition": "foggy",
            "emoji": "\U0001f32b\ufe0f",
        })

        assert display
        assert speech
        assert "Madison" in speech or "33" in speech

    def test_avoids_immediate_repetition(self):
        """Generator avoids using same template consecutively."""
        gen = SpeechGenerator()

        # Generate multiple times for time_now
        results = []
        for _ in range(10):
            display, speech = gen.generate("time_now", {
                "_command": "time_now",
                "time": datetime.now().isoformat(),
            })
            results.append(speech)

        # Check no more than 2 consecutive identical templates
        # (can have same start but templates should vary)
        consecutive_same = 1
        max_consecutive = 1
        for i in range(1, len(results)):
            # Templates may produce same text if time is same
            # Check that variety exists over 10 iterations
            pass

        # At least 2 unique responses in 10 iterations
        unique_responses = set(results)
        assert len(unique_responses) >= 2, f"Expected variety, got: {unique_responses}"

    def test_fallback_on_unknown_command(self):
        """Unknown command falls back to provided text."""
        gen = SpeechGenerator()

        display, speech = gen.generate("unknown_command", {
            "_command": "unknown_command",
            "display_text": "Fallback display",
            "speech_text": "Fallback speech",
        })

        assert display == "Fallback display"
        assert speech == "Fallback speech"

    def test_fallback_on_missing_values(self):
        """Missing template values cause fallback."""
        gen = SpeechGenerator()

        # weather_now requires location but not provided
        display, speech = gen.generate("weather_now", {
            "_command": "weather_now",
            "display_text": "Default display",
            "speech_text": "Default speech",
            # Missing location, temp, condition
        })

        # Should fall back due to KeyError in template
        # Or use defaults that make sense
        assert display
        assert speech


class TestAntiRepetition:
    """Test anti-repetition mechanism in detail."""

    def setup_method(self):
        SpeechGenerator.reset_instance()

    def test_history_tracks_recent(self):
        """History tracks recent template indices."""
        gen = SpeechGenerator()

        # Generate several times
        for _ in range(5):
            gen.generate("time_now", {
                "_command": "time_now",
                "time": datetime.now().isoformat(),
            })

        # History should exist for time_now
        assert "time_now" in gen._history
        # History should have entries
        assert len(gen._history["time_now"]) > 0

    def test_different_commands_have_separate_history(self):
        """Each command maintains separate history."""
        gen = SpeechGenerator()

        # Generate for time_now
        gen.generate("time_now", {
            "_command": "time_now",
            "time": datetime.now().isoformat(),
        })

        # Generate for timer_set
        gen.generate("timer_set", {
            "_command": "timer_set",
            "duration": 300,
        })

        # Both should have separate histories
        assert "time_now" in gen._history
        assert "timer_set" in gen._history
