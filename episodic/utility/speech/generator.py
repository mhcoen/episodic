"""
Speech Generator for Utility Commands.

Generates varied, natural speech output while avoiding repetition.
Uses a singleton pattern to maintain anti-repetition history.
"""

import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .templates import TEMPLATES, get_templates, get_emoji, get_weather_extension
from .formatters import format_duration_speech, format_time_speech


class SpeechGenerator:
    """
    Generates varied speech responses for utility commands.

    Tracks recently used templates to avoid repetition.
    """

    _instance: Optional["SpeechGenerator"] = None

    def __init__(self):
        # Track last N used template indices per command
        # Maps command -> deque of recent template indices
        self._history: Dict[str, deque] = {}
        self._history_size = 3

    @classmethod
    def get_instance(cls) -> "SpeechGenerator":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def generate(
        self,
        command: str,
        values: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Generate display and speech text for a command.

        Args:
            command: Command name (e.g., "timer_set", "weather_now")
            values: Data from UtilityResult including _command, label, etc.

        Returns:
            Tuple of (display_text, speech_text)
            display_text includes emoji prefix
            speech_text is for TTS
        """
        # Check for labeled variant
        has_label = bool(values.get("label"))

        # Get available templates
        templates = get_templates(command, has_label)

        if not templates:
            # No templates - fall back to existing speech_text or display_text
            fallback_display = values.get("display_text", "Done")
            fallback_speech = values.get("speech_text", fallback_display)
            return (fallback_display, fallback_speech)

        # Select template avoiding recent ones
        template_idx = self._select_template(command, len(templates))
        display_template, speech_template = templates[template_idx]

        # If speech_template is None, use display_template for both
        if speech_template is None:
            speech_template = display_template

        # Format values for templates
        formatted_values = self._format_values(values)

        # Apply templates
        try:
            display_text = display_template.format(**formatted_values)
            speech_text = speech_template.format(**formatted_values)
        except KeyError:
            # Missing placeholder - fall back
            fallback_display = values.get("display_text", "Done")
            fallback_speech = values.get("speech_text", fallback_display)
            return (fallback_display, fallback_speech)

        # Add weather extension if applicable
        if command == "weather_now":
            condition = values.get("condition", "")
            extension = get_weather_extension(condition)
            speech_text += extension

        # Get emoji
        result_emoji = values.get("emoji")
        emoji = get_emoji(command, result_emoji)

        # Combine emoji and display text
        if emoji:
            display_text = f"{emoji}  {display_text}"

        return (display_text, speech_text)

    def _select_template(self, command: str, num_templates: int) -> int:
        """
        Select a template index, avoiding recently used ones.

        Args:
            command: Command name for history tracking
            num_templates: Number of available templates

        Returns:
            Template index to use
        """
        if num_templates == 1:
            return 0

        # Get history for this command
        if command not in self._history:
            self._history[command] = deque(maxlen=self._history_size)

        history = self._history[command]

        # Find indices not in recent history
        available = [i for i in range(num_templates) if i not in history]

        if not available:
            # All templates used recently - pick least recent
            available = [i for i in range(num_templates)]

        # Random selection from available
        selected = random.choice(available)

        # Update history
        history.append(selected)

        return selected

    def _format_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw values for template substitution.

        Applies special formatting for durations, times, etc.
        """
        formatted = dict(values)

        # Format duration
        if "duration" in values or "duration_s" in values:
            duration_s = values.get("duration", values.get("duration_s", 0))
            if isinstance(duration_s, (int, float)):
                formatted["duration"] = format_duration_speech(int(duration_s))

        # Format time (for alarms)
        if "time" in values:
            time_val = values.get("time")
            if isinstance(time_val, str):
                # Parse ISO format
                try:
                    dt = datetime.fromisoformat(time_val)
                    formatted["time"] = dt.strftime("%I:%M %p").lstrip("0")
                except ValueError:
                    # Already formatted
                    pass
            elif isinstance(time_val, datetime):
                formatted["time"] = time_val.strftime("%I:%M %p").lstrip("0")

        # Format current time (for time_now)
        if command_is_time_now := values.get("_command") == "time_now":
            if "time" in values:
                try:
                    dt = datetime.fromisoformat(values["time"])
                    formatted["time"] = dt.strftime("%I:%M %p").lstrip("0")
                except (ValueError, TypeError):
                    pass

        # Capitalize condition for display
        if "condition" in formatted:
            condition = formatted["condition"]
            if isinstance(condition, str) and condition:
                formatted["condition"] = condition.capitalize()

        # Ensure common fields have defaults
        formatted.setdefault("location", "here")
        formatted.setdefault("condition", "")
        formatted.setdefault("temp", 0)
        formatted.setdefault("high", 0)
        formatted.setdefault("low", 0)
        formatted.setdefault("label", "")
        formatted.setdefault("text", "")
        formatted.setdefault("station", "")

        return formatted
