"""
Confidence Calculator for Voice Grammar.

Deterministic confidence scoring with explicit mutation invariants.

INVARIANT: Mutations never execute unless:
  - is_exact_template, OR
  - args_complete AND has_domain_keyword AND NOT fuzzy_match_used
"""

from dataclasses import dataclass
from typing import Set, Dict


@dataclass
class ParseFeatures:
    """Explicit features for deterministic scoring."""
    has_query_marker: bool = False
    has_action_marker: bool = False
    has_domain_keyword: bool = False
    args_complete: bool = False
    args_partial: bool = False
    is_exact_template: bool = False
    has_past_tense: bool = False
    has_conjunction: bool = False
    has_opinion_request: bool = False
    has_explanation_request: bool = False
    word_count: int = 0
    fuzzy_match_used: bool = False


class ConfidenceCalculator:
    """
    Deterministic confidence with explicit invariants.

    INVARIANT: Mutations never execute unless:
      - is_exact_template, OR
      - args_complete AND has_domain_keyword AND NOT fuzzy_match_used
    """

    MUTATE_COMMANDS: Set[str] = {
        "timer_set", "timer_cancel", "timer_pause", "timer_resume",
        "alarm_set", "alarm_cancel", "alarm_snooze",
        "remind_set", "remind_cancel",
        "note_add", "note_delete",
        "list_add", "list_remove",
        "dnd_on", "dnd_off",
        "media_play", "media_pause", "media_stop",
    }

    READ_COMMANDS: Set[str] = {
        "time_now", "date_today",
        "timer_list", "alarm_list", "remind_list", "note_list",
        "weather_now", "weather_forecast",
        "news_headlines",
        "media_status", "status",
    }

    SYSTEM_COMMANDS: Set[str] = {
        "cancel", "undo", "repeat", "stop", "noop",
        "stop_tts",
    }

    THRESHOLDS: Dict[str, float] = {
        "mutate": 0.80,
        "read": 0.55,
        "system": 0.70,
    }

    def classify_command(self, command: str) -> str:
        """Classify a command into mutate, read, or system."""
        if command in self.MUTATE_COMMANDS:
            return "mutate"
        elif command in self.READ_COMMANDS:
            return "read"
        elif command in self.SYSTEM_COMMANDS:
            return "system"
        return "unknown"

    def get_threshold(self, command: str) -> float:
        """Get the confidence threshold for a command."""
        command_class = self.classify_command(command)
        return self.THRESHOLDS.get(command_class, 0.70)

    def calculate(self, features: ParseFeatures, command: str) -> float:
        """Calculate confidence score."""
        command_class = self.classify_command(command)

        # Disqualifiers
        if features.has_past_tense:
            return 0.0
        if features.has_opinion_request:
            return 0.0
        if features.has_explanation_request:
            return 0.0

        # Base score
        if features.is_exact_template:
            score = 0.95
        else:
            score = 0.0
            if features.has_domain_keyword:
                score += 0.35
            if features.has_query_marker or features.has_action_marker:
                score += 0.25
            if features.args_complete:
                score += 0.25
            elif features.args_partial:
                score += 0.10

        # Penalties
        if features.has_conjunction:
            score -= 0.15
        if features.fuzzy_match_used:
            score -= 0.10

        score = max(0.0, min(0.99, score))

        # MUTATION GATE
        if command_class == "mutate":
            if features.fuzzy_match_used:
                score = min(score, self.THRESHOLDS["mutate"] - 0.01)
            elif not features.is_exact_template:
                if not (features.args_complete and features.has_domain_keyword):
                    score = min(score, self.THRESHOLDS["mutate"] - 0.01)

        return score

    def decide_action(self, confidence: float, command: str) -> str:
        """
        Returns: "execute" | "confirm" | "reject"

        INVARIANT: mutate never "execute" unless confidence >= threshold.
        """
        command_class = self.classify_command(command)
        threshold = self.THRESHOLDS.get(command_class, 0.70)

        if confidence >= threshold:
            return "execute"
        elif confidence >= threshold - 0.15:
            return "confirm"
        else:
            return "reject"

    def check_mutation_acceptable(self, features: ParseFeatures) -> bool:
        """
        Check if mutation should be allowed based on features.

        Returns True if:
        - is_exact_template, OR
        - args_complete AND has_domain_keyword AND NOT fuzzy_match_used
        """
        if features.is_exact_template:
            return True
        return (
            features.args_complete and
            features.has_domain_keyword and
            not features.fuzzy_match_used
        )
