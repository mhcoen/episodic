"""
Preempt Router for Voice Grammar.

First-pass routing BEFORE grammar parsing. Handles ONLY hard overrides.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set

from ..types import UtilityQuery


@dataclass
class RuntimeState:
    """Runtime state for stateful command resolution."""
    media_playing: bool = False
    media_source: Optional[str] = None
    tts_speaking: bool = False
    last_pending_mutation: Optional[str] = None
    active_timers: List[str] = field(default_factory=list)
    dnd_active: bool = False
    last_command_category: Optional[str] = None
    timezone: str = "America/Chicago"


class PreemptRouter:
    """
    Preempt router with explicit patterns.

    STOP: Exact matches + "stop ..." prefix (with blacklist)
    REPEAT: Exact matches only

    "cancel" is NOT here — goes to grammar for "cancel my 7am alarm"
    "what" is NOT here — too broad, would swallow "what time is it"
    """

    STOP_EXACT: Set[str] = {
        "stop", "silence", "shut up", "enough", "quiet",
        "stop it", "stop that", "stop playing", "stop talking",
    }

    # Prefixes that do NOT trigger stop
    STOP_BLACKLIST_PREFIXES: List[str] = [
        "stop by",      # "stop by the store"
        "stop for",     # "stop for coffee"
        "stop at",      # "stop at the corner"
        "stop and",     # "stop and think"
    ]

    REPEAT_EXACT: Set[str] = {
        "repeat", "say that again", "again", "pardon", "huh",
        "come again", "what did you say", "i did not hear that",
        "i did not catch that",
    }

    def preempt_check(self, text: str, state: RuntimeState) -> Optional[UtilityQuery]:
        """
        Check for global preempts before parsing.
        Returns UtilityQuery if preempted, None to continue to grammar.
        """
        text_lower = text.lower().strip()

        # Check exact stop triggers
        if text_lower in self.STOP_EXACT:
            return self._resolve_stop(state, text_lower, text)

        # Check "stop ..." prefix (excluding blacklist)
        if text_lower.startswith("stop "):
            for prefix in self.STOP_BLACKLIST_PREFIXES:
                if text_lower.startswith(prefix):
                    return None
            if len(text_lower.split()) <= 3:
                return self._resolve_stop(state, text_lower, text)

        # Check repeat triggers
        if text_lower in self.REPEAT_EXACT:
            return UtilityQuery(
                category="system",
                command="repeat",
                args={},
                confidence=0.95,
                source="preempt",
                raw_input=text,
            )

        return None

    def _resolve_stop(self, state: RuntimeState, text_lower: str, raw_input: str) -> UtilityQuery:
        """Stateful stop resolution."""
        if state.tts_speaking:
            return UtilityQuery(
                category="system",
                command="stop_tts",
                args={},
                confidence=0.99,
                source="preempt",
                raw_input=raw_input,
            )
        if state.media_playing:
            return UtilityQuery(
                category="media",
                command="media_stop",
                args={},
                confidence=0.99,
                source="preempt",
                raw_input=raw_input,
            )
        if state.last_pending_mutation:
            return UtilityQuery(
                category="system",
                command="cancel",
                args={"target": state.last_pending_mutation},
                confidence=0.95,
                source="preempt",
                raw_input=raw_input,
            )
        return UtilityQuery(
            category="system",
            command="noop",
            args={"reason": "nothing_active"},
            confidence=0.90,
            source="preempt",
            raw_input=raw_input,
        )
