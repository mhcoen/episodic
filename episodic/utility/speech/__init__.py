"""
Speech Response Generation for Utility Commands.

Generates varied, natural speech output for utility commands.
The same text is used for both display (with emoji) and TTS.
"""

from .generator import SpeechGenerator
from .formatters import (
    format_time_speech,
    format_duration_speech,
    format_temp_speech,
    format_ordinal,
    format_for_speech,
)

__all__ = [
    "SpeechGenerator",
    "format_time_speech",
    "format_duration_speech",
    "format_temp_speech",
    "format_ordinal",
    "format_for_speech",
]
