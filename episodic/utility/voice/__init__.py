"""
Voice Grammar Parser for Utility Commands.

Converts spoken/typed utterances into UtilityQuery AST nodes
with deterministic confidence scoring.
"""

from .pipeline import parse_utterance, VoiceParseResult

__all__ = ["parse_utterance", "VoiceParseResult"]
