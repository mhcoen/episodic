"""Extraction pipeline contribution for the gsuite plugin.

Bundles gate keywords, phrases, intents, and contacts for
the extraction pipeline.
"""

from typing import Dict, List

from episodic.mcp.extraction.types import (
    IntentDefinition,
    PluginExtractionContribution,
)
from episodic.mcp.extraction.gate import (
    CALENDAR_KEYWORDS,
    CALENDAR_PHRASES,
    EMAIL_KEYWORDS,
    EMAIL_PHRASES,
)
from episodic.mcp.extraction.prompt import GSUITE_INTENTS


def _get_contacts() -> Dict[str, str]:
    """Load contacts from config, if available."""
    try:
        from episodic.config import config
        return config.get("contacts", {})
    except Exception:
        return {}


def build_extraction_contribution() -> PluginExtractionContribution:
    """Build the extraction contribution for gsuite."""
    gate_keywords: List[str] = sorted(
        set(CALENDAR_KEYWORDS) | set(EMAIL_KEYWORDS)
    )
    gate_phrases: List[List[str]] = list(CALENDAR_PHRASES) + list(EMAIL_PHRASES)
    intents: List[IntentDefinition] = list(GSUITE_INTENTS.values())

    return PluginExtractionContribution(
        gate_keywords=gate_keywords,
        gate_phrases=gate_phrases,
        intents=intents,
        contacts=_get_contacts(),
    )
