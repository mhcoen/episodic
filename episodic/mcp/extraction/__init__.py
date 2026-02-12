"""MCP Intent Extraction Pipeline.

Standalone module for determining whether a voice/text utterance is a
command directed at an MCP service or ordinary chat. Produces an
ExtractionResult and DispatchabilityVerdict at its boundary.
"""

from episodic.mcp.extraction.extractor import check_dispatchability, extract_intent
from episodic.mcp.extraction.gate import matched_domains
from episodic.mcp.extraction.prompt import (
    GSUITE_INTENTS,
    UNKNOWN_COMMAND_INTENT,
    build_extraction_prompt,
    get_intents_for_domains,
)
from episodic.mcp.extraction.types import (
    ArgDefinition,
    DispatchabilityVerdict,
    ExtractionResult,
    IntentDefinition,
)

__all__ = [
    "matched_domains",
    "extract_intent",
    "check_dispatchability",
    "build_extraction_prompt",
    "get_intents_for_domains",
    "GSUITE_INTENTS",
    "UNKNOWN_COMMAND_INTENT",
    "ArgDefinition",
    "IntentDefinition",
    "ExtractionResult",
    "DispatchabilityVerdict",
]
