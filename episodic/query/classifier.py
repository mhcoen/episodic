"""
LLM Fallback Classifier for FreeText and Ambiguous Inputs.

When the CFG parser fails to match a structured pattern (producing FreeText),
or matches ambiguously (MQLCommand with no explicit markers), this classifier
uses an LLM to determine if the input is asking about conversation history
(memory query) vs a general question, and extracts intent in a single call.

Architecture:
1. CFG (lexer → parser) tries structured patterns
2. If CFG fails → FreeText, or produces ambiguous MQLCommand
3. Single LLM call classifies AND extracts intent
4. Result routes to retrieval (memory) or chat (general)
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Combined classification + extraction prompt
INTENT_PROMPT = """Analyze this input and return a JSON object:

Input: "{input}"

Return:
{{
  "is_memory_query": true/false,
  "target": "string or null",
  "mode": "browse" or "answer",
  "temporal_hint": "string or null",
  "speaker_hint": "user"/"assistant"/"both"/null
}}

Field definitions:
- is_memory_query: Is this asking about our past conversations?
- target: Topic being asked about (if memory query, else null)
- mode: "browse" for listing/exploring, "answer" for specific answer (if memory query)
- temporal_hint: Time reference like "yesterday", "last week" (if any)
- speaker_hint: Who said it - "user", "assistant", "both", or null

Examples:
"When did we discuss coffee?" → {{"is_memory_query": true, "target": "coffee", "mode": "browse", "temporal_hint": null, "speaker_hint": "both"}}
"What is machine learning?" → {{"is_memory_query": false, "target": null, "mode": null, "temporal_hint": null, "speaker_hint": null}}
"What did I say about the project yesterday?" → {{"is_memory_query": true, "target": "the project", "mode": "answer", "temporal_hint": "yesterday", "speaker_hint": "user"}}
"Anything about coffee in our past chats?" → {{"is_memory_query": true, "target": "coffee", "mode": "browse", "temporal_hint": null, "speaker_hint": "both"}}
"Explain quantum physics" → {{"is_memory_query": false, "target": null, "mode": null, "temporal_hint": null, "speaker_hint": null}}

Respond with only valid JSON."""


@dataclass
class ClassificationResult:
    """Result of combined classification and intent extraction."""
    is_memory_query: bool
    target: Optional[str]
    mode: Optional[str]  # "browse" or "answer"
    temporal_hint: Optional[str]
    speaker_hint: Optional[str]  # "user", "assistant", "both", or None
    confidence: str  # "high", "medium", "low"
    raw_response: str


def classify_and_extract_intent(user_input: str, model: Optional[str] = None) -> ClassificationResult:
    """
    Classify input and extract memory query intent in a single LLM call.

    Args:
        user_input: The user's input text
        model: Optional model to use (defaults to config classifier_model or gpt-4o-mini)

    Returns:
        ClassificationResult with is_memory_query flag and extracted intent fields
    """
    from episodic.config import config
    from episodic.llm import query_llm

    # Use intent_model if set, otherwise classifier_model (legacy), otherwise gpt-4o-mini
    if model is None:
        model = config.get("intent_model") or config.get("classifier_model", "gpt-4o-mini")

    prompt = INTENT_PROMPT.format(input=user_input)

    # Build extra kwargs for the LLM call
    extra_kwargs = {}

    # For Ollama models, use format="json" to enforce valid JSON output
    # LiteLLM passes this through to Ollama's native JSON mode
    if model and model.startswith("ollama/"):
        extra_kwargs["format"] = "json"

    try:
        response_text, _ = query_llm(
            prompt=prompt,
            model=model,
            system_message="You are an intent analyzer. Respond with only valid JSON.",
            temperature=0.0,  # Deterministic classification
            **extra_kwargs
        )

        # Parse JSON response
        response_clean = response_text.strip()
        # Handle markdown code blocks
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            response_clean = "\n".join(lines[1:-1])

        intent = json.loads(response_clean)

        is_memory = intent.get("is_memory_query", False)
        confidence = "high"  # JSON parsed successfully

        if config.get("debug"):
            logger.debug(f"[CLASSIFIER] Input: {user_input[:50]}...")
            logger.debug(f"[CLASSIFIER] Intent: {intent}")

        return ClassificationResult(
            is_memory_query=is_memory,
            target=intent.get("target"),
            mode=intent.get("mode"),
            temporal_hint=intent.get("temporal_hint"),
            speaker_hint=intent.get("speaker_hint"),
            confidence=confidence,
            raw_response=response_text
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classifier JSON: {e}")
        logger.warning(f"Raw response: {response_text}")
        # Fallback: try to detect MEMORY/GENERAL in raw text
        response_upper = response_text.upper()
        is_memory = "MEMORY" in response_upper and "GENERAL" not in response_upper
        return ClassificationResult(
            is_memory_query=is_memory,
            target=user_input if is_memory else None,
            mode="answer" if is_memory else None,
            temporal_hint=None,
            speaker_hint=None,
            confidence="low",
            raw_response=response_text
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        # On error, default to GENERAL (safer to let LLM handle it)
        return ClassificationResult(
            is_memory_query=False,
            target=None,
            mode=None,
            temporal_hint=None,
            speaker_hint=None,
            confidence="low",
            raw_response=f"ERROR: {e}"
        )


# Backwards compatibility aliases (deprecated)
def classify_freetext(user_input: str, model: Optional[str] = None) -> ClassificationResult:
    """Deprecated: Use classify_and_extract_intent() instead."""
    return classify_and_extract_intent(user_input, model)


def extract_memory_intent(user_input: str, model: Optional[str] = None) -> dict:
    """Deprecated: Use classify_and_extract_intent() instead."""
    result = classify_and_extract_intent(user_input, model)
    return {
        "target": result.target,
        "mode": result.mode,
        "temporal_hint": result.temporal_hint,
        "speaker_hint": result.speaker_hint
    }
