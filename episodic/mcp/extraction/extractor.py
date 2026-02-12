"""Extraction LLM caller and dispatchability checker.

extract_intent() calls the LLM to classify an utterance and extract
structured arguments. check_dispatchability() validates the result
against registered intents using deterministic hard gates.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set

from episodic.mcp.extraction.prompt import (
    build_extraction_prompt,
    get_intents_for_domains,
)
from episodic.mcp.extraction.types import (
    ArgDefinition,
    DispatchabilityVerdict,
    ExtractionResult,
    IntentDefinition,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _get_extraction_model() -> str:
    """Resolve the model string for extraction calls."""
    from episodic.config import config

    model = config.get("extraction_model")
    if model:
        return model
    model = config.get("intent_model")
    if model:
        return model
    return _DEFAULT_MODEL


def _call_extraction_llm(
    utterance: str,
    system_prompt: str,
    model: str,
) -> str:
    """Make the synchronous LLM call for extraction."""
    from episodic.llm import query_llm

    extra_kwargs: Dict[str, Any] = {}
    if "ollama/" in model.lower():
        extra_kwargs["format"] = "json"
    else:
        extra_kwargs["response_format"] = {"type": "json_object"}

    response_text, _ = query_llm(
        prompt=utterance,
        model=model,
        system_message=system_prompt,
        temperature=0,
        max_tokens=300,
        **extra_kwargs,
    )
    return response_text


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
        stripped = stripped[first_newline + 1:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


def _parse_extraction_response(raw_json: str) -> ExtractionResult:
    """Parse the LLM JSON response into an ExtractionResult."""
    try:
        cleaned = _strip_markdown_fences(raw_json) if raw_json else raw_json
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Extraction LLM returned invalid JSON: %s", raw_json[:200])
        return ExtractionResult(
            intent=None,
            args={},
            confidence=0.0,
            followup_suggestion=None,
            raw_json=raw_json if isinstance(raw_json, str) else "",
        )

    intent = data.get("intent")
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    confidence = float(data.get("confidence", 0.0))
    followup = data.get("followup_suggestion")

    return ExtractionResult(
        intent=intent,
        args=args,
        confidence=confidence,
        followup_suggestion=followup,
        raw_json=raw_json,
    )


async def extract_intent(
    utterance: str,
    matched_domains: Set[str],
    contacts: Dict[str, str],
    recent_context: Optional[str] = None,
) -> ExtractionResult:
    """Call extraction LLM, parse JSON response, return ExtractionResult.

    On JSON parse failure, returns ExtractionResult with intent=None.
    Uses the 'extraction' model context if configured, otherwise falls
    back to 'intent' context, otherwise falls back to a hardcoded
    Haiku model string.
    """
    system_prompt = build_extraction_prompt(
        domains=matched_domains,
        contacts=contacts,
        recent_context=recent_context,
    )
    model = _get_extraction_model()

    raw_json = await asyncio.to_thread(
        _call_extraction_llm, utterance, system_prompt, model
    )
    return _parse_extraction_response(raw_json)


def _validate_arg_type(value: Any, expected_type: str) -> bool:
    """Check if a value conforms to the declared arg type."""
    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    elif expected_type == "list":
        return isinstance(value, list)
    return True  # Unknown types pass


def check_dispatchability(
    result: ExtractionResult,
    registered_intents: Dict[str, IntentDefinition],
) -> DispatchabilityVerdict:
    """Core-computed dispatchability. No model confidence used.

    Hard gates (all must pass):
    1. intent is not None
    2. intent is registered OR is "router.unknown_command"
    3. Each arg value matches declared type (string, boolean, list)
    4. All required args are present and non-empty

    Returns DispatchabilityVerdict with full status information.
    """
    # Gate 1: null intent
    if result.intent is None:
        return DispatchabilityVerdict(
            dispatchable=False,
            intent=None,
            args={},
            action_class=None,
            missing_required_args=[],
            is_unknown_command=False,
            unknown_command_hint=None,
            followup_suggestion=result.followup_suggestion,
            error=None,
        )

    # Gate 2: router.unknown_command
    if result.intent == "router.unknown_command":
        hint = result.args.get("hint") if isinstance(result.args, dict) else None
        return DispatchabilityVerdict(
            dispatchable=False,
            intent="router.unknown_command",
            args=result.args,
            action_class=None,
            missing_required_args=[],
            is_unknown_command=True,
            unknown_command_hint=hint,
            followup_suggestion=result.followup_suggestion,
            error=None,
        )

    # Gate 3: intent registration
    if result.intent not in registered_intents:
        return DispatchabilityVerdict(
            dispatchable=False,
            intent=result.intent,
            args=result.args,
            action_class=None,
            missing_required_args=[],
            is_unknown_command=False,
            unknown_command_hint=None,
            followup_suggestion=result.followup_suggestion,
            error=f"Unregistered intent: {result.intent}",
        )

    intent_def = registered_intents[result.intent]

    # Gate 4: arg type validation — strip invalid args
    validated_args: Dict[str, Any] = {}
    for arg_name, value in result.args.items():
        if arg_name in intent_def.args:
            arg_def = intent_def.args[arg_name]
            if _validate_arg_type(value, arg_def.type):
                validated_args[arg_name] = value
            # Invalid type: silently strip
        else:
            # Extra args not in schema: pass through
            validated_args[arg_name] = value

    # Gate 5: required args check
    missing: List[str] = []
    for arg_name, arg_def in intent_def.args.items():
        if not arg_def.required:
            continue
        value = validated_args.get(arg_name)
        if value is None or value == "":
            missing.append(arg_name)

    dispatchable = len(missing) == 0

    return DispatchabilityVerdict(
        dispatchable=dispatchable,
        intent=result.intent,
        args=validated_args,
        action_class=intent_def.action_class,
        missing_required_args=missing,
        is_unknown_command=False,
        unknown_command_hint=None,
        followup_suggestion=result.followup_suggestion,
        error=None,
    )
