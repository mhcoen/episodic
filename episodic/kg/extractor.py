"""LLM call wrapper for KG extraction."""

import hashlib
import json
import time

from .prompt_template import (
    EXTRACTION_SYSTEM_PROMPT,
    RETRY_ADDENDUM,
    format_extraction_input,
    build_extraction_context,
)

VALIDATOR_VERSION = "kg_validator_v1"

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds: 2, 4, 8


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit / 429 error."""
    err_str = str(e).lower()
    return ('rate' in err_str and 'limit' in err_str) or '429' in err_str


def clean_llm_json(raw: str) -> str:
    """Strip common LLM output artifacts that break JSON parsing.

    - Leading/trailing whitespace
    - Markdown code fences
    - BOM characters
    """
    s = raw.strip()
    # Strip BOM
    s = s.lstrip('\ufeff')
    # Strip markdown fences
    if s.startswith('```'):
        first_newline = s.index('\n') if '\n' in s else len(s)
        s = s[first_newline + 1:]
    if s.endswith('```'):
        s = s[:-3].rstrip()
    return s


def extract_patch(
    node_id: int,
    lookback: int = 3,
    conn=None,
) -> dict:
    """Run the extraction pipeline for a single node.

    Returns a patch result dict:
    {
        'node_id': int,
        'patch_json': str | None,
        'patch_hash': str | None,
        'applied': 0,
        'rejection_reason': str | None,
        'model_id': str,
        'extraction_time_ms': int,
        'raw_output': str,
    }
    """
    from episodic.config import config
    from episodic.llm import query_llm

    model = config.get('extraction_model') or 'claude-haiku-4-5-20251001'
    params = config.get_model_params('extraction', model=model)

    start_ms = int(time.time() * 1000)

    # Step 1: Build context
    context = build_extraction_context(node_id, lookback, conn)
    if context is None:
        return {
            'node_id': node_id,
            'patch_json': None,
            'patch_hash': None,
            'applied': 0,
            'rejection_reason': 'non_user_node',
            'model_id': model,
            'extraction_time_ms': int(time.time() * 1000) - start_ms,
            'raw_output': '',
        }

    # Step 2: Format input
    user_message = format_extraction_input(
        node_id=context['node_id'],
        source_text=context['source_text'],
        recent_context=context['recent_context'],
        entity_dictionary=context['entity_dictionary'],
        kg_neighborhood=context['kg_neighborhood'],
    )

    # Step 3: Call LLM
    system_prompt = EXTRACTION_SYSTEM_PROMPT

    # Build extra kwargs for JSON mode
    extra_kwargs = dict(params)
    extra_kwargs.pop('stop', None)  # Remove stop sequences for JSON mode

    # Request JSON mode for supported providers
    if model and 'ollama/' in model.lower():
        extra_kwargs['format'] = 'json'
    else:
        extra_kwargs['response_format'] = {'type': 'json_object'}

    raw_output = ''
    for attempt in range(MAX_RETRIES + 1):
        try:
            response_text, cost_info = query_llm(
                prompt=user_message,
                model=model,
                system_message=system_prompt,
                **extra_kwargs,
            )
            raw_output = response_text or ''
            break  # success
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** (attempt + 1))  # 2, 4, 8s
                continue
            # Non-rate-limit error or exhausted retries
            elapsed = int(time.time() * 1000) - start_ms
            return {
                'node_id': node_id,
                'patch_json': None,
                'patch_hash': None,
                'applied': 0,
                'rejection_reason': f'llm_call_failed: {e}',
                'model_id': model,
                'extraction_time_ms': elapsed,
                'raw_output': '',
            }

    # Step 4: Parse response
    cleaned = clean_llm_json(raw_output)
    patch = _try_parse_json(cleaned)

    # Step 5: Retry if JSON parse failed
    if patch is None:
        retry_prompt = system_prompt + '\n\n' + RETRY_ADDENDUM
        try:
            response_text, cost_info = query_llm(
                prompt=user_message,
                model=model,
                system_message=retry_prompt,
                **extra_kwargs,
            )
            raw_output = response_text or ''
            cleaned = clean_llm_json(raw_output)
            patch = _try_parse_json(cleaned)
        except Exception:
            pass

    # Step 6: If still failed
    elapsed = int(time.time() * 1000) - start_ms
    if patch is None:
        return {
            'node_id': node_id,
            'patch_json': None,
            'patch_hash': None,
            'applied': 0,
            'rejection_reason': 'extractor_output_invalid_json',
            'model_id': model,
            'extraction_time_ms': elapsed,
            'raw_output': raw_output,
        }

    # Step 7: Compute canonical JSON and hash
    canonical = json.dumps(patch, sort_keys=True, separators=(',', ':'))
    patch_hash = hashlib.sha256(canonical.encode()).hexdigest()

    return {
        'node_id': node_id,
        'patch_json': canonical,
        'patch_hash': patch_hash,
        'applied': 0,  # Set by applicator
        'rejection_reason': None,
        'model_id': model,
        'extraction_time_ms': elapsed,
        'raw_output': raw_output,
    }


def _try_parse_json(s: str) -> dict | None:
    """Attempt to parse a string as JSON dict. Returns None on failure."""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
