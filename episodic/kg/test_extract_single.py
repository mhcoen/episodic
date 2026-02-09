"""Single-paragraph extraction test harness.

Usage:
    python -m episodic.kg.test_extract_single [--model MODEL] [--text TEXT]

No database writes. No HWM. No batch.
Text in -> LLM extraction -> validation -> human-readable summary.
"""

import hashlib
import json
import sys
import time

from .prompt_template import (
    EXTRACTION_SYSTEM_PROMPT,
    RETRY_ADDENDUM,
    format_extraction_input,
    normalize_text,
)
from .extractor import clean_llm_json, _try_parse_json
from .validator import validate_patch, repair_patch

DEFAULT_TEXT = (
    "The household is chaotic. Biscuit is a seven-year-old golden retriever"
    " -- my wife Sarah picked her out as a puppy. Patches is a three-year-old"
    " border collie, completely insane energy levels. The MacBook Pro M3 Max"
    " is the daily driver, specced with 64 gigs of RAM for running local"
    " models. Emma is a sophomore at MIT studying computer science."
)

NODE_ID = 999  # Fake node_id for testing


def run_extraction(text: str, model: str) -> None:
    """Run full extraction + validation on a single paragraph."""
    from episodic.llm import query_llm

    source_text = normalize_text(text)
    print(f"=== Model: {model} ===")
    print(f"=== Source text ({len(source_text)} chars) ===")
    print(f"  {source_text}")
    print()

    # Build input (no context, no entity dictionary, no neighborhood)
    user_message = format_extraction_input(
        node_id=NODE_ID,
        source_text=source_text,
        recent_context=[],
        entity_dictionary=[],
        kg_neighborhood=[],
    )

    # Call LLM
    extra_kwargs: dict = {'temperature': 0.0}
    if model and 'ollama/' in model.lower():
        extra_kwargs['format'] = 'json'
    else:
        extra_kwargs['response_format'] = {'type': 'json_object'}

    print("Calling LLM...")
    t0 = time.time()
    try:
        response_text, cost_info = query_llm(
            prompt=user_message,
            model=model,
            system_message=EXTRACTION_SYSTEM_PROMPT,
            **extra_kwargs,
        )
    except Exception as e:
        print(f"LLM call failed: {e}")
        return

    elapsed = time.time() - t0
    raw = response_text or ''
    print(f"LLM responded in {elapsed:.1f}s")
    print()

    # Print raw JSON
    print("=== RAW LLM RESPONSE ===")
    try:
        formatted = json.dumps(json.loads(clean_llm_json(raw)), indent=2)
        print(formatted)
    except Exception:
        print(raw)
    print()

    # Parse
    cleaned = clean_llm_json(raw)
    patch = _try_parse_json(cleaned)
    if patch is None:
        print("ERROR: Could not parse LLM output as JSON")
        return

    # Repair
    patch = repair_patch(patch, source_text)

    # Validate (no topic scope, no existing entities, no DB)
    vresult = validate_patch(
        patch=patch,
        source_text=source_text,
        node_id=NODE_ID,
        topic_entity_ids=set(),
        existing_canonical_keys={},
        conn=None,
        entity_dictionary=[],
    )

    # Print results
    cp = vresult.cleaned_patch or patch

    print("=== ENTITIES ===")
    for e in cp.get('entities', []):
        print(f"  {e['entity_key']}: {e['canonical_name']} ({e['entity_type']})")
    if not cp.get('entities'):
        print("  (none)")

    print()
    print("=== MENTIONS ===")
    for m in cp.get('mentions', []):
        print(
            f"  {m['mention_key']}: \"{m['surface_text']}\" "
            f"[{m['span_start']}:{m['span_end']}] "
            f"-> {m.get('entity_ref', 'null')}"
        )
    if not cp.get('mentions'):
        print("  (none)")

    print()
    print("=== EDGES (kept) ===")
    for edge in cp.get('edges', []):
        subj = edge['subj_ref']
        obj = edge['obj_ref']
        pred = edge['predicate']
        # Resolve names
        subj_name = _resolve_name(subj, cp.get('entities', []))
        obj_name = _resolve_name(obj, cp.get('entities', []))
        print(f"  {subj_name} --{pred}--> {obj_name}")
    if not cp.get('edges'):
        print("  (none)")

    print()
    print("=== EDGES STRIPPED (by validator) ===")
    stripped = [w for w in vresult.warnings if w.startswith('stripped:')]
    for w in stripped:
        print(f"  {w}")
    if not stripped:
        print("  (none)")

    print()
    print("=== WARNINGS ===")
    non_strip = [w for w in vresult.warnings if not w.startswith('stripped:')]
    for w in non_strip:
        print(f"  {w}")
    if not non_strip:
        print("  (none)")

    print()
    print("=== SUMMARY ===")
    print(f"  Entities: {len(cp.get('entities', []))}")
    print(f"  Mentions: {len(cp.get('mentions', []))}")
    print(f"  Edges kept: {len(cp.get('edges', []))}")
    print(f"  Edges stripped: {len(stripped)}")
    print(f"  Warnings: {len(non_strip)}")

    # Notes
    notes = patch.get('notes')
    if notes:
        print()
        print(f"=== NOTES ===")
        print(f"  {notes}")


def _resolve_name(ref: str, entities: list[dict]) -> str:
    """Resolve entity ref to display name."""
    if ref == 'user:self':
        return '<user>'
    for e in entities:
        if e.get('entity_key') == ref:
            return e['canonical_name']
    return ref


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Single-paragraph KG extraction test')
    parser.add_argument('--model', default='claude-haiku-4-5-20251001', help='Model to use')
    parser.add_argument('--text', default=None, help='Text to extract from')
    args = parser.parse_args()

    text = args.text or DEFAULT_TEXT
    run_extraction(text, args.model)
