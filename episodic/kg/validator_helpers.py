"""Constants, helpers, and ValidationResult for KG patch validation.

Split out of validator.py so the correctness-critical validate_patch stays in
one focused module. Re-exported from validator.py (import *) so external imports
(STRIP_* codes, DOMAIN_RANGE, repair_patch, resolve_entity_type,
ValidationResult, ...) are unchanged.
"""

import re
from dataclasses import dataclass, field

from .prompt_template import normalize_text


VALIDATOR_VERSION = "kg_validator_v1"

# --- Strip reason codes (every removal must have exactly one) ---
STRIP_ASSERTION_INVALID = 'assertion_invalid'
STRIP_ENTITY_INVALID = 'entity_invalid'
STRIP_MENTION_INVALID = 'mention_invalid'
STRIP_ALIAS_INVALID = 'alias_invalid'
STRIP_EDGE_USER_SELF_AS_OBJECT = 'edge_user_self_as_object'
STRIP_EDGE_DOMAIN_RANGE_VIOLATION = 'edge_domain_range_violation'
STRIP_EDGE_MISSING_SUBJECT_MENTION = 'edge_missing_subject_mention'
STRIP_EDGE_MISSING_OBJECT_MENTION = 'edge_missing_object_mention'
STRIP_EDGE_REJECTION_CUE = 'edge_rejection_cue_in_assertion'
STRIP_EDGE_NEGATE_POLARITY = 'edge_negate_polarity'
STRIP_EDGE_BASIC_VALIDATION = 'edge_basic_validation'
STRIP_CASCADE_ASSERTION = 'cascade_from_assertion_removal'
STRIP_CASCADE_ENTITY = 'cascade_from_entity_removal'
STRIP_EDGE_WANTS_QUESTION_CONTEXT = 'edge_wants_in_question_context'

# --- Rejection and temporal cue lists for check 9k ---
REJECTION_CUES = [
    "declined", "turned down", "rejected", "refused",
    "didn't accept", "chose not to", "didn't take",
    "offered me a position", "offered a position",
]

TEMPORAL_CUES = [
    "used to", "previously", "no longer", "stopped using",
    "replaced", "switched from", "before I switched",
    "back when", "formerly", "in the past",
]

# --- Wants-in-question-context cue lists for check 9k-5 ---
WANTS_NEGATIVE_CUES = [
    '?', 'tell me about', 'what is', 'what are',
    'how does', 'how do', 'can you explain', 'explain',
    'thoughts on', 'describe', 'how do i',
]

WANTS_POSITIVE_CUES = [
    'i want to', 'i need to', "i'm looking for",
    'my goal is', 'i plan to', "i'm hoping to",
    "i'd like to", 'i decided to', 'i wish i could',
    'i want a', 'i need a',
]

# String-to-number confidence coercion
_CONFIDENCE_MAP = {
    'high': 0.95,
    'very high': 0.98,
    'medium': 0.75,
    'moderate': 0.75,
    'low': 0.5,
    'very low': 0.3,
}


def repair_patch(patch: dict, source_text: str) -> dict:
    """Best-effort repair of common LLM extraction errors.

    Fixes:
    - Mention span offsets that don't match surface_text (re-anchored)
    - String confidence values coerced to floats
    - Assertion spans that overshoot source_text length

    Mutates and returns the patch dict. Run BEFORE validate_patch().
    """
    src_len = len(source_text)

    # Repair assertion spans
    for a in patch.get('assertions', []):
        if not isinstance(a, dict):
            continue
        se = a.get('span_end')
        if isinstance(se, int) and se > src_len:
            a['span_end'] = src_len

    # Repair assertion tags (unwrap dicts, coerce to allowed strings)
    for a in patch.get('assertions', []):
        if not isinstance(a, dict):
            continue
        tags = a.get('tags', [])
        if not isinstance(tags, list):
            a['tags'] = []
            continue
        repaired_tags = []
        for tag in tags:
            if isinstance(tag, str):
                repaired_tags.append(tag)
            elif isinstance(tag, dict):
                # LLM sometimes wraps tags as {"time": "TIME_PAST"} — extract the value
                for v in tag.values():
                    if isinstance(v, str):
                        repaired_tags.append(v)
        a['tags'] = repaired_tags

    # Repair mentions
    for m in patch.get('mentions', []):
        if not isinstance(m, dict):
            continue
        # Fix confidence
        conf = m.get('confidence')
        if isinstance(conf, str):
            m['confidence'] = _coerce_confidence(conf)

        # Fix span offsets
        surface = m.get('surface_text', '')
        ss = m.get('span_start')
        se = m.get('span_end')
        if not surface or not isinstance(ss, int) or not isinstance(se, int):
            continue
        actual = source_text[ss:se] if 0 <= ss < se <= src_len else ''
        if actual == surface:
            continue  # Already correct
        # Search for surface_text in source_text
        fixed = _find_best_span(surface, source_text, ss)
        if fixed is not None:
            m['span_start'], m['span_end'] = fixed

    # Repair edge confidence
    for edge in patch.get('edges', []):
        if not isinstance(edge, dict):
            continue
        conf = edge.get('confidence')
        if isinstance(conf, str):
            edge['confidence'] = _coerce_confidence(conf)

    return patch


def _coerce_confidence(val: str) -> float:
    """Convert string confidence to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return _CONFIDENCE_MAP.get(val.lower().strip(), 0.9)


def _find_best_span(
    surface: str, source: str, hint_start: int,
) -> tuple[int, int] | None:
    """Find the best-matching span for surface_text in source.

    If surface appears exactly once, use that position.
    If multiple, pick the one closest to hint_start.
    Returns (start, end) or None if not found.
    """
    positions = []
    start = 0
    while True:
        idx = source.find(surface, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        # Try case-insensitive as fallback
        source_lower = source.lower()
        surface_lower = surface.lower()
        start = 0
        while True:
            idx = source_lower.find(surface_lower, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1

    if not positions:
        return None

    # Pick closest to hint_start
    best = min(positions, key=lambda p: abs(p - hint_start))
    return (best, best + len(surface))

ALLOWED_ENTITY_TYPES = {'person', 'artifact', 'topic', 'org'}
ALLOWED_PREDICATES = {
    'uses', 'wants', 'prefers', 'role', 'has', 'located_at',
    'part_of', 'related_to', 'is_a', 'powered_by',
    'studies', 'affiliated_with', 'works_on',  # Phase 1.2
    'deadline', 'scheduled_for', 'starts_at', 'ends_at', 'recurring',  # Temporal
}
ALLOWED_POLARITIES = {'affirm', 'negate'}
ALLOWED_CERTAINTIES = {'explicit', 'hedged'}
ALLOWED_STATUSES = {'active'}
ALLOWED_TAGS = {
    'SENTIMENT_POS', 'SENTIMENT_NEG',
    'PROFICIENCY_LOW', 'PROFICIENCY_HIGH',
    'CONSTRAINT_HARD', 'CONSTRAINT_SOFT',
    'TIME_PAST', 'TIME_FUTURE',
}

_ASSERTION_KEY_RE = re.compile(r'^a\d+$')
_ENTITY_KEY_RE = re.compile(r'^e\d+$')
_MENTION_KEY_RE = re.compile(r'^m\d+$')

# Domain and range constraints per predicate.
# Keys: predicate name. Values: (allowed_subj_types, allowed_obj_types).
DOMAIN_RANGE: dict[str, tuple[set[str], set[str]]] = {
    'uses':       ({'person'},                     {'artifact', 'topic', 'org'}),
    'wants':      ({'person'},                     {'artifact', 'topic', 'org'}),
    'prefers':    ({'person'},                     {'artifact', 'topic', 'org'}),
    'role':       ({'person'},                     {'topic'}),
    'related_to': ({'person'},                     {'person'}),
    'located_at': ({'person', 'artifact', 'org'},  {'org'}),
    'part_of':    ({'artifact', 'org'},            {'artifact', 'org'}),
    'is_a':       ({'person', 'artifact', 'org'},  {'topic'}),
    'powered_by': ({'artifact'},                   {'artifact', 'topic'}),
    'has':        ({'person', 'artifact', 'org'},  {'person', 'artifact', 'topic', 'org'}),
    'studies':        ({'person'},                     {'topic'}),                       # Phase 1.2
    'affiliated_with':({'person', 'org'},              {'org'}),                         # Phase 1.2
    'works_on':       ({'person'},                     {'artifact', 'topic'}),           # Phase 1.2
    'deadline':       ({'artifact', 'topic', 'org'},              {'topic'}),           # Temporal
    'scheduled_for':  ({'person', 'artifact', 'topic', 'org'},   {'topic'}),           # Temporal
    'starts_at':      ({'person', 'artifact', 'topic', 'org'},   {'topic'}),           # Temporal
    'ends_at':        ({'person', 'artifact', 'topic', 'org'},   {'topic'}),           # Temporal
    'recurring':      ({'artifact', 'topic', 'org'},              {'topic'}),           # Temporal
}


def _resolve_canonical_names(
    entity_ref: str,
    patch_entities: list[dict],
    patch_aliases: list[dict],
    entity_dictionary: list[dict] | None = None,
) -> list[str]:
    """Resolve an entity reference to its canonical_name + aliases.

    Returns a list of name strings (canonical_name first, then aliases).
    Empty list if unresolvable.
    """
    names: list[str] = []
    if _ENTITY_KEY_RE.match(entity_ref):
        for ent in patch_entities:
            if ent.get('entity_key') == entity_ref:
                cn = ent.get('canonical_name')
                if cn:
                    names.append(cn)
                break
        # Also check patch aliases for this entity_ref
        for alias in patch_aliases:
            if alias.get('entity_ref') == entity_ref:
                at = alias.get('alias_text')
                if at and at not in names:
                    names.append(at)
    elif entity_ref.startswith('db:'):
        try:
            eid = int(entity_ref[3:])
        except ValueError:
            return names
        if entity_dictionary:
            for ent in entity_dictionary:
                if ent.get('entity_id') == eid:
                    cn = ent.get('canonical_name')
                    if cn:
                        names.append(cn)
                    for a in ent.get('aliases', []):
                        if a and a not in names:
                            names.append(a)
                    break
    return names


def _mention_fallback_match(
    entity_ref: str,
    span_text: str,
    patch_entities: list[dict],
    patch_aliases: list[dict],
    entity_dictionary: list[dict] | None = None,
) -> bool:
    """Check if entity's canonical_name or alias appears in span text.

    Case-insensitive substring match. Used as fallback when no mention
    exists for an edge's entity_ref.
    """
    names = _resolve_canonical_names(
        entity_ref, patch_entities, patch_aliases, entity_dictionary,
    )
    if not names:
        return False
    span_lower = span_text.lower()
    return any(name.lower() in span_lower for name in names)


def resolve_entity_type(
    entity_ref: str,
    patch_entities: list[dict],
    entity_dictionary: list[dict] | None = None,
) -> str | None:
    """Resolve an entity reference to its entity_type string.

    - "user:self" → "person"
    - "eN" → lookup from patch_entities list
    - "db:N" → lookup from entity_dictionary list
    Returns None if unresolvable.
    """
    if entity_ref == 'user:self':
        return 'person'
    if _ENTITY_KEY_RE.match(entity_ref):
        for ent in patch_entities:
            if ent.get('entity_key') == entity_ref:
                return ent.get('entity_type')
        return None
    if entity_ref.startswith('db:'):
        try:
            eid = int(entity_ref[3:])
        except ValueError:
            return None
        if entity_dictionary:
            for ent in entity_dictionary:
                if ent.get('entity_id') == eid:
                    return ent.get('entity_type')
        return None
    return None


@dataclass
class ValidationResult:
    """Immutable result of patch validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_patch: dict | None = None


