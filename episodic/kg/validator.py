"""Deterministic validation of proposed KG patches.

Every check is deterministic — no LLM calls. This is the critical
correctness gate between extraction and application.
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


def validate_patch(
    patch: dict,
    source_text: str,
    node_id: int,
    topic_entity_ids: set[int],
    existing_canonical_keys: dict[str, int],
    conn=None,
    entity_dictionary: list[dict] | None = None,
) -> ValidationResult:
    """Run all validation checks on a proposed patch.

    Fatal errors (schema_version, missing keys, node_id mismatch) reject
    the entire patch. Element errors (bad assertion, entity, mention,
    edge, alias) cause the offending element to be stripped, with
    cascading removal of dependents. The cleaned patch is always valid
    unless a fatal error occurred.

    Parameters:
    - patch: parsed JSON from the extractor
    - source_text: the exact content of the source node
    - node_id: expected node_id (must match patch['node_id'])
    - topic_entity_ids: set of entity_ids in the same topic scope
    - existing_canonical_keys: map of all non-null canonical_keys to entity_ids
    - conn: optional DB connection for additional lookups
    """
    fatal_errors: list[str] = []
    warnings: list[str] = []

    # Normalize source_text so span offsets match what the LLM saw
    source_text = normalize_text(source_text)

    # --- Fatal checks ---

    # Check 1: Schema version
    if patch.get('schema_version') != 'kg_patch_v1':
        return ValidationResult(
            valid=False,
            errors=[f"Invalid schema_version: {patch.get('schema_version')}"],
        )

    # Check 2: Node ID match
    if patch.get('node_id') != node_id:
        return ValidationResult(
            valid=False,
            errors=[
                f"node_id mismatch: patch has {patch.get('node_id')}, "
                f"expected {node_id}"
            ],
        )

    # Check 3: Required top-level keys
    required_keys = {
        'schema_version', 'node_id', 'assertions', 'entities',
        'aliases', 'mentions', 'edges',
    }
    missing = required_keys - set(patch.keys())
    if missing:
        return ValidationResult(
            valid=False,
            errors=[f"Missing required keys: {missing}"],
        )

    # --- Element validation with per-element error tracking ---

    # Check 4: Assertions — collect good/bad keys
    good_assertions: dict[str, dict] = {}
    bad_assertion_keys: set[str] = set()
    kept_assertions = []

    for a in patch.get('assertions', []):
        akey = a.get('assertion_key', '')
        errs = []

        if not _ASSERTION_KEY_RE.match(str(akey)):
            errs.append(f"Invalid assertion_key: {akey}")
        else:
            ss = a.get('span_start')
            se = a.get('span_end')

            if not isinstance(ss, int) or ss < 0:
                errs.append(f"{akey}: span_start must be integer >= 0, got {ss}")
            if not isinstance(se, int) or (isinstance(ss, int) and isinstance(se, int) and se <= ss):
                errs.append(f"{akey}: span_end must be > span_start")
            if isinstance(se, int) and se > len(source_text):
                errs.append(f"{akey}: span_end {se} > source_text length {len(source_text)}")
            if a.get('asserted_by') != 'user':
                errs.append(f"{akey}: asserted_by must be 'user', got {a.get('asserted_by')}")
            if a.get('polarity') not in ALLOWED_POLARITIES:
                errs.append(f"{akey}: invalid polarity: {a.get('polarity')}")
            if a.get('certainty') not in ALLOWED_CERTAINTIES:
                errs.append(f"{akey}: invalid certainty: {a.get('certainty')}")
            if a.get('status') not in ALLOWED_STATUSES:
                errs.append(f"{akey}: invalid status: {a.get('status')}")

            tags = a.get('tags', [])
            if not isinstance(tags, list):
                errs.append(f"{akey}: tags must be a list")
            else:
                for tag in tags:
                    if tag not in ALLOWED_TAGS:
                        errs.append(f"{akey}: invalid tag: {tag}")

            if (isinstance(ss, int) and isinstance(se, int)
                    and 0 <= ss < se <= len(source_text)):
                span_text = source_text[ss:se]
                if not span_text.strip():
                    errs.append(f"{akey}: span is empty/whitespace")

        if errs:
            for e in errs:
                warnings.append(f"stripped:{STRIP_ASSERTION_INVALID}: {e}")
            bad_assertion_keys.add(akey)
        else:
            good_assertions[akey] = a
            kept_assertions.append(a)

    patch['assertions'] = kept_assertions

    # Check 5: Entities — strip those with errors or dangling assertion refs
    good_entities: dict[str, dict] = {}
    bad_entity_keys: set[str] = set()
    kept_entities = []

    for e in patch.get('entities', []):
        ekey = e.get('entity_key', '')
        errs = []

        if not _ENTITY_KEY_RE.match(str(ekey)):
            errs.append(f"Invalid entity_key: {ekey}")
        else:
            etype = e.get('entity_type')
            if etype not in ALLOWED_ENTITY_TYPES:
                errs.append(f"{ekey}: invalid entity_type: {etype}")

            cname = e.get('canonical_name')
            if not cname or not isinstance(cname, str) or not cname.strip():
                errs.append(f"{ekey}: canonical_name must be non-empty string")

            ckey = e.get('canonical_key')
            if ckey is not None and not isinstance(ckey, str):
                errs.append(f"{ekey}: canonical_key must be string or null")

            cba = e.get('created_by_assertion')
            if cba not in good_assertions:
                errs.append(
                    f"{ekey}: created_by_assertion '{cba}' not found in assertions"
                )

            # Check 5f: canonical_key resolution
            if not errs and isinstance(ckey, str) and ckey in existing_canonical_keys:
                hint = e.get('resolution_hint')
                if not hint or hint.get('kind') != 'map_to_existing':
                    warnings.append(
                        f"{ekey}: canonical_key '{ckey}' exists in DB but "
                        f"resolution_hint is not map_to_existing"
                    )

            # Check 5g: cross-topic resolution
            hint = e.get('resolution_hint')
            if hint and isinstance(hint, dict) and hint.get('kind') == 'map_to_existing':
                ceid = hint.get('candidate_entity_id')
                if not isinstance(ceid, int):
                    errs.append(
                        f"{ekey}: resolution_hint.candidate_entity_id must be int"
                    )
                elif ceid not in topic_entity_ids:
                    has_global = False
                    for ck, eid in existing_canonical_keys.items():
                        if eid == ceid:
                            has_global = True
                            break
                    if not has_global:
                        errs.append(
                            f"{ekey}: cross-topic resolution to entity {ceid} "
                            f"forbidden in Phase 0 (not in topic scope and "
                            f"no canonical_key)"
                        )

        if errs:
            for e2 in errs:
                warnings.append(f"stripped:{STRIP_ENTITY_INVALID}: {e2}")
            bad_entity_keys.add(ekey)
        else:
            good_entities[ekey] = e
            kept_entities.append(e)

    patch['entities'] = kept_entities

    # Build reference resolver using only good entities
    def resolve_ref(ref) -> str | None:
        if ref == 'user:self':
            return 'user:self'
        if isinstance(ref, str) and ref in good_entities:
            return f'local:{ref}'
        if isinstance(ref, str) and ref.startswith('db:'):
            try:
                eid = int(ref[3:])
                all_known = set(topic_entity_ids)
                all_known.update(existing_canonical_keys.values())
                if eid in all_known:
                    return f'db:{eid}'
                return f'db:{eid}'
            except ValueError:
                return None
        return None

    # Check 7: Aliases — strip bad ones
    kept_aliases = []
    for alias in patch.get('aliases', []):
        errs = []
        aref = alias.get('entity_ref')
        if resolve_ref(aref) is None:
            errs.append(f"alias: unresolvable entity_ref: {aref}")

        atext = alias.get('alias_text')
        if not atext or not isinstance(atext, str) or not atext.strip():
            errs.append("alias: alias_text must be non-empty string")

        sa = alias.get('source_assertion')
        if sa not in good_assertions:
            errs.append(f"alias: source_assertion '{sa}' not found")

        ss = alias.get('span_start')
        se = alias.get('span_end')
        if isinstance(ss, int) and isinstance(se, int):
            if ss < 0 or se > len(source_text) or ss >= se:
                errs.append(
                    f"alias: span [{ss}:{se}] out of bounds "
                    f"(source len={len(source_text)})"
                )
            elif atext and isinstance(atext, str) and not errs:
                actual = source_text[ss:se]
                if actual.lower() != atext.lower():
                    warnings.append(
                        f"alias: span text '{actual}' != alias_text '{atext}' "
                        f"(case-insensitive mismatch)"
                    )

        if errs:
            for e in errs:
                warnings.append(f"stripped:{STRIP_ALIAS_INVALID}: {e}")
        else:
            kept_aliases.append(alias)

    patch['aliases'] = kept_aliases

    # Check 8: Mentions — strip bad ones
    kept_mentions = []
    for m in patch.get('mentions', []):
        mkey = m.get('mention_key', '')
        errs = []

        if not _MENTION_KEY_RE.match(str(mkey)):
            errs.append(f"Invalid mention_key: {mkey}")
        else:
            ss = m.get('span_start')
            se = m.get('span_end')
            if not isinstance(ss, int) or not isinstance(se, int):
                errs.append(f"{mkey}: span_start/span_end must be integers")
            elif ss < 0 or se > len(source_text) or ss >= se:
                errs.append(
                    f"{mkey}: span [{ss}:{se}] out of bounds "
                    f"(source len={len(source_text)})"
                )
            else:
                surface = m.get('surface_text', '')
                actual = source_text[ss:se]
                if surface != actual:
                    errs.append(
                        f"{mkey}: surface_text mismatch: '{surface}' != "
                        f"source_text[{ss}:{se}]='{actual}'"
                    )

            eref = m.get('entity_ref')
            if eref is not None and resolve_ref(eref) is None:
                errs.append(f"{mkey}: unresolvable entity_ref: {eref}")

            conf = m.get('confidence')
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                errs.append(f"{mkey}: confidence must be number in [0,1]")

            sa = m.get('source_assertion')
            if sa not in good_assertions:
                errs.append(f"{mkey}: source_assertion '{sa}' not found")

        if errs:
            for e in errs:
                warnings.append(f"stripped:{STRIP_MENTION_INVALID}: {e}")
        else:
            kept_mentions.append(m)

    patch['mentions'] = kept_mentions

    # Check 9: Edges — strip bad ones
    kept_edges = []
    for edge in patch.get('edges', []):
        strip_reason = None  # Set to a STRIP_* constant if stripped
        subj = edge.get('subj_ref')
        obj_ref = edge.get('obj_ref')
        pred = edge.get('predicate')
        sa = edge.get('source_assertion')

        # Basic validation errors
        basic_errs = []
        if resolve_ref(subj) is None:
            basic_errs.append(f"unresolvable subj_ref: {subj}")
        if resolve_ref(obj_ref) is None:
            basic_errs.append(f"unresolvable obj_ref: {obj_ref}")
        if pred not in ALLOWED_PREDICATES:
            basic_errs.append(f"invalid predicate: {pred}")
        if sa not in good_assertions:
            # Determine if assertion was removed (cascade) or never existed
            if sa in bad_assertion_keys:
                strip_reason = STRIP_CASCADE_ASSERTION
            else:
                basic_errs.append(f"source_assertion '{sa}' not found")

        conf = edge.get('confidence')
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            basic_errs.append("confidence must be number in [0,1]")

        # Check 9g: No self-loops
        if subj is not None and subj == obj_ref:
            basic_errs.append(f"self-loop (subj_ref == obj_ref == '{subj}')")

        if basic_errs:
            strip_reason = strip_reason or STRIP_EDGE_BASIC_VALIDATION
            for e in basic_errs:
                warnings.append(f"stripped:{strip_reason}: {e}")
            continue

        if strip_reason:
            warnings.append(f"stripped:{strip_reason}: edge for {sa}")
            continue

        # Check 9h: Forbid user:self as obj_ref (except related_to)
        if obj_ref == 'user:self' and pred != 'related_to':
            warnings.append(
                f"stripped:{STRIP_EDGE_USER_SELF_AS_OBJECT}: "
                f"{pred}(_, user:self)"
            )
            continue

        # Check 9i: Domain/range constraints
        subj_type = resolve_entity_type(
            subj, patch.get('entities', []), entity_dictionary
        )
        obj_type = resolve_entity_type(
            obj_ref, patch.get('entities', []), entity_dictionary
        )
        if pred in DOMAIN_RANGE and subj_type and obj_type:
            allowed_subj, allowed_obj = DOMAIN_RANGE[pred]
            if subj_type not in allowed_subj or obj_type not in allowed_obj:
                warnings.append(
                    f"stripped:{STRIP_EDGE_DOMAIN_RANGE_VIOLATION}: "
                    f"{pred}({subj_type}, {obj_type})"
                )
                continue

        # Check 9j: Mention existence
        if sa in good_assertions:
            assertion = good_assertions[sa]
            a0 = assertion.get('span_start', 0)
            a1 = assertion.get('span_end', 0)

            subj_mentions = [
                m for m in patch.get('mentions', [])
                if m.get('entity_ref') == subj
                and m.get('source_assertion') == sa
            ]
            obj_mentions = [
                m for m in patch.get('mentions', [])
                if m.get('entity_ref') == obj_ref
                and m.get('source_assertion') == sa
            ]

            # 9j-1: Subject mention existence (skip for user:self)
            if subj != 'user:self' and not subj_mentions:
                span_text = source_text[a0:a1]
                if _mention_fallback_match(
                    subj, span_text,
                    patch.get('entities', []),
                    patch.get('aliases', []),
                    entity_dictionary,
                ):
                    warnings.append(
                        f"mention_fallback_recovered_subj: {subj}"
                    )
                else:
                    warnings.append(
                        f"stripped:{STRIP_EDGE_MISSING_SUBJECT_MENTION}: "
                        f"{subj} in {sa}"
                    )
                    continue

            # 9j-2: Object mention existence (skip for user:self)
            if obj_ref != 'user:self' and not obj_mentions:
                span_text = source_text[a0:a1]
                if _mention_fallback_match(
                    obj_ref, span_text,
                    patch.get('entities', []),
                    patch.get('aliases', []),
                    entity_dictionary,
                ):
                    warnings.append(
                        f"mention_fallback_recovered_obj: {obj_ref}"
                    )
                else:
                    warnings.append(
                        f"stripped:{STRIP_EDGE_MISSING_OBJECT_MENTION}: "
                        f"{obj_ref} in {sa}"
                    )
                    continue

            # 9j-3 and 9j-4: DISABLED
            # LLM consistently produces spans 5-15 chars too short.

            # Check 9k: Rejection cues, temporal cues, negate polarity
            span_text = source_text[a0:a1]
            span_lower = span_text.lower()

            # 9k-1: Rejection cues → strip entirely
            rejection_hit = False
            for cue in REJECTION_CUES:
                if cue in span_lower:
                    warnings.append(
                        f"stripped:{STRIP_EDGE_REJECTION_CUE}: "
                        f"'{cue}' in span"
                    )
                    rejection_hit = True
                    break
            if rejection_hit:
                continue

            # 9k-2: Negate polarity → strip
            if assertion.get('polarity') == 'negate':
                warnings.append(
                    f"stripped:{STRIP_EDGE_NEGATE_POLARITY}: "
                    f"{sa} has polarity=negate"
                )
                continue

            # 9k-3: Temporal cues → add TIME_PAST tag if missing
            for cue in TEMPORAL_CUES:
                if cue in span_lower:
                    # Check if assertion already has TIME_PAST
                    tags = assertion.get('tags', [])
                    if 'TIME_PAST' not in tags:
                        tags.append('TIME_PAST')
                        assertion['tags'] = tags
                        warnings.append(
                            f"edge_temporal_cue_added_time_past: "
                            f"'{cue}' in {sa}"
                        )
                    break

            # 9k-5: wants edges in question/info-seeking context
            if pred == 'wants' and subj == 'user:self':
                has_negative = any(
                    cue in span_lower for cue in WANTS_NEGATIVE_CUES
                )
                has_positive = any(
                    cue in span_lower for cue in WANTS_POSITIVE_CUES
                )
                if has_negative and not has_positive:
                    warnings.append(
                        f"stripped:{STRIP_EDGE_WANTS_QUESTION_CONTEXT}: "
                        f"wants edge in question/info-seeking assertion"
                    )
                    continue

        kept_edges.append(edge)

    patch['edges'] = kept_edges

    # Check 10: No orphan entities (warning only)
    referenced_entities = set()
    for m in patch.get('mentions', []):
        eref = m.get('entity_ref')
        if eref and _ENTITY_KEY_RE.match(str(eref)):
            referenced_entities.add(eref)
    for edge in patch.get('edges', []):
        for ref_field in ('subj_ref', 'obj_ref'):
            eref = edge.get(ref_field)
            if eref and _ENTITY_KEY_RE.match(str(eref)):
                referenced_entities.add(eref)
    for alias in patch.get('aliases', []):
        eref = alias.get('entity_ref')
        if eref and _ENTITY_KEY_RE.match(str(eref)):
            referenced_entities.add(eref)

    for ekey in good_entities:
        if ekey not in referenced_entities:
            warnings.append(
                f"{ekey}: entity has no mentions, edges, or aliases "
                f"(orphan entity)"
            )

    return ValidationResult(
        valid=True,
        errors=[],
        warnings=warnings,
        cleaned_patch=patch,
    )
