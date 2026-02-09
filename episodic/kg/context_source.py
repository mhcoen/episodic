"""KG read-side context injection.

Detects entity mentions in user input, retrieves relevant edges from the
knowledge graph, applies bounded closure rules, and formats facts for
injection into the conversation context. No LLM calls on the read path.
"""

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from episodic.config import config
from episodic.debug_utils import debug_print


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EdgeFact:
    subj_name: str
    predicate: str
    obj_name: str
    source_node_id: int
    rank_score: float
    tags: list[str] = field(default_factory=list)
    assertion_id: int | None = None


@dataclass
class DerivedFact:
    subj_name: str
    predicate: str
    obj_name: str
    rule: str
    source_node_ids: list[int] = field(default_factory=list)


@dataclass
class KGContextResult:
    text: str
    matched_entities: list[dict]
    edge_count: int
    derived_count: int
    budget_used: int
    budget_total: int
    cache_status: str  # "hit" or "rebuilt"
    edges: list['EdgeFact'] = field(default_factory=list)
    derived: list['DerivedFact'] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREDICATE_PRIORITY = {
    'has': 0,
    'is_a': 1,
    'located_at': 2,
    'part_of': 3,
    'related_to': 4,
    'powered_by': 5,
    'role': 6,
    'uses': 7,
    'prefers': 8,
    'wants': 9,
}

PAST_TENSE_CUES = re.compile(
    r'\b(used to|previously|before|back when|formerly|in the past|once had)\b',
    re.IGNORECASE,
)

_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def _normalize_surface(text: str) -> str:
    """Casefold, NFC-normalize, strip punctuation."""
    text = unicodedata.normalize('NFC', text.casefold())
    return _PUNCT_RE.sub('', text).strip()


# ---------------------------------------------------------------------------
# MentionDictionary
# ---------------------------------------------------------------------------

class MentionDictionary:
    """Maps normalized surface forms to entity IDs."""

    def __init__(self):
        self._dict: dict[str, list[tuple[int, float, str]]] = {}
        self._hwm: str = ""

    def _needs_rebuild(self, conn: sqlite3.Connection) -> bool:
        """Check if the dictionary needs rebuilding."""
        try:
            row = conn.execute(
                "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
            ).fetchone()
            current_hwm = row[0] if row else "0"
        except sqlite3.OperationalError:
            return False
        return current_hwm != self._hwm

    def rebuild(self, conn: sqlite3.Connection) -> str:
        """Rebuild the dictionary from the database. Returns cache status."""
        if not self._needs_rebuild(conn):
            return "hit"

        new_dict: dict[str, list[tuple[int, float, str]]] = {}

        # Load canonical names
        try:
            cursor = conn.execute(
                "SELECT entity_id, canonical_name FROM kg_entities"
            )
            for entity_id, canonical_name in cursor.fetchall():
                key = _normalize_surface(canonical_name)
                if key:
                    new_dict.setdefault(key, []).append(
                        (entity_id, 1.0, "canonical")
                    )
        except sqlite3.OperationalError:
            pass

        # Load aliases
        try:
            cursor = conn.execute(
                "SELECT entity_id, alias FROM kg_entity_aliases"
            )
            for entity_id, alias in cursor.fetchall():
                key = _normalize_surface(alias)
                if key:
                    new_dict.setdefault(key, []).append(
                        (entity_id, 0.8, "alias")
                    )
        except sqlite3.OperationalError:
            pass

        # Load curations (alias_add minus alias_remove)
        try:
            adds: dict[int, set[str]] = {}
            removes: dict[int, set[str]] = {}
            cursor = conn.execute(
                "SELECT entity_id, curation_type, value FROM kg_curations"
            )
            for entity_id, ctype, value in cursor.fetchall():
                if ctype == 'alias_add':
                    adds.setdefault(entity_id, set()).add(value)
                elif ctype == 'alias_remove':
                    removes.setdefault(entity_id, set()).add(value)

            for entity_id, alias_set in adds.items():
                removed = removes.get(entity_id, set())
                for alias in alias_set - removed:
                    key = _normalize_surface(alias)
                    if key:
                        new_dict.setdefault(key, []).append(
                            (entity_id, 0.8, "alias")
                        )
        except sqlite3.OperationalError:
            pass

        # Update HWM
        try:
            row = conn.execute(
                "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
            ).fetchone()
            self._hwm = row[0] if row else "0"
        except sqlite3.OperationalError:
            self._hwm = "0"

        self._dict = new_dict
        return "rebuilt"

    def detect_mentions(
        self, user_text: str, max_entities: int = 5
    ) -> list[tuple[int, str, float]]:
        """Detect entity mentions in user text. Longest-match-first.

        Returns list of (entity_id, surface_form, weight).
        """
        normalized_text = _normalize_surface(user_text)
        if not normalized_text:
            return []

        # Sort surface forms by length descending for longest-match-first
        sorted_forms = sorted(self._dict.keys(), key=len, reverse=True)

        matches: list[tuple[int, str, float]] = []
        seen_entity_ids: set[int] = set()
        consumed_ranges: list[tuple[int, int]] = []

        for form in sorted_forms:
            if len(matches) >= max_entities:
                break
            # Find all occurrences of this form in the text
            start = 0
            while True:
                idx = normalized_text.find(form, start)
                if idx == -1:
                    break
                end = idx + len(form)

                # Check word boundaries
                if idx > 0 and normalized_text[idx - 1].isalnum():
                    start = end
                    continue
                if end < len(normalized_text) and normalized_text[end].isalnum():
                    start = end
                    continue

                # Check overlap with already consumed ranges
                overlaps = any(
                    not (end <= cs or idx >= ce) for cs, ce in consumed_ranges
                )
                if overlaps:
                    start = end
                    continue

                # Pick the best (highest weight) entity for this form
                for entity_id, weight, kind in self._dict[form]:
                    if entity_id not in seen_entity_ids:
                        matches.append((entity_id, form, weight))
                        seen_entity_ids.add(entity_id)
                        consumed_ranges.append((idx, end))
                        break

                start = end

        return matches[:max_entities]


# ---------------------------------------------------------------------------
# retrieve_neighborhood
# ---------------------------------------------------------------------------

def retrieve_neighborhood(
    entity_id: int,
    conn: sqlite3.Connection,
    user_text: str = "",
    co_mentioned_ids: Optional[set[int]] = None,
    max_edges: int = 5,
) -> list[EdgeFact]:
    """Retrieve relevant edges for an entity.

    Filters by active assertions, deduplicates, ranks, and caps.
    """
    include_past = config.get('kg_include_past', False) or bool(
        PAST_TENSE_CUES.search(user_text)
    )
    co_mentioned = co_mentioned_ids or set()

    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT e.edge_id, e.subj_entity_id, e.predicate, e.obj_entity_id,
                   e.assertion_id,
                   a.source_node_id, a.tags,
                   s.canonical_name AS subj_name,
                   o.canonical_name AS obj_name
            FROM kg_edges e
            JOIN kg_assertions a ON e.assertion_id = a.assertion_id
            JOIN kg_entities s ON e.subj_entity_id = s.entity_id
            JOIN kg_entities o ON e.obj_entity_id = o.entity_id
            WHERE (e.subj_entity_id = ? OR e.obj_entity_id = ?)
              AND a.status = 'active'
            ORDER BY a.source_node_id DESC
            """,
            (entity_id, entity_id),
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.row_factory = None

    # Filter TIME_PAST
    filtered = []
    for row in rows:
        tags_raw = row['tags']
        tags = _parse_tags(tags_raw)
        if 'TIME_PAST' in tags and not include_past:
            continue
        filtered.append((row, tags))

    # Dedup: same (subj, predicate, obj) — keep most recent source_node_id
    seen_triples: dict[tuple, dict] = {}
    for row, tags in filtered:
        triple = (row['subj_entity_id'], row['predicate'], row['obj_entity_id'])
        if triple not in seen_triples:
            seen_triples[triple] = (row, tags)

    # Rank
    facts: list[EdgeFact] = []
    for (row, tags) in seen_triples.values():
        other_id = (
            row['obj_entity_id']
            if row['subj_entity_id'] == entity_id
            else row['subj_entity_id']
        )
        co_mentioned_bonus = 10.0 if other_id in co_mentioned else 0.0
        recency = row['source_node_id'] / 1_000_000.0  # normalize
        pred_priority = PREDICATE_PRIORITY.get(row['predicate'], 5)
        rank_score = co_mentioned_bonus + recency - pred_priority * 0.01

        facts.append(EdgeFact(
            subj_name=row['subj_name'],
            predicate=row['predicate'],
            obj_name=row['obj_name'],
            source_node_id=row['source_node_id'],
            rank_score=rank_score,
            tags=tags,
            assertion_id=row['assertion_id'],
        ))

    facts.sort(key=lambda f: f.rank_score, reverse=True)
    return facts[:max_edges]


def _parse_tags(tags_raw) -> list[str]:
    """Parse tags from DB (JSON string or None)."""
    if not tags_raw:
        return []
    if isinstance(tags_raw, list):
        return tags_raw
    try:
        import json
        parsed = json.loads(tags_raw)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
# apply_closure_rules
# ---------------------------------------------------------------------------

def apply_closure_rules(
    matched_entity_ids: list[int],
    edges: list[EdgeFact],
    conn: sqlite3.Connection,
    max_derived: int = 3,
) -> list[DerivedFact]:
    """Apply bounded closure rules. Two rules only.

    KINSHIP_LOCATION: user:self --related_to--> P, P --located_at--> O,
                      P is mentioned → derive "P located_at O"
    DEVICE_SPEC: user:self --has--> D, D --has--> S,
                 D is mentioned → derive "D has S"
    """
    derived: list[DerivedFact] = []
    matched_set = set(matched_entity_ids)

    # Get user:self entity_id
    try:
        row = conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
        ).fetchone()
        user_self_id = row[0] if row else None
    except sqlite3.OperationalError:
        return []

    if user_self_id is None:
        return []

    # KINSHIP_LOCATION: user:self --related_to--> P (in matched), P --located_at--> O
    for ef in edges:
        if len(derived) >= max_derived:
            break
        if ef.subj_name == '<user>' and ef.predicate == 'related_to':
            # Find the entity_id of the object person
            person_id = _entity_id_by_name(ef.obj_name, conn)
            if person_id is None or person_id not in matched_set:
                continue
            # Look for person --located_at--> location
            loc_edges = _query_edges_for(person_id, 'located_at', conn)
            for loc_edge in loc_edges:
                if len(derived) >= max_derived:
                    break
                derived.append(DerivedFact(
                    subj_name=ef.obj_name,
                    predicate='located_at',
                    obj_name=loc_edge['obj_name'],
                    rule='KINSHIP_LOCATION',
                    source_node_ids=[ef.source_node_id, loc_edge['source_node_id']],
                ))

    # DEVICE_SPEC: user:self --has--> D (in matched), D --has--> S
    for ef in edges:
        if len(derived) >= max_derived:
            break
        if ef.subj_name == '<user>' and ef.predicate == 'has':
            device_id = _entity_id_by_name(ef.obj_name, conn)
            if device_id is None or device_id not in matched_set:
                continue
            spec_edges = _query_edges_for(device_id, 'has', conn)
            for spec_edge in spec_edges:
                if len(derived) >= max_derived:
                    break
                derived.append(DerivedFact(
                    subj_name=ef.obj_name,
                    predicate='has',
                    obj_name=spec_edge['obj_name'],
                    rule='DEVICE_SPEC',
                    source_node_ids=[ef.source_node_id, spec_edge['source_node_id']],
                ))

    # Dedup derived facts by (subj, predicate, obj) — keep first occurrence
    seen_triples: set[tuple[str, str, str]] = set()
    deduped: list[DerivedFact] = []
    for d in derived:
        triple = (d.subj_name, d.predicate, d.obj_name)
        if triple not in seen_triples:
            seen_triples.add(triple)
            deduped.append(d)

    return deduped[:max_derived]


def _entity_id_by_name(name: str, conn: sqlite3.Connection) -> Optional[int]:
    """Look up entity_id by canonical_name."""
    try:
        row = conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_name = ?",
            (name,),
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def _query_edges_for(
    entity_id: int, predicate: str, conn: sqlite3.Connection
) -> list[dict]:
    """Query edges for a specific entity and predicate (2-hop lookup)."""
    try:
        cursor = conn.execute(
            """
            SELECT e.subj_entity_id, e.predicate, e.obj_entity_id,
                   a.source_node_id,
                   o.canonical_name AS obj_name
            FROM kg_edges e
            JOIN kg_assertions a ON e.assertion_id = a.assertion_id
            JOIN kg_entities o ON e.obj_entity_id = o.entity_id
            WHERE e.subj_entity_id = ? AND e.predicate = ?
              AND a.status = 'active'
            """,
            (entity_id, predicate),
        )
        return [
            {'obj_name': row[4], 'source_node_id': row[3]}
            for row in cursor.fetchall()
        ]
    except sqlite3.OperationalError:
        return []


# ---------------------------------------------------------------------------
# format_kg_context
# ---------------------------------------------------------------------------

def format_kg_context(
    facts: list[EdgeFact],
    derived_facts: list[DerivedFact],
    budget_tokens: int = 500,
) -> str:
    """Format facts into a context string, respecting token budget.

    Token estimate: len(text) // 4.
    Drops lowest-ranked facts first if over budget.
    """
    if not facts and not derived_facts:
        return ""

    lines: list[tuple[float, str]] = []

    for f in facts:
        line = f"- {f.subj_name} {f.predicate} {f.obj_name} [node:{f.source_node_id}]"
        lines.append((f.rank_score, line))

    for d in derived_facts:
        nodes_str = " + ".join(f"node:{nid}" for nid in d.source_node_ids)
        line = f"- (derived:{d.rule}) {d.subj_name} {d.predicate} {d.obj_name} [from {nodes_str}]"
        # Derived facts get a slight rank boost so they're kept if possible
        lines.append((999.0, line))

    # Sort by rank descending
    lines.sort(key=lambda x: x[0], reverse=True)

    result_lines: list[str] = []
    tokens_used = 0
    header = "Known facts about entities mentioned:"
    tokens_used += len(header) // 4

    for _rank, line in lines:
        line_tokens = len(line) // 4
        if tokens_used + line_tokens > budget_tokens:
            break
        result_lines.append(line)
        tokens_used += line_tokens

    if not result_lines:
        return ""

    return header + "\n" + "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

# Module-level singleton
_mention_dict = MentionDictionary()


def get_kg_context(
    user_text: str,
    conn: sqlite3.Connection,
) -> Optional[KGContextResult]:
    """Main entry point: detect mentions, retrieve edges, apply closure, format.

    Returns None if no KG tables or no matches.
    """
    # Check KG tables exist
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
        ).fetchone()
        if row is None:
            debug_print("KG tables not found, skipping", category='kg')
            return None
    except sqlite3.OperationalError:
        return None

    max_entities = config.get('kg_max_entities', 5)
    max_edges = config.get('kg_max_edges', 5)
    max_derived = config.get('kg_max_derived', 3)
    budget = config.get('kg_budget', 500)

    # Rebuild mention dictionary if needed
    cache_status = _mention_dict.rebuild(conn)

    # Detect mentions
    matches = _mention_dict.detect_mentions(user_text, max_entities=max_entities)
    if not matches:
        debug_print("No entity mentions detected", category='kg')
        return None

    debug_print(
        f"Matched {len(matches)} entities: "
        + ", ".join(f"{form}(id={eid})" for eid, form, _ in matches),
        category='kg',
    )

    # Retrieve edges for each matched entity
    matched_ids = [eid for eid, _, _ in matches]
    co_mentioned = set(matched_ids)
    all_facts: list[EdgeFact] = []

    for entity_id, _form, _weight in matches:
        facts = retrieve_neighborhood(
            entity_id, conn,
            user_text=user_text,
            co_mentioned_ids=co_mentioned - {entity_id},
            max_edges=max_edges,
        )
        all_facts.extend(facts)

    # Deduplicate across entities (same triple)
    seen_triples: set[tuple[str, str, str]] = set()
    deduped: list[EdgeFact] = []
    for f in all_facts:
        triple = (f.subj_name, f.predicate, f.obj_name)
        if triple not in seen_triples:
            seen_triples.add(triple)
            deduped.append(f)
    all_facts = deduped

    # Apply closure rules, then filter out derived triples already in neighborhood
    derived = apply_closure_rules(matched_ids, all_facts, conn, max_derived=max_derived)
    derived = [d for d in derived
               if (d.subj_name, d.predicate, d.obj_name) not in seen_triples]

    debug_print(
        f"Retrieved {len(all_facts)} edges, {len(derived)} derived facts",
        category='kg',
    )

    # Format
    text = format_kg_context(all_facts, derived, budget_tokens=budget)
    if not text:
        return None

    budget_used = len(text) // 4

    return KGContextResult(
        text=text,
        matched_entities=[
            {'entity_id': eid, 'surface_form': form, 'weight': w}
            for eid, form, w in matches
        ],
        edge_count=len(all_facts),
        derived_count=len(derived),
        budget_used=budget_used,
        budget_total=budget,
        cache_status=cache_status,
        edges=all_facts,
        derived=derived,
    )
