"""System prompt and input formatter for KG extraction."""

import json
import sqlite3
from typing import Optional

from .db_kg import _use_conn

# Smart quote → ASCII normalization replacements (order: longest first)
_NORMALIZE_PAIRS = [
    ('\u2026', '...'),  # horizontal ellipsis → three dots
    ('\u2014', '--'),   # em dash → double hyphen
    ('\u2013', '-'),    # en dash → hyphen
    ('\u201c', '"'),    # left double curly quote
    ('\u201d', '"'),    # right double curly quote
    ('\u2018', "'"),    # left single curly quote
    ('\u2019', "'"),    # right single curly quote
]


def normalize_text(text: str) -> str:
    """Replace Unicode smart quotes and typographic chars with ASCII equivalents.

    This ensures the LLM sees simple ASCII characters for span offset
    computation, avoiding mismatches caused by multi-byte or unexpected
    Unicode characters in the source text.
    """
    for old, new in _NORMALIZE_PAIRS:
        text = text.replace(old, new)
    return text


from episodic.kg._extraction_prompts import (  # noqa: F401  (re-exported)
    EXTRACTION_SYSTEM_PROMPT,
    RETRY_ADDENDUM,
)


def format_extraction_input(
    node_id: int,
    source_text: str,
    recent_context: list[str],
    entity_dictionary: list[dict],
    kg_neighborhood: list[dict] | None = None,
) -> str:
    """Format the user message content for the extraction model call.

    Returns a JSON string that becomes the user message content.
    """
    payload = {
        'node_id': node_id,
        'source_text': source_text,
        'recent_context': recent_context,
        'entity_dictionary': entity_dictionary,
        'kg_neighborhood': kg_neighborhood or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_extraction_context(
    node_id: int,
    lookback: int = 3,
    conn=None,
) -> Optional[dict]:
    """Assemble the inputs needed for extraction of a single node.

    Returns dict with keys: node_id, source_text, recent_context,
    entity_dictionary, kg_neighborhood.

    Returns None if the node's role is not 'user' (Phase 0: user turns only).

    The node_id here is the rowid from the nodes table.
    """
    with _use_conn(conn) as c:
        # Step 1: Fetch the node's content and role
        try:
            row = c.execute(
                "SELECT id, content, role FROM nodes WHERE rowid = ?",
                (node_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            row = c.execute(
                "SELECT id, content, role FROM nodes WHERE node_id = ?",
                (node_id,)
            ).fetchone()

        if row is None:
            return None

        node_uuid = row[0]
        source_text = normalize_text(row[1]) if row[1] else row[1]
        role = row[2]

        if role != 'user':
            return None

        if not source_text or not source_text.strip():
            return None

        # Step 2: Fetch preceding turns for recent_context
        recent_context = []
        try:
            rows = c.execute(
                "SELECT role, content FROM nodes WHERE rowid < ? "
                "AND role IN ('user', 'assistant') "
                "ORDER BY rowid DESC LIMIT ?",
                (node_id, lookback)
            ).fetchall()
        except sqlite3.OperationalError:
            rows = c.execute(
                "SELECT role, content FROM nodes WHERE node_id < ? "
                "AND role IN ('user', 'assistant') "
                "ORDER BY node_id DESC LIMIT ?",
                (node_id, lookback)
            ).fetchall()

        for r in reversed(rows):
            ctx_role, ctx_content = r[0], r[1]
            if ctx_content:
                recent_context.append(f"{ctx_role}: {ctx_content}")

        # Step 3: Determine topic scope for this node
        topic_entity_ids = set()
        try:
            # Find which topic contains this node (via topic_nodes table)
            topic_row = c.execute(
                "SELECT topic_start_node_id FROM topic_nodes "
                "WHERE node_id = ? LIMIT 1",
                (node_uuid,)
            ).fetchone()

            if topic_row:
                topic_start = topic_row[0]
                # Get all rowids in this topic
                topic_node_rows = c.execute(
                    "SELECT turn_idx FROM topic_nodes "
                    "WHERE topic_start_node_id = ?",
                    (topic_start,)
                ).fetchall()
                topic_rowids = {r[0] for r in topic_node_rows}

                # Get entity_ids created within this topic's nodes
                if topic_rowids:
                    placeholders = ','.join('?' * len(topic_rowids))
                    ent_rows = c.execute(
                        f"SELECT entity_id FROM kg_entities "
                        f"WHERE created_node_id IN ({placeholders})",
                        list(topic_rowids)
                    ).fetchall()
                    topic_entity_ids = {r[0] for r in ent_rows}
        except sqlite3.OperationalError:
            pass  # topic_nodes may not exist

        # Step 4: Build entity dictionary
        entity_dictionary = _build_entity_dictionary(
            topic_entity_ids, c
        )

        # Step 5: Build KG neighborhood for recently mentioned entities
        kg_neighborhood = _build_kg_neighborhood(
            source_text, recent_context, entity_dictionary, c
        )

        return {
            'node_id': node_id,
            'source_text': source_text,
            'recent_context': recent_context,
            'entity_dictionary': entity_dictionary,
            'kg_neighborhood': kg_neighborhood,
        }


def _build_entity_dictionary(
    topic_entity_ids: set[int],
    conn: sqlite3.Connection,
) -> list[dict]:
    """Build the entity dictionary: all entities in topic scope plus all
    entities with non-null canonical_key (global scope)."""
    try:
        c = conn
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT entity_id, entity_type, canonical_name, canonical_key "
            "FROM kg_entities ORDER BY entity_id"
        ).fetchall()

        result = []
        seen = set()
        for row in rows:
            eid = row['entity_id']
            # Include if in topic scope or has canonical_key (global)
            if eid not in topic_entity_ids and row['canonical_key'] is None:
                continue
            if eid in seen:
                continue
            seen.add(eid)

            aliases = []
            try:
                alias_rows = c.execute(
                    "SELECT alias FROM kg_entity_aliases WHERE entity_id = ?",
                    (eid,)
                ).fetchall()
                aliases = [a[0] for a in alias_rows]
            except sqlite3.OperationalError:
                pass

            result.append({
                'entity_id': eid,
                'entity_type': row['entity_type'],
                'canonical_name': row['canonical_name'],
                'canonical_key': row['canonical_key'],
                'aliases': aliases,
            })

        return result
    except sqlite3.OperationalError:
        return []


def _build_kg_neighborhood(
    source_text: str,
    recent_context: list[str],
    entity_dictionary: list[dict],
    conn: sqlite3.Connection,
) -> list[dict]:
    """Build KG neighborhood for entities mentioned in source or context."""
    # Combine source + context for entity mention detection
    combined = source_text.lower()
    for ctx in recent_context:
        combined += ' ' + ctx.lower()

    # Find entities mentioned by name or alias
    mentioned_eids = []
    for ent in entity_dictionary:
        names = [ent['canonical_name'].lower()]
        names.extend(a.lower() for a in ent.get('aliases', []))
        if any(name in combined for name in names):
            mentioned_eids.append(ent['entity_id'])

    if not mentioned_eids:
        return []

    # Fetch edges for mentioned entities (limit 20)
    try:
        placeholders = ','.join('?' * len(mentioned_eids))
        rows = conn.execute(
            f"SELECT e.subj_entity_id, e.predicate, e.obj_entity_id, "
            f"a.polarity, a.tags "
            f"FROM kg_edges e "
            f"JOIN kg_assertions a ON e.assertion_id = a.assertion_id "
            f"WHERE (e.subj_entity_id IN ({placeholders}) "
            f"OR e.obj_entity_id IN ({placeholders})) "
            f"AND a.status = 'active' "
            f"AND (a.quarantined = 0 OR a.quarantined IS NULL) "
            f"LIMIT 20",
            mentioned_eids + mentioned_eids
        ).fetchall()

        # Build entity_id -> name map
        eid_to_name = {e['entity_id']: e['canonical_name']
                       for e in entity_dictionary}

        result = []
        for row in rows:
            subj_name = eid_to_name.get(row[0], f'entity_{row[0]}')
            obj_name = eid_to_name.get(row[2], f'entity_{row[2]}')
            result.append({
                'subject': subj_name,
                'predicate': row[1],
                'object': obj_name,
                'polarity': row[3],
            })
        return result
    except sqlite3.OperationalError:
        return []
