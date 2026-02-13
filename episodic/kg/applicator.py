"""Transactional patch application to SQLite KG tables."""

import json
import time
import sqlite3

from .db_kg import _use_conn
from .validator import VALIDATOR_VERSION


def apply_patch(
    patch: dict,
    node_id: int,
    patch_json: str,
    patch_hash: str,
    model_id: str,
    extraction_time_ms: int,
    conn=None,
    quarantine: bool = False,
    source_origin: str = "",
) -> dict:
    """Apply a validated patch to the KG tables in a single transaction.

    Note: All span offsets (span_start, span_end) in the patch are relative
    to the *normalized* source text (smart quotes replaced with ASCII).
    The normalization is applied upstream in build_extraction_context() and
    validate_patch() before the patch reaches this function.

    Returns:
    {
        'applied': True,
        'entities_created': int,
        'entities_resolved': int,
        'assertions_created': int,
        'edges_created': int,
        'mentions_created': int,
        'aliases_created': int,
    }

    On failure, rolls back and raises.
    """
    with _use_conn(conn) as c:
        c.execute("PRAGMA foreign_keys=ON")

        counts = {
            'applied': True,
            'entities_created': 0,
            'entities_resolved': 0,
            'assertions_created': 0,
            'edges_created': 0,
            'mentions_created': 0,
            'aliases_created': 0,
        }

        try:
            # Step 1: Record the patch (optimistic applied=1)
            c.execute(
                "INSERT OR REPLACE INTO kg_patches "
                "(node_id, patch_json, patch_hash, validator_version, "
                "applied, rejection_reason, model_id, extraction_time_ms) "
                "VALUES (?, ?, ?, ?, 1, NULL, ?, ?)",
                (node_id, patch_json, patch_hash, VALIDATOR_VERSION,
                 model_id, extraction_time_ms)
            )

            # Step 2: Insert assertions
            assertion_id_map: dict[str, int] = {}
            q_val = 1 if quarantine else 0
            for a in patch.get('assertions', []):
                tags_json = json.dumps(a.get('tags', []))
                cursor = c.execute(
                    "INSERT INTO kg_assertions "
                    "(source_node_id, span_start, span_end, asserted_by, "
                    "polarity, certainty, status, tags, quarantined, "
                    "source_origin) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (node_id, a['span_start'], a['span_end'],
                     a['asserted_by'], a['polarity'], a['certainty'],
                     a['status'], tags_json, q_val, source_origin or None)
                )
                assertion_id_map[a['assertion_key']] = cursor.lastrowid
                counts['assertions_created'] += 1

            # Step 3: Resolve and insert entities
            entity_id_map: dict[str, int] = {}

            # Look up user:self entity_id
            row = c.execute(
                "SELECT entity_id FROM kg_entities "
                "WHERE canonical_key = 'user:self'"
            ).fetchone()
            user_self_id = row[0] if row else 1
            entity_id_map['user:self'] = user_self_id

            for e in patch.get('entities', []):
                ekey = e['entity_key']
                ckey = e.get('canonical_key')
                hint = e.get('resolution_hint')

                resolved_id = None

                # Case A: resolution_hint.kind == 'map_to_existing'
                if (hint and isinstance(hint, dict)
                        and hint.get('kind') == 'map_to_existing'):
                    resolved_id = hint.get('candidate_entity_id')
                    counts['entities_resolved'] += 1

                # Case B: canonical_key exists in DB
                elif ckey is not None:
                    existing = c.execute(
                        "SELECT entity_id FROM kg_entities "
                        "WHERE canonical_key = ?",
                        (ckey,)
                    ).fetchone()
                    if existing:
                        resolved_id = existing[0]
                        counts['entities_resolved'] += 1

                # Case C: new entity
                if resolved_id is None:
                    cursor = c.execute(
                        "INSERT INTO kg_entities "
                        "(entity_type, canonical_key, canonical_name, "
                        "created_node_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (e['entity_type'], ckey, e['canonical_name'],
                         node_id, time.time())
                    )
                    resolved_id = cursor.lastrowid
                    counts['entities_created'] += 1

                entity_id_map[ekey] = resolved_id

            # Helper to resolve entity references
            def resolve_entity_ref(ref: str) -> int | None:
                if ref == 'user:self':
                    return entity_id_map.get('user:self')
                if ref in entity_id_map:
                    return entity_id_map[ref]
                if isinstance(ref, str) and ref.startswith('db:'):
                    try:
                        return int(ref[3:])
                    except ValueError:
                        return None
                return None

            # Step 4: Insert aliases
            for alias in patch.get('aliases', []):
                eid = resolve_entity_ref(alias['entity_ref'])
                if eid is None:
                    continue
                try:
                    c.execute(
                        "INSERT OR IGNORE INTO kg_entity_aliases "
                        "(entity_id, alias, source_node_id, "
                        "span_start, span_end) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (eid, alias['alias_text'], node_id,
                         alias['span_start'], alias['span_end'])
                    )
                    counts['aliases_created'] += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate alias

            # Step 5: Insert mentions
            for m in patch.get('mentions', []):
                eref = m.get('entity_ref')
                eid = resolve_entity_ref(eref) if eref else None
                try:
                    c.execute(
                        "INSERT OR IGNORE INTO kg_mentions "
                        "(node_id, span_start, span_end, surface_text, "
                        "entity_id, confidence) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (node_id, m['span_start'], m['span_end'],
                         m['surface_text'], eid, m['confidence'])
                    )
                    counts['mentions_created'] += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate mention

            # Step 6: Insert edges
            for edge in patch.get('edges', []):
                subj_id = resolve_entity_ref(edge['subj_ref'])
                obj_id = resolve_entity_ref(edge['obj_ref'])
                a_id = assertion_id_map.get(edge['source_assertion'])
                if subj_id is None or obj_id is None or a_id is None:
                    continue
                try:
                    c.execute(
                        "INSERT INTO kg_edges "
                        "(subj_entity_id, predicate, obj_entity_id, "
                        "assertion_id) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(subj_entity_id, predicate, obj_entity_id) "
                        "DO UPDATE SET assertion_id = excluded.assertion_id",
                        (subj_id, edge['predicate'], obj_id, a_id)
                    )
                    counts['edges_created'] += 1
                except sqlite3.IntegrityError:
                    pass  # Shouldn't happen with ON CONFLICT

            # Step 7: Advance high-water mark
            c.execute(
                "UPDATE kg_state SET value = ? "
                "WHERE key = 'high_water_mark'",
                (str(node_id),)
            )

            # Step 8: Commit
            c.commit()

        except Exception:
            c.rollback()
            raise

        return counts


def record_rejected_patch(
    node_id: int,
    patch_json: str | None,
    patch_hash: str | None,
    rejection_reason: str,
    model_id: str,
    extraction_time_ms: int,
    conn=None,
):
    """Record a patch that failed validation or extraction.
    applied=0, rejection_reason set.
    """
    with _use_conn(conn) as c:
        c.execute(
            "INSERT OR REPLACE INTO kg_patches "
            "(node_id, patch_json, patch_hash, validator_version, "
            "applied, rejection_reason, model_id, extraction_time_ms) "
            "VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
            (node_id, patch_json or '', patch_hash or '',
             VALIDATOR_VERSION, rejection_reason,
             model_id, extraction_time_ms)
        )
        c.commit()
