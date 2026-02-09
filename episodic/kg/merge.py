"""Entity merge operation for the knowledge graph.

Merges two entities into one (survivor absorbs merged), rewriting all
edges, aliases, and mentions in a single atomic transaction.
"""

import json
import sqlite3
import time
from typing import Optional


def merge_entities(
    survivor_id: int,
    merged_id: int,
    reason: str,
    conn: sqlite3.Connection,
    created_by_node_id: Optional[int] = None,
) -> dict:
    """Merge merged_id into survivor_id. Returns counts dict.

    The entire merge is one SQLite transaction. If any step fails,
    the transaction is rolled back completely.
    """
    # --- Input validation ---
    if survivor_id == merged_id:
        raise ValueError("Cannot merge an entity into itself")

    for eid, label in [(survivor_id, "Survivor"), (merged_id, "Merged")]:
        row = conn.execute(
            "SELECT entity_id, merged_into_entity_id FROM kg_entities "
            "WHERE entity_id = ?", (eid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"{label} entity {eid} does not exist")
        if row[1] is not None:
            raise ValueError(
                f"{label} entity {eid} is already tombstoned "
                f"(merged into {row[1]})"
            )

    counts = {
        'moved_edges': 0,
        'dropped_edges': 0,
        'moved_aliases': 0,
        'dropped_aliases': 0,
        'moved_mentions': 0,
    }

    # --- Step 1: Rewrite kg_edges ---
    edges = conn.execute(
        "SELECT edge_id, subj_entity_id, predicate, obj_entity_id, assertion_id "
        "FROM kg_edges WHERE subj_entity_id = ? OR obj_entity_id = ?",
        (merged_id, merged_id),
    ).fetchall()

    for edge_id, subj, pred, obj, assertion_id in edges:
        new_subj = survivor_id if subj == merged_id else subj
        new_obj = survivor_id if obj == merged_id else obj

        # Check for conflict with existing triple
        conflict = conn.execute(
            "SELECT edge_id, assertion_id FROM kg_edges "
            "WHERE subj_entity_id = ? AND predicate = ? AND obj_entity_id = ?",
            (new_subj, pred, new_obj),
        ).fetchone()

        if conflict and conflict[0] != edge_id:
            # Conflict: keep the edge with the higher source_node_id
            existing_eid, existing_aid = conflict
            cur_node = conn.execute(
                "SELECT source_node_id FROM kg_assertions "
                "WHERE assertion_id = ?", (assertion_id,)
            ).fetchone()
            existing_node = conn.execute(
                "SELECT source_node_id FROM kg_assertions "
                "WHERE assertion_id = ?", (existing_aid,)
            ).fetchone()

            cur_nid = cur_node[0] if cur_node else 0
            exist_nid = existing_node[0] if existing_node else 0

            if cur_nid > exist_nid:
                # Current edge is newer — delete existing, update current
                conn.execute(
                    "DELETE FROM kg_edges WHERE edge_id = ?", (existing_eid,)
                )
                conn.execute(
                    "UPDATE kg_edges SET subj_entity_id = ?, obj_entity_id = ? "
                    "WHERE edge_id = ?", (new_subj, new_obj, edge_id),
                )
            else:
                # Existing edge is newer — delete current
                conn.execute(
                    "DELETE FROM kg_edges WHERE edge_id = ?", (edge_id,)
                )
            counts['dropped_edges'] += 1
        else:
            # No conflict — update in place
            conn.execute(
                "UPDATE kg_edges SET subj_entity_id = ?, obj_entity_id = ? "
                "WHERE edge_id = ?", (new_subj, new_obj, edge_id),
            )
            counts['moved_edges'] += 1

    # --- Step 2: Rewrite kg_entity_aliases ---
    aliases = conn.execute(
        "SELECT alias_id, alias, source_node_id FROM kg_entity_aliases "
        "WHERE entity_id = ?", (merged_id,),
    ).fetchall()

    for alias_id, alias_text, src_node_id in aliases:
        conflict = conn.execute(
            "SELECT alias_id, source_node_id FROM kg_entity_aliases "
            "WHERE entity_id = ? AND alias = ?",
            (survivor_id, alias_text),
        ).fetchone()

        if conflict:
            # Keep the one with lower source_node_id (earliest provenance)
            existing_aid, existing_src = conflict
            if src_node_id < existing_src:
                conn.execute(
                    "DELETE FROM kg_entity_aliases WHERE alias_id = ?",
                    (existing_aid,),
                )
                conn.execute(
                    "UPDATE kg_entity_aliases SET entity_id = ? "
                    "WHERE alias_id = ?", (survivor_id, alias_id),
                )
            else:
                conn.execute(
                    "DELETE FROM kg_entity_aliases WHERE alias_id = ?",
                    (alias_id,),
                )
            counts['dropped_aliases'] += 1
        else:
            conn.execute(
                "UPDATE kg_entity_aliases SET entity_id = ? "
                "WHERE alias_id = ?", (survivor_id, alias_id),
            )
            counts['moved_aliases'] += 1

    # --- Step 3: Rewrite kg_mentions ---
    mentions = conn.execute(
        "SELECT mention_id, node_id, span_start, span_end "
        "FROM kg_mentions WHERE entity_id = ?", (merged_id,),
    ).fetchall()

    for mention_id, node_id, span_s, span_e in mentions:
        conflict = conn.execute(
            "SELECT mention_id FROM kg_mentions "
            "WHERE node_id = ? AND span_start = ? AND span_end = ? "
            "AND entity_id = ?",
            (node_id, span_s, span_e, survivor_id),
        ).fetchone()

        if conflict:
            conn.execute(
                "DELETE FROM kg_mentions WHERE mention_id = ?", (mention_id,)
            )
        else:
            conn.execute(
                "UPDATE kg_mentions SET entity_id = ? WHERE mention_id = ?",
                (survivor_id, mention_id),
            )
            counts['moved_mentions'] += 1

    # --- Step 4: Tombstone the merged entity ---
    now = time.time()
    conn.execute(
        "UPDATE kg_entities SET merged_into_entity_id = ?, merged_at = ?, "
        "merged_reason = ? WHERE entity_id = ?",
        (survivor_id, now, reason, merged_id),
    )

    # --- Step 5: Write merge log ---
    conn.execute(
        "INSERT INTO kg_merges "
        "(survivor_id, merged_id, created_at, created_by_node_id, reason, counts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (survivor_id, merged_id, now, created_by_node_id, reason,
         json.dumps(counts)),
    )

    # --- Step 6: Invalidate MentionDictionary cache ---
    conn.execute(
        "INSERT OR REPLACE INTO kg_state (key, value) "
        "VALUES ('merge_epoch', ?)",
        (str(now),),
    )

    conn.commit()

    return {
        'survivor_id': survivor_id,
        'merged_id': merged_id,
        **counts,
    }
