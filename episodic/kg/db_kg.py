"""Data access layer for knowledge graph tables (read-only)."""

import contextlib
import json
import sqlite3
from typing import Optional


@contextlib.contextmanager
def _use_conn(conn=None):
    """Yield a usable connection. If *conn* is provided, yield it as-is.
    Otherwise open one via ``get_connection()`` (context manager).
    """
    if conn is not None:
        yield conn
    else:
        from episodic.db_connection import get_connection
        with get_connection() as c:
            yield c


def kg_tables_exist(conn=None) -> bool:
    """Check if kg_entities table exists."""
    try:
        with _use_conn(conn) as c:
            cursor = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
            )
            return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        return False


def get_all_entities(conn=None) -> list[dict]:
    """Fetch all entities from kg_entities."""
    if not kg_tables_exist(conn):
        return []
    try:
        with _use_conn(conn) as c:
            c.row_factory = sqlite3.Row
            cursor = c.execute(
                "SELECT entity_id, entity_type, canonical_key, canonical_name, "
                "created_node_id FROM kg_entities ORDER BY entity_id"
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []


def get_all_edges(conn=None) -> list[dict]:
    """Fetch all active edges with their assertion data."""
    if not kg_tables_exist(conn):
        return []
    try:
        with _use_conn(conn) as c:
            c.row_factory = sqlite3.Row
            cursor = c.execute(
                "SELECT e.edge_id, e.subj_entity_id, e.predicate, e.obj_entity_id, "
                "e.assertion_id, "
                "a.source_node_id AS node_id, a.span_start, a.span_end, a.polarity, a.tags, "
                "a.asserted_by "
                "FROM kg_edges e "
                "JOIN kg_assertions a ON e.assertion_id = a.assertion_id "
                "WHERE a.status = 'active' "
                "AND (a.quarantined = 0 OR a.quarantined IS NULL) "
                "ORDER BY e.edge_id"
            )
            rows = []
            for row in cursor.fetchall():
                d = dict(row)
                if d.get('tags'):
                    try:
                        d['tags'] = json.loads(d['tags'])
                    except (json.JSONDecodeError, TypeError):
                        d['tags'] = []
                else:
                    d['tags'] = []
                rows.append(d)
            return rows
    except sqlite3.OperationalError:
        return []


def get_entity_aliases(entity_id: int, conn=None) -> list[str]:
    """Fetch aliases for an entity, including curations."""
    if not kg_tables_exist(conn):
        return []
    try:
        with _use_conn(conn) as c:
            # Check if auxiliary tables exist
            has_aliases = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entity_aliases'"
            ).fetchone() is not None

            has_curations = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_curations'"
            ).fetchone() is not None

            aliases = set()

            if has_aliases:
                for row in c.execute(
                    "SELECT alias FROM kg_entity_aliases WHERE entity_id = ?",
                    (entity_id,),
                ).fetchall():
                    aliases.add(row[0])

            if has_curations:
                for row in c.execute(
                    "SELECT value FROM kg_curations "
                    "WHERE entity_id = ? AND curation_type = 'alias_add'",
                    (entity_id,),
                ).fetchall():
                    aliases.add(row[0])

                for row in c.execute(
                    "SELECT value FROM kg_curations "
                    "WHERE entity_id = ? AND curation_type = 'alias_remove'",
                    (entity_id,),
                ).fetchall():
                    aliases.discard(row[0])

            return sorted(aliases)
    except sqlite3.OperationalError:
        return []


def get_entity_degree(entity_id: int, conn=None) -> int:
    """Count edges where entity is subject OR object."""
    if not kg_tables_exist(conn):
        return 0
    try:
        with _use_conn(conn) as c:
            row = c.execute(
                "SELECT COUNT(*) FROM kg_edges "
                "WHERE subj_entity_id = ? OR obj_entity_id = ?",
                (entity_id, entity_id),
            ).fetchone()
            return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def get_assertion_span_text(
    node_id: int, span_start: int, span_end: int, conn=None
) -> Optional[str]:
    """Fetch the span text from the nodes table for the given node_id.

    The KG system uses integer node references. Tries ``node_id`` column
    first (test fixtures), then falls back to ``rowid`` (live DB where
    the primary key is a TEXT UUID).
    """
    try:
        with _use_conn(conn) as c:
            try:
                cursor = c.execute(
                    "SELECT content FROM nodes WHERE node_id = ?", (node_id,)
                )
            except sqlite3.OperationalError:
                cursor = c.execute(
                    "SELECT content FROM nodes WHERE rowid = ?", (node_id,)
                )
            row = cursor.fetchone()
            if row is None:
                return None
            content = row[0] if isinstance(row, (tuple, list)) else row['content']
            if content is None:
                return None
            if span_start < 0 or span_end > len(content) or span_start >= span_end:
                return None
            return content[span_start:span_end]
    except sqlite3.OperationalError:
        return None


def get_all_mentions(conn=None) -> list[dict]:
    """Fetch all entity mentions."""
    if not kg_tables_exist(conn):
        return []
    try:
        with _use_conn(conn) as c:
            if c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_mentions'"
            ).fetchone() is None:
                return []

            c.row_factory = sqlite3.Row
            cursor = c.execute(
                "SELECT mention_id, node_id, span_start, span_end, surface_text, "
                "entity_id, confidence "
                "FROM kg_mentions ORDER BY node_id, span_start"
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []


def get_node_id_range(conn=None) -> tuple[int, int]:
    """Get the min and max node_id from kg_assertions."""
    if not kg_tables_exist(conn):
        return (0, 0)
    try:
        with _use_conn(conn) as c:
            row = c.execute(
                "SELECT MIN(source_node_id), MAX(source_node_id) FROM kg_assertions"
            ).fetchone()
            if row is None or row[0] is None:
                return (0, 0)
            return (row[0], row[1])
    except sqlite3.OperationalError:
        return (0, 0)
