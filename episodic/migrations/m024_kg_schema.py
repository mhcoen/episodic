"""Add knowledge graph schema tables.

Creates all KG tables (kg_entities, kg_edges, kg_assertions, kg_mentions,
kg_entity_aliases, kg_patches, kg_curations, kg_state, kg_skiplist) and
seeds the user:self entity and initial state rows.
"""

import sqlite3
from episodic.migrations import Migration


class AddKGSchema(Migration):
    """Add knowledge graph schema tables."""

    def __init__(self):
        super().__init__(
            version=24,
            description="Add knowledge graph schema tables"
        )

    def up(self, conn: sqlite3.Connection) -> None:
        from episodic.kg.schema import ensure_kg_schema
        # If old (seeder) schema exists, drop and recreate
        if self._has_stale_schema(conn):
            self.down(conn)
        ensure_kg_schema(conn)

    def _has_stale_schema(self, conn: sqlite3.Connection) -> bool:
        """Check if KG tables exist with outdated column names or constraints."""
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='kg_assertions'"
            ).fetchone()
            if row and 'source_node_id' not in row[0]:
                return True  # Old schema uses 'node_id' not 'source_node_id'
        except Exception:
            pass
        # Check if kg_edges is missing the 'has' predicate
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='kg_edges'"
            ).fetchone()
            if row and "'has'" not in row[0]:
                return True  # Missing 'has' predicate in CHECK constraint
        except Exception:
            pass
        return False

    def down(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        # Drop indices first
        for idx in [
            'uq_kg_entities_canonical_key', 'idx_kg_entities_type_name',
            'idx_kg_assertions_node', 'idx_kg_edges_subj', 'idx_kg_edges_obj',
            'idx_kg_edges_pred', 'idx_kg_patches_node',
        ]:
            cursor.execute(f"DROP INDEX IF EXISTS {idx}")
        # Drop tables in FK-safe order
        for table in [
            'kg_skiplist', 'kg_state', 'kg_curations', 'kg_patches',
            'kg_mentions', 'kg_edges', 'kg_assertions',
            'kg_entity_aliases', 'kg_entities',
        ]:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()


migration = AddKGSchema()
