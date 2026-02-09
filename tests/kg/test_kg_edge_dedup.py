"""Tests for KG edge deduplication.

T1: Duplicate triple on insert → only one row, latest assertion wins
T2: Migration dedup of existing duplicates
T3: Closure dedup — two paths yielding same derived triple → one fact
"""

import sqlite3
import time

import pytest

from episodic.kg.schema import ensure_kg_schema, _migrate_edge_dedup
from episodic.kg.context_source import (
    apply_closure_rules,
    DerivedFact,
    EdgeFact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_entities(conn):
    """Seed minimal entities for edge tests. Returns entity_ids dict."""
    ensure_kg_schema(conn)
    user_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('person', NULL, 'Emma', 1, ?)",
        (time.time(),),
    )
    emma_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('org', NULL, 'MIT', 2, ?)",
        (time.time(),),
    )
    mit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Need nodes table for applicator
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    for nid in range(1, 4):
        conn.execute("INSERT OR IGNORE INTO nodes VALUES (?, ?, 'user')",
                     (nid, f"node {nid}"))

    conn.commit()
    return {'<user>': user_id, 'Emma': emma_id, 'MIT': mit_id}


def _insert_assertion(conn, node_id, span_start, span_end):
    """Insert an assertion and return its ID."""
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
        "asserted_by, polarity, certainty, status, tags) "
        "VALUES (?, ?, ?, 'user', 'affirm', 'explicit', 'active', '[]')",
        (node_id, span_start, span_end),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# T1: test_edge_dedup_on_insert
# ---------------------------------------------------------------------------

def test_edge_dedup_on_insert():
    """Two inserts of the same triple from different nodes → one row, latest assertion."""
    conn = sqlite3.connect(':memory:')
    eids = _seed_entities(conn)

    # First assertion from node 1
    a1 = _insert_assertion(conn, 1, 0, 10)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'located_at', ?, ?) "
        "ON CONFLICT(subj_entity_id, predicate, obj_entity_id) "
        "DO UPDATE SET assertion_id = excluded.assertion_id",
        (eids['Emma'], eids['MIT'], a1),
    )
    conn.commit()

    # Second assertion from node 2 (same triple)
    a2 = _insert_assertion(conn, 2, 0, 20)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'located_at', ?, ?) "
        "ON CONFLICT(subj_entity_id, predicate, obj_entity_id) "
        "DO UPDATE SET assertion_id = excluded.assertion_id",
        (eids['Emma'], eids['MIT'], a2),
    )
    conn.commit()

    # Only one edge row
    rows = conn.execute("SELECT * FROM kg_edges").fetchall()
    assert len(rows) == 1, f"Expected 1 edge, got {len(rows)}"

    # Assertion is the latest (a2)
    edge = rows[0]
    assert edge[4] == a2, f"Expected assertion_id={a2}, got {edge[4]}"

    conn.close()


# ---------------------------------------------------------------------------
# T2: test_edge_dedup_migration
# ---------------------------------------------------------------------------

def test_edge_dedup_migration():
    """Insert duplicates manually (bypassing unique), run migration, verify deduped."""
    conn = sqlite3.connect(':memory:')

    # Create the OLD schema (with 4-column unique, no 3-column index)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kg_entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            canonical_key TEXT,
            canonical_name TEXT NOT NULL,
            created_node_id INTEGER NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS kg_assertions (
            assertion_id INTEGER PRIMARY KEY,
            source_node_id INTEGER NOT NULL,
            span_start INTEGER NOT NULL,
            span_end INTEGER NOT NULL,
            asserted_by TEXT NOT NULL,
            polarity TEXT NOT NULL,
            certainty TEXT NOT NULL,
            status TEXT NOT NULL,
            tags TEXT,
            UNIQUE(source_node_id, span_start, span_end)
        );
        CREATE TABLE IF NOT EXISTS kg_edges (
            edge_id INTEGER PRIMARY KEY,
            subj_entity_id INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            obj_entity_id INTEGER NOT NULL,
            assertion_id INTEGER NOT NULL,
            UNIQUE(subj_entity_id, predicate, obj_entity_id, assertion_id)
        );
        CREATE TABLE IF NOT EXISTS kg_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO kg_state (key, value) VALUES ('high_water_mark', '0');
        INSERT INTO kg_state (key, value) VALUES ('schema_version', 'kg_v1');
    """)

    # Insert entities
    conn.execute(
        "INSERT INTO kg_entities VALUES (1, 'person', NULL, 'Emma', 1, ?)",
        (time.time(),),
    )
    conn.execute(
        "INSERT INTO kg_entities VALUES (2, 'org', NULL, 'MIT', 2, ?)",
        (time.time(),),
    )

    # Insert two assertions
    conn.execute(
        "INSERT INTO kg_assertions VALUES (1, 1, 0, 10, 'user', 'affirm', "
        "'explicit', 'active', '[]')"
    )
    conn.execute(
        "INSERT INTO kg_assertions VALUES (2, 2, 0, 20, 'user', 'affirm', "
        "'explicit', 'active', '[]')"
    )

    # Insert two edges with same triple but different assertions (old schema allows this)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'located_at', 2, 1)"
    )
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'located_at', 2, 2)"
    )
    conn.commit()

    # Verify 2 rows exist pre-migration
    count = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    assert count == 2

    # Run migration
    _migrate_edge_dedup(conn)
    conn.commit()

    # Verify 1 row exists post-migration
    count = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    assert count == 1, f"Expected 1 edge after migration, got {count}"

    # The kept row should be the one with the highest rowid (most recent)
    row = conn.execute("SELECT assertion_id FROM kg_edges").fetchone()
    assert row[0] == 2, f"Expected assertion_id=2 (latest), got {row[0]}"

    # Verify unique index exists
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='uq_kg_edges_triple'"
    ).fetchone()
    assert idx is not None, "uq_kg_edges_triple index should exist"

    conn.close()


# ---------------------------------------------------------------------------
# T3: test_closure_dedup
# ---------------------------------------------------------------------------

def test_closure_dedup():
    """Two closure paths yielding same derived triple → only one derived fact."""
    conn = sqlite3.connect(':memory:')
    eids = _seed_entities(conn)

    # Create two related_to edges from user to Emma (different source nodes)
    a1 = _insert_assertion(conn, 1, 0, 10)
    a2 = _insert_assertion(conn, 2, 0, 20)

    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'related_to', ?, ?)",
        (eids['<user>'], eids['Emma'], a1),
    )

    # Emma located_at MIT
    a3 = _insert_assertion(conn, 1, 15, 25)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'located_at', ?, ?)",
        (eids['Emma'], eids['MIT'], a3),
    )
    conn.commit()

    # Build edges list with two related_to entries (simulating pre-dedup input)
    edges = [
        EdgeFact(
            subj_name='<user>', predicate='related_to', obj_name='Emma',
            source_node_id=1, rank_score=1.0, tags=[],
        ),
        EdgeFact(
            subj_name='<user>', predicate='related_to', obj_name='Emma',
            source_node_id=2, rank_score=0.5, tags=[],
        ),
    ]

    derived = apply_closure_rules(
        matched_entity_ids=[eids['Emma']],
        edges=edges,
        conn=conn,
        max_derived=5,
    )

    # Should only have ONE derived fact for Emma located_at MIT
    located_facts = [d for d in derived
                     if d.subj_name == 'Emma' and d.predicate == 'located_at']
    assert len(located_facts) == 1, (
        f"Expected 1 derived located_at fact, got {len(located_facts)}: {located_facts}"
    )

    conn.close()
