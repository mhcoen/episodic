"""DDL for KG tables and schema management."""

import contextlib
import sqlite3
import time


KG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('person', 'artifact', 'topic', 'org')),
    canonical_key TEXT,
    canonical_name TEXT NOT NULL,
    created_node_id INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_kg_entities_canonical_key
    ON kg_entities(canonical_key) WHERE canonical_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_kg_entities_type_name
    ON kg_entities(entity_type, canonical_name);

CREATE TABLE IF NOT EXISTS kg_entity_aliases (
    alias_id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    alias TEXT NOT NULL,
    source_node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    UNIQUE(entity_id, alias)
);

CREATE TABLE IF NOT EXISTS kg_assertions (
    assertion_id INTEGER PRIMARY KEY,
    source_node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    asserted_by TEXT NOT NULL CHECK(asserted_by IN ('user')),
    polarity TEXT NOT NULL CHECK(polarity IN ('affirm', 'negate')),
    certainty TEXT NOT NULL CHECK(certainty IN ('explicit', 'hedged')),
    status TEXT NOT NULL CHECK(status IN ('active')),
    tags TEXT,
    UNIQUE(source_node_id, span_start, span_end)
);
CREATE INDEX IF NOT EXISTS idx_kg_assertions_node ON kg_assertions(source_node_id);

CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id INTEGER PRIMARY KEY,
    subj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    predicate TEXT NOT NULL CHECK(predicate IN ('uses', 'wants', 'prefers', 'role', 'has', 'located_at', 'part_of', 'related_to', 'is_a', 'powered_by')),
    obj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    assertion_id INTEGER NOT NULL REFERENCES kg_assertions(assertion_id),
    UNIQUE(subj_entity_id, predicate, obj_entity_id, assertion_id)
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_subj ON kg_edges(subj_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_obj ON kg_edges(obj_entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_pred ON kg_edges(predicate);

CREATE TABLE IF NOT EXISTS kg_mentions (
    mention_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    surface_text TEXT NOT NULL,
    entity_id INTEGER REFERENCES kg_entities(entity_id),
    confidence REAL NOT NULL,
    UNIQUE(node_id, span_start, span_end)
);

CREATE TABLE IF NOT EXISTS kg_patches (
    patch_id INTEGER PRIMARY KEY,
    node_id INTEGER NOT NULL UNIQUE,
    patch_json TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    validator_version TEXT NOT NULL,
    applied INTEGER NOT NULL CHECK(applied IN (0, 1)),
    rejection_reason TEXT,
    model_id TEXT,
    extraction_time_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_kg_patches_node ON kg_patches(node_id);

CREATE TABLE IF NOT EXISTS kg_curations (
    curation_id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    curation_type TEXT NOT NULL CHECK(curation_type IN ('alias_add', 'alias_remove')),
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_skiplist (
    node_id INTEGER PRIMARY KEY,
    reason TEXT,
    created_at REAL NOT NULL
);
"""

SEED_USER_SELF_SQL = """
INSERT OR IGNORE INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at)
VALUES ('person', 'user:self', '<user>', 0, ?);
"""

SEED_STATE_SQL = """
INSERT OR IGNORE INTO kg_state (key, value) VALUES ('high_water_mark', '0');
INSERT OR IGNORE INTO kg_state (key, value) VALUES ('schema_version', 'kg_v1');
"""


@contextlib.contextmanager
def _use_conn(conn=None):
    """Yield a usable connection."""
    if conn is not None:
        yield conn
    else:
        from episodic.db_connection import get_connection
        with get_connection() as c:
            yield c


def ensure_kg_schema(conn=None):
    """Create all KG tables if they don't exist. Seed user:self and state rows.

    Idempotent — safe to call on every batch run.
    """
    with _use_conn(conn) as c:
        c.execute("PRAGMA foreign_keys=ON")
        c.executescript(KG_SCHEMA_SQL)
        c.execute(SEED_USER_SELF_SQL, (time.time(),))
        c.executescript(SEED_STATE_SQL)
        c.commit()


def migrate_kg_schema(conn=None):
    """For /migrate integration. Check current schema_version in kg_state,
    apply any pending migrations. Phase 0 has only one version (kg_v1),
    so this is a no-op after initial creation.
    """
    with _use_conn(conn) as c:
        try:
            row = c.execute(
                "SELECT value FROM kg_state WHERE key = 'schema_version'"
            ).fetchone()
            if row and row[0] == 'kg_v1':
                return  # Already at current version
        except Exception:
            pass
        ensure_kg_schema(c)
