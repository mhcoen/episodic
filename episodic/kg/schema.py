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
    predicate TEXT NOT NULL,
    obj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    assertion_id INTEGER NOT NULL REFERENCES kg_assertions(assertion_id)
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

CREATE TABLE IF NOT EXISTS kg_promotions (
    promotion_id INTEGER PRIMARY KEY,
    assertion_id INTEGER NOT NULL REFERENCES kg_assertions(assertion_id),
    promoted_at REAL NOT NULL,
    promoted_by TEXT NOT NULL DEFAULT 'cli_user',
    source_origin TEXT
);
CREATE INDEX IF NOT EXISTS idx_kg_promotions_assertion
    ON kg_promotions(assertion_id);
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
        _migrate_edge_dedup(c)
        _migrate_predicate_check(c)
        _migrate_entity_merge(c)
        _migrate_assertion_asserted_by(c)
        _migrate_assertion_quarantine(c)
        c.commit()


def _migrate_edge_dedup(conn):
    """Migrate kg_edges from 4-column UNIQUE to 3-column UNIQUE.

    Idempotent: checks if uq_kg_edges_triple exists. If not, dedup
    existing rows (keep highest rowid per triple) and create the index.
    """
    # Check if the new unique index already exists
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='uq_kg_edges_triple'"
    ).fetchone()
    if row:
        return  # Already migrated

    # Delete duplicates, keeping the row with the highest rowid (most recent)
    conn.execute("""
        DELETE FROM kg_edges WHERE rowid NOT IN (
            SELECT MAX(rowid) FROM kg_edges
            GROUP BY subj_entity_id, predicate, obj_entity_id
        )
    """)

    # Drop old 4-column unique if it exists (SQLite can't ALTER INDEX,
    # but CREATE UNIQUE INDEX IF NOT EXISTS on the new columns will work
    # since we just deduped. The old UNIQUE constraint was inline in the
    # CREATE TABLE, so it persists as a nameless index — harmless.)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_kg_edges_triple
        ON kg_edges(subj_entity_id, predicate, obj_entity_id)
    """)


def _migrate_predicate_check(conn):
    """Remove hardcoded CHECK constraint on kg_edges.predicate.

    Idempotent: checks if the old CHECK exists in the table DDL. If so,
    recreates the table without it (validator enforces predicate validity).
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='kg_edges'"
    ).fetchone()
    if not row or 'CHECK(predicate IN' not in (row[0] or ''):
        return  # Already clean or table doesn't exist

    conn.executescript("""
        CREATE TABLE kg_edges_new (
            edge_id INTEGER PRIMARY KEY,
            subj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
            predicate TEXT NOT NULL,
            obj_entity_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
            assertion_id INTEGER NOT NULL REFERENCES kg_assertions(assertion_id)
        );
        INSERT INTO kg_edges_new SELECT * FROM kg_edges;
        DROP TABLE kg_edges;
        ALTER TABLE kg_edges_new RENAME TO kg_edges;
        CREATE INDEX IF NOT EXISTS idx_kg_edges_subj ON kg_edges(subj_entity_id);
        CREATE INDEX IF NOT EXISTS idx_kg_edges_obj ON kg_edges(obj_entity_id);
        CREATE INDEX IF NOT EXISTS idx_kg_edges_pred ON kg_edges(predicate);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_kg_edges_triple
            ON kg_edges(subj_entity_id, predicate, obj_entity_id);
    """)


def _migrate_entity_merge(conn):
    """Add merge tombstone columns to kg_entities and create kg_merges table.

    Idempotent: checks if merged_into_entity_id column already exists.
    """
    # Check if column already exists
    try:
        conn.execute("SELECT merged_into_entity_id FROM kg_entities LIMIT 0")
        has_col = True
    except sqlite3.OperationalError:
        has_col = False

    if not has_col:
        conn.execute(
            "ALTER TABLE kg_entities ADD COLUMN "
            "merged_into_entity_id INTEGER NULL REFERENCES kg_entities(entity_id)"
        )
        conn.execute(
            "ALTER TABLE kg_entities ADD COLUMN merged_at REAL NULL"
        )
        conn.execute(
            "ALTER TABLE kg_entities ADD COLUMN merged_reason TEXT NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_merged "
            "ON kg_entities(merged_into_entity_id) "
            "WHERE merged_into_entity_id IS NOT NULL"
        )

    # Create kg_merges table (append-only log)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kg_merges (
            merge_id INTEGER PRIMARY KEY,
            survivor_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
            merged_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
            created_at REAL NOT NULL,
            created_by_node_id INTEGER NULL,
            reason TEXT,
            counts TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kg_merges_survivor
            ON kg_merges(survivor_id);
        CREATE INDEX IF NOT EXISTS idx_kg_merges_merged
            ON kg_merges(merged_id);
    """)


def _migrate_assertion_asserted_by(conn):
    """Expand kg_assertions.asserted_by CHECK constraint.

    Original: CHECK(asserted_by IN ('user'))
    New: CHECK(asserted_by IN ('user','assistant','mcp_client','web','email','calendar','rag'))

    Uses table-rebuild pattern (same as _migrate_predicate_check).
    Idempotent: checks if old CHECK exists in table DDL.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='kg_assertions'"
    ).fetchone()
    if not row:
        return
    ddl = row[0] or ''
    # Only migrate if the old single-value CHECK is present
    if "asserted_by IN ('user')" not in ddl:
        return

    # Check for quarantine columns — if they exist we must preserve them
    has_quarantine = 'quarantined' in ddl
    quarantine_cols = (
        ", quarantined INTEGER NOT NULL DEFAULT 0, source_origin TEXT"
        if has_quarantine else ""
    )
    quarantine_select = (
        ", quarantined, source_origin" if has_quarantine else ""
    )

    conn.executescript(f"""
        CREATE TABLE kg_assertions_new (
            assertion_id INTEGER PRIMARY KEY,
            source_node_id INTEGER NOT NULL,
            span_start INTEGER NOT NULL,
            span_end INTEGER NOT NULL,
            asserted_by TEXT NOT NULL CHECK(asserted_by IN
                ('user','assistant','mcp_client','web','email','calendar','rag')),
            polarity TEXT NOT NULL CHECK(polarity IN ('affirm', 'negate')),
            certainty TEXT NOT NULL CHECK(certainty IN ('explicit', 'hedged')),
            status TEXT NOT NULL CHECK(status IN ('active')),
            tags TEXT{quarantine_cols},
            UNIQUE(source_node_id, span_start, span_end)
        );
        INSERT INTO kg_assertions_new
            SELECT assertion_id, source_node_id, span_start, span_end,
                   asserted_by, polarity, certainty, status, tags
                   {quarantine_select}
            FROM kg_assertions;
        DROP TABLE kg_assertions;
        ALTER TABLE kg_assertions_new RENAME TO kg_assertions;
        CREATE INDEX IF NOT EXISTS idx_kg_assertions_node
            ON kg_assertions(source_node_id);
    """)


def _migrate_assertion_quarantine(conn):
    """Add quarantine columns to kg_assertions.

    Adds: quarantined INTEGER NOT NULL DEFAULT 0, source_origin TEXT.
    Creates partial index for efficient quarantine queries.
    Idempotent: checks if column exists.
    """
    try:
        conn.execute("SELECT quarantined FROM kg_assertions LIMIT 0")
        return  # Already has column
    except sqlite3.OperationalError:
        pass

    conn.execute(
        "ALTER TABLE kg_assertions ADD COLUMN "
        "quarantined INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute(
        "ALTER TABLE kg_assertions ADD COLUMN source_origin TEXT"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kg_assertions_quarantined "
        "ON kg_assertions(quarantined) WHERE quarantined = 1"
    )


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
