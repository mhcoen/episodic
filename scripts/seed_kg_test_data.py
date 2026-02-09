#!/usr/bin/env python3
"""Seed the knowledge graph tables with plausible test data from real conversations.

Usage:
    python scripts/seed_kg_test_data.py           # Seed if empty
    python scripts/seed_kg_test_data.py --clear    # Drop and re-seed
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# KG Schema DDL — all 9 tables
# ---------------------------------------------------------------------------

KG_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    canonical_key TEXT,
    canonical_name TEXT NOT NULL,
    created_node_id INTEGER,
    created_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS kg_assertions (
    assertion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    asserted_by TEXT NOT NULL DEFAULT 'extractor',
    polarity TEXT NOT NULL DEFAULT 'affirm',
    certainty TEXT NOT NULL DEFAULT 'explicit',
    status TEXT NOT NULL DEFAULT 'active',
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subj_entity_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    obj_entity_id INTEGER NOT NULL,
    assertion_id INTEGER NOT NULL,
    FOREIGN KEY (subj_entity_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (obj_entity_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (assertion_id) REFERENCES kg_assertions(assertion_id)
);

CREATE TABLE IF NOT EXISTS kg_mentions (
    mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    surface_text TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (entity_id) REFERENCES kg_entities(entity_id)
);

CREATE TABLE IF NOT EXISTS kg_entity_aliases (
    entity_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES kg_entities(entity_id)
);

CREATE TABLE IF NOT EXISTS kg_patches (
    patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL,
    patch_data TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'applied',
    created_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS kg_curations (
    curation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    curation_type TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (entity_id) REFERENCES kg_entities(entity_id)
);

CREATE TABLE IF NOT EXISTS kg_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_skiplist (
    node_id INTEGER PRIMARY KEY
);
"""

KG_DROP_DDL = """
DROP TABLE IF EXISTS kg_skiplist;
DROP TABLE IF EXISTS kg_state;
DROP TABLE IF EXISTS kg_curations;
DROP TABLE IF EXISTS kg_patches;
DROP TABLE IF EXISTS kg_entity_aliases;
DROP TABLE IF EXISTS kg_mentions;
DROP TABLE IF EXISTS kg_edges;
DROP TABLE IF EXISTS kg_assertions;
DROP TABLE IF EXISTS kg_entities;
"""


def get_db_connection():
    """Connect to the live Episodic database."""
    db_path = os.environ.get("EPISODIC_DB_PATH") or os.path.expanduser(
        "~/.episodic/episodic.db"
    )
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    print(f"Connected to: {db_path}")
    return conn


def tables_exist(conn):
    """Check if KG tables already have data."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_entities'"
    )
    if cursor.fetchone() is None:
        return False
    cursor = conn.execute("SELECT COUNT(*) FROM kg_entities")
    return cursor.fetchone()[0] > 0


def create_tables(conn):
    """Create all KG schema tables."""
    conn.executescript(KG_SCHEMA_DDL)
    print("Created KG schema tables (9 tables)")


def drop_tables(conn):
    """Drop all KG tables."""
    conn.executescript(KG_DROP_DDL)
    print("Dropped all KG tables")


def find_span(content, term):
    """Find the position of a term in content (case-insensitive).

    Returns (start, end) or None.
    """
    m = re.search(re.escape(term), content, re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return None


def find_sentence_span(content, term):
    """Find a sentence-like span containing the term.

    Returns (start, end) covering a wider context.
    """
    m = re.search(re.escape(term), content, re.IGNORECASE)
    if not m:
        return None
    # Expand to approximate sentence boundaries
    start = m.start()
    end = m.end()
    # Walk back to sentence start
    while start > 0 and content[start - 1] not in ".!?\n":
        start -= 1
    if start > 0:
        start += 1  # skip the punctuation
    # Walk forward to sentence end
    while end < len(content) and content[end] not in ".!?\n":
        end += 1
    if end < len(content):
        end += 1  # include the punctuation
    return start, end


def scan_messages(conn):
    """Scan user messages for entity mentions and return useful spans."""
    cursor = conn.execute(
        "SELECT rowid, id, content FROM nodes "
        "WHERE role='user' AND length(content) > 20 "
        "ORDER BY rowid"
    )
    rows = cursor.fetchall()
    print(f"Scanned {len(rows)} user messages")
    return [(r[0], r[1], r[2]) for r in rows]


def seed_data(conn):
    """Insert plausible KG data using real conversation content."""
    messages = scan_messages(conn)
    if not messages:
        print("No user messages found — cannot seed KG data")
        return 0, 0, 0

    now = time.time()

    # --------------- ENTITIES ---------------
    # entity_id will be autoincremented starting at 1

    entities = [
        # (entity_type, canonical_key, canonical_name, created_node_id_hint)
        ("person", "user:self", "self", 0),                    # 1
        ("artifact", None, "Python", None),                    # 2
        ("artifact", None, "tenacity", None),                  # 3
        ("artifact", None, "speech synthesizer", None),        # 4
        ("artifact", None, "microphone", None),                # 5
        ("artifact", None, "Episodic", None),                  # 6
        ("topic", None, "weather", None),                      # 7
        ("topic", None, "Italian cooking", None),              # 8
        ("topic", None, "language models", None),              # 9
        ("org", None, "OpenAI", None),                         # 10
    ]

    # For each entity (except user:self), try to find a real message
    # containing its name to get a proper created_node_id
    msg_index = {}  # term -> (rowid, content)
    for rowid, nid, content in messages:
        for term in [
            "Python", "tenacity", "speech synthesizer", "microphone",
            "Episodic", "weather", "pasta", "Italian",
            "language model", "ChatGPT", "OpenAI",
        ]:
            if term.lower() in content.lower() and term.lower() not in msg_index:
                msg_index[term.lower()] = (rowid, content)

    # Resolve created_node_id for each entity
    term_map = {
        "Python": "python",
        "tenacity": "tenacity",
        "speech synthesizer": "speech synthesizer",
        "microphone": "microphone",
        "Episodic": "episodic",
        "weather": "weather",
        "Italian cooking": "italian",
        "language models": "language model",
        "OpenAI": "chatgpt",  # likely mentioned near ChatGPT
    }

    for i, (etype, ckey, cname, cnid) in enumerate(entities):
        if cname == "self":
            continue
        lookup = term_map.get(cname, cname.lower())
        if lookup in msg_index:
            entities[i] = (etype, ckey, cname, msg_index[lookup][0])

    # Insert entities
    entity_ids = {}
    for etype, ckey, cname, cnid in entities:
        conn.execute(
            "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
            "created_node_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (etype, ckey, cname, cnid or 0, now),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        entity_ids[cname] = eid

    entity_count = len(entities)
    print(f"Inserted {entity_count} entities")

    # --------------- ALIASES ---------------
    aliases = [
        ("Python", "python3"),
        ("Python", "py"),
        ("tenacity", "retry library"),
        ("speech synthesizer", "TTS"),
        ("speech synthesizer", "synthesizer"),
        ("microphone", "mic"),
        ("Episodic", "episodic"),
        ("weather", "forecast"),
        ("Italian cooking", "pasta"),
        ("language models", "LLMs"),
        ("language models", "AI"),
        ("OpenAI", "ChatGPT"),
    ]
    for entity_name, alias in aliases:
        eid = entity_ids.get(entity_name)
        if eid:
            conn.execute(
                "INSERT INTO kg_entity_aliases (entity_id, alias) VALUES (?, ?)",
                (eid, alias),
            )
    print(f"Inserted {len(aliases)} aliases")

    # --------------- ASSERTIONS + EDGES ---------------
    # Each edge needs an assertion backed by a real message span

    self_id = entity_ids["self"]

    # Define edges we want to create
    # (subject_name, predicate, object_name, search_term, tags)
    edge_defs = [
        ("self", "uses", "Python", "python", "[]"),
        ("self", "uses", "tenacity", "tenacity", "[]"),
        ("self", "uses", "speech synthesizer", "speech", "[]"),
        ("self", "uses", "microphone", "microphone", "[]"),
        ("self", "uses", "Episodic", "episodic", "[]"),
        ("self", "wants", "Italian cooking", "pasta", "[]"),
        ("self", "wants", "language models", "language model", "[]"),
        ("self", "prefers", "weather", "weather", "[]"),
        ("self", "role", "OpenAI", "chatgpt", '["TIME_PAST"]'),
        # A second edge with SENTIMENT_NEG — microphone quality issue
        ("self", "uses", "microphone", "microphone", '["SENTIMENT_NEG"]'),
    ]

    edge_count = 0
    for subj_name, predicate, obj_name, search_term, tags in edge_defs:
        subj_id = entity_ids.get(subj_name)
        obj_id = entity_ids.get(obj_name)
        if not subj_id or not obj_id:
            continue

        # Find a real message containing the search term
        target_msg = None
        for rowid, nid, content in messages:
            if search_term.lower() in content.lower():
                span = find_sentence_span(content, search_term)
                if span:
                    target_msg = (rowid, content, span[0], span[1])
                    break

        if not target_msg:
            # Fallback: create a synthetic span
            target_msg = (messages[0][0], messages[0][2], 0, min(30, len(messages[0][2])))

        rowid, content, span_start, span_end = target_msg

        # Determine asserted_by based on predicate
        asserted_by = "user" if predicate in ("uses", "wants", "prefers") else "extractor"

        # Insert assertion
        conn.execute(
            "INSERT INTO kg_assertions (node_id, span_start, span_end, asserted_by, "
            "polarity, certainty, status, tags) "
            "VALUES (?, ?, ?, ?, 'affirm', 'explicit', 'active', ?)",
            (rowid, span_start, span_end, asserted_by, tags),
        )
        assertion_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert edge
        conn.execute(
            "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
            "VALUES (?, ?, ?, ?)",
            (subj_id, predicate, obj_id, assertion_id),
        )
        edge_count += 1

    print(f"Inserted {edge_count} edges with assertions")

    # --------------- KG STATE ---------------
    # Record high-water mark
    max_rowid = messages[-1][0] if messages else 0
    conn.execute(
        "INSERT OR REPLACE INTO kg_state (key, value) VALUES ('high_water_mark', ?)",
        (str(max_rowid),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO kg_state (key, value) VALUES ('last_seed_time', ?)",
        (str(now),),
    )

    conn.commit()
    return entity_count, edge_count, len(aliases)


def main():
    parser = argparse.ArgumentParser(description="Seed KG test data")
    parser.add_argument("--clear", action="store_true", help="Drop and recreate KG tables")
    args = parser.parse_args()

    conn = get_db_connection()

    if args.clear:
        drop_tables(conn)
        create_tables(conn)
        ent, edg, ali = seed_data(conn)
        print(f"\nSeeded: {ent} entities, {edg} edges, {ali} aliases")
    elif tables_exist(conn):
        print("KG tables already have data. Use --clear to re-seed.")
    else:
        create_tables(conn)
        ent, edg, ali = seed_data(conn)
        print(f"\nSeeded: {ent} entities, {edg} edges, {ali} aliases")

    conn.close()


if __name__ == "__main__":
    main()
