"""
Test fixtures for retrieval system tests.

Golden fixtures aligned with v1.1 spec success criteria.
"""
import pytest
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# =============================================================================
# Golden Fixture Data: fx_base
# =============================================================================

FX_BASE_NODES = [
    # Root system node
    {"id": "S0", "role": "system", "parent_id": None, 
     "created_at": "2026-01-01T00:00:00.000000Z", "content": "System initialized"},
    
    # Exchange 1 (segment 1: coffee)
    {"id": "U1", "role": "user", "parent_id": "S0",
     "created_at": "2026-01-02T10:00:00.000000Z", "content": "talk about coffee"},
    {"id": "A1", "role": "assistant", "parent_id": "U1",
     "created_at": "2026-01-02T10:00:01.000000Z", "content": "coffee reply v1"},
    
    # Exchange 2 (segment 1: coffee)
    {"id": "U2", "role": "user", "parent_id": "A1",
     "created_at": "2026-01-02T11:00:00.000000Z", "content": "espresso grinder"},
    {"id": "A2", "role": "assistant", "parent_id": "U2",
     "created_at": "2026-01-02T11:00:01.000000Z", "content": "grinder reply"},
    
    # Branching: alternate assistant for U2 (off-ancestry)
    {"id": "A2b", "role": "assistant", "parent_id": "U2",
     "created_at": "2026-01-02T11:00:02.000000Z", "content": "grinder reply alt"},
    
    # Exchange 3 (segment 2: legal) - continues from A2, not A2b
    {"id": "U3", "role": "user", "parent_id": "A2",
     "created_at": "2026-01-03T09:00:00.000000Z", "content": "topic shift: legal"},
    {"id": "A3", "role": "assistant", "parent_id": "U3",
     "created_at": "2026-01-03T09:00:01.000000Z", "content": "legal reply"},
]

FX_BASE_TOPICS = [
    {"id": 1, "name": "coffee", "start_node_id": "U1", "end_node_id": "U2"},
    {"id": 2, "name": "legal", "start_node_id": "U3", "end_node_id": None},  # ongoing
]

FX_BASE_HEAD = "A3"


# =============================================================================
# Golden Fixture Data: fx_chroma_grinder (Chroma results for "grinder")
# =============================================================================

FX_CHROMA_GRINDER = {
    "ids": [["U2", "U1"]],
    "distances": [[0.10, 0.30]],
    "metadatas": [[
        {"user_id": "U2", "assistant_id": "A2b", "timestamp": "2026-01-02T11:00:00.000000Z"},
        {"user_id": "U1", "assistant_id": "A1", "timestamp": "2026-01-02T10:00:00.000000Z"},
    ]],
}


# =============================================================================
# Golden Fixture Data: fx_overlap (segment overlap violation)
# =============================================================================

FX_OVERLAP_NODES = FX_BASE_NODES.copy()

FX_OVERLAP_TOPICS = [
    {"id": 1, "name": "coffee", "start_node_id": "U1", "end_node_id": "U2"},
    {"id": 2, "name": "beverages", "start_node_id": "U1", "end_node_id": "A1"},  # overlaps with topic 1
]


# =============================================================================
# Test Database Setup Helpers
# =============================================================================

def create_test_schema(conn: sqlite3.Connection) -> None:
    """Create the minimal schema needed for retrieval tests."""
    cursor = conn.cursor()
    
    # nodes table (matches existing Episodic schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT UNIQUE,
            content TEXT NOT NULL,
            parent_id TEXT,
            role TEXT,
            provider TEXT,
            model TEXT,
            is_meta_query BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now') || 'Z'),
            FOREIGN KEY(parent_id) REFERENCES nodes(id)
        )
    """)
    
    # topics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_node_id TEXT NOT NULL,
            end_node_id TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(start_node_id) REFERENCES nodes(id),
            FOREIGN KEY(end_node_id) REFERENCES nodes(id)
        )
    """)
    
    # state table for head tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS state (
            name TEXT PRIMARY KEY,
            head_id TEXT,
            FOREIGN KEY(head_id) REFERENCES nodes(id)
        )
    """)
    
    conn.commit()


def populate_nodes(conn: sqlite3.Connection, nodes: List[Dict]) -> None:
    """Insert nodes into the database."""
    cursor = conn.cursor()
    for node in nodes:
        cursor.execute("""
            INSERT INTO nodes (id, content, parent_id, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (node["id"], node["content"], node.get("parent_id"), 
              node.get("role"), node.get("created_at")))
    conn.commit()


def populate_topics(conn: sqlite3.Connection, topics: List[Dict]) -> None:
    """Insert topics into the database."""
    cursor = conn.cursor()
    for topic in topics:
        cursor.execute("""
            INSERT INTO topics (id, name, start_node_id, end_node_id)
            VALUES (?, ?, ?, ?)
        """, (topic["id"], topic["name"], topic["start_node_id"], topic.get("end_node_id")))
    conn.commit()


def set_head(conn: sqlite3.Connection, head_id: str) -> None:
    """Set the head pointer."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO state (name, head_id) VALUES ('head', ?)", (head_id,))
    conn.commit()


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def retrieval_conn():
    """In-memory SQLite connection with row_factory set."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_test_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def fx_base(retrieval_conn):
    """Base fixture with coffee/legal topics and branching assistant."""
    populate_nodes(retrieval_conn, FX_BASE_NODES)
    populate_topics(retrieval_conn, FX_BASE_TOPICS)
    set_head(retrieval_conn, FX_BASE_HEAD)
    return retrieval_conn


@pytest.fixture
def fx_overlap(retrieval_conn):
    """Fixture with overlapping segments for AUDIT testing."""
    populate_nodes(retrieval_conn, FX_OVERLAP_NODES)
    populate_topics(retrieval_conn, FX_OVERLAP_TOPICS)
    set_head(retrieval_conn, FX_BASE_HEAD)
    return retrieval_conn


@pytest.fixture
def migration_conn():
    """Connection for migration tests with isolation_level=None."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Create base nodes table without FTS
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            parent_id TEXT,
            role TEXT,
            is_meta_query BOOLEAN DEFAULT FALSE,
            created_at TEXT
        )
    """)
    yield conn
    conn.close()


class FakeChroma:
    """Stub Chroma client for testing."""
    
    def __init__(self, responses: Dict[str, Dict] = None):
        self.responses = responses or {}
        self.query_count = 0
        self.last_query = None
    
    def query(self, query_texts: List[str], n_results: int = 10, **kwargs) -> Dict:
        self.query_count += 1
        self.last_query = query_texts[0] if query_texts else None
        
        # Return configured response or empty
        if self.last_query and self.last_query in self.responses:
            return self.responses[self.last_query]
        
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}


@pytest.fixture
def fake_chroma_grinder():
    """FakeChroma configured with grinder query response."""
    return FakeChroma(responses={"grinder": FX_CHROMA_GRINDER})


@pytest.fixture
def fake_chroma_empty():
    """FakeChroma that returns empty for all queries."""
    return FakeChroma()


# =============================================================================
# AUDIT Log Capture
# =============================================================================

class AuditLogCapture:
    """Capture AUDIT log messages for assertion."""
    
    def __init__(self):
        self.messages: List[str] = []
    
    def debug(self, msg: str, *args):
        if "AUDIT" in msg:
            self.messages.append(msg % args if args else msg)
    
    def warning(self, msg: str, *args):
        if "AUDIT" in msg:
            self.messages.append(msg % args if args else msg)
    
    def info(self, msg: str, *args):
        pass
    
    def error(self, msg: str, *args):
        pass
    
    def contains(self, substring: str) -> bool:
        return any(substring in msg for msg in self.messages)
    
    def clear(self):
        self.messages.clear()


@pytest.fixture
def audit_capture():
    """Fixture to capture AUDIT log messages."""
    return AuditLogCapture()


# =============================================================================
# Golden Expected Outputs
# =============================================================================

GOLDEN_GRINDER_FUSED = {
    "ordered_exchange_ids": ["U2", "U1"],
    "display_pairs": {
        "U2": "A2b",  # From metadata.assistant_id (valid even though off-ancestry)
        "U1": "A1",
    }
}

GOLDEN_SEGMENT_1_NODES = ["U1", "A1", "U2", "A2"]  # coffee segment
GOLDEN_SEGMENT_2_NODES = ["U3", "A3"]  # legal segment (ongoing, head=A3)

GOLDEN_OVERLAP_FIRST_WINS = {
    "U1": 1,  # First topic by id ASC
    "A1": 1,
}
