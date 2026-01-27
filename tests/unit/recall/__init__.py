"""
Test fixtures for recall module tests.

Provides:
- SQLite fixture database with topics, nodes, state
- FakeChroma for deterministic semantic hits
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class FakeChroma:
    """Fake Chroma collection that returns fixed results."""
    
    def __init__(self, fixed_results: Optional[Dict] = None):
        """
        Args:
            fixed_results: Dict with keys 'ids', 'distances', 'metadatas', 'documents'
                Each value is a list of lists (batch format).
        """
        self._results = fixed_results or {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'documents': [[]]
        }
    
    def query(self, query_texts: List[str], n_results: int = 10, **kwargs) -> Dict:
        """Return fixed results regardless of query."""
        return self._results
    
    def set_results(self, ids: List[str], distances: List[float], metadatas: List[Dict]):
        """Set fixed results for next query."""
        self._results = {
            'ids': [ids],
            'distances': [distances],
            'metadatas': [metadatas],
            'documents': [[''] * len(ids)]
        }


def create_test_db(tmp_path) -> Tuple[sqlite3.Connection, str]:
    """
    Create a test SQLite database with recall fixtures.
    
    Returns:
        (connection, db_path)
    
    Schema:
        - nodes: Linear chain with IDs node_1..node_12
        - topics: 3 topics with defined boundaries
        - state: head pointing to node_12
    """
    db_path = str(tmp_path / "test_recall.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Create schema
    conn.executescript("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT UNIQUE,
            content TEXT NOT NULL,
            parent_id TEXT,
            role TEXT,
            provider TEXT,
            model TEXT,
            is_meta_query INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(parent_id) REFERENCES nodes(id)
        );
        
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_node_id TEXT NOT NULL,
            end_node_id TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(start_node_id) REFERENCES nodes(id),
            FOREIGN KEY(end_node_id) REFERENCES nodes(id)
        );
        
        CREATE TABLE state (
            name TEXT PRIMARY KEY,
            head_id TEXT
        );
        
        CREATE TABLE compressions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            compressed_node_id TEXT NOT NULL,
            original_branch_head TEXT NOT NULL,
            original_node_count INTEGER NOT NULL,
            original_words INTEGER NOT NULL,
            compressed_words INTEGER NOT NULL,
            compression_ratio REAL NOT NULL,
            strategy TEXT NOT NULL,
            duration_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_nodes_parent ON nodes(parent_id);
    """)
    
    # Insert nodes: linear chain node_1 -> node_2 -> ... -> node_12
    # Alternating user/assistant roles
    nodes = []
    parent_id = None
    for i in range(1, 13):
        node_id = f"node_{i}"
        role = "user" if i % 2 == 1 else "assistant"
        content = f"Content for node {i}"
        nodes.append((node_id, f"n{i}", content, parent_id, role, "test", "test-model",
                      f"2026-01-{15 + i // 4:02d} 10:{i:02d}:00"))
        parent_id = node_id
    
    conn.executemany("""
        INSERT INTO nodes (id, short_id, content, parent_id, role, provider, model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, nodes)
    
    # Insert topics:
    # Topic 1 (id=1): node_1 to node_4 (closed)
    # Topic 2 (id=2): node_5 to node_8 (closed)
    # Topic 3 (id=3): node_9 to NULL (ongoing)
    topics = [
        ("topic-alpha", "node_1", "node_4"),
        ("topic-beta", "node_5", "node_8"),
        ("topic-gamma", "node_9", None),  # Ongoing
    ]
    
    conn.executemany("""
        INSERT INTO topics (name, start_node_id, end_node_id)
        VALUES (?, ?, ?)
    """, topics)
    
    # Set head to last node
    conn.execute("INSERT INTO state (name, head_id) VALUES ('head', 'node_12')")
    
    conn.commit()
    return conn, db_path


def create_overlap_db(tmp_path) -> sqlite3.Connection:
    """
    Create a test DB where one node belongs to multiple topics.
    
    For testing first-match-wins behavior.
    
    Node chain: node_1 -> node_2 -> node_3 -> node_4 -> node_5
    Topic 1 (id=1): node_1 to node_3
    Topic 2 (id=2): node_2 to node_4  (overlaps with topic 1 at node_2, node_3)
    """
    db_path = str(tmp_path / "test_overlap.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.executescript("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT UNIQUE,
            content TEXT NOT NULL,
            parent_id TEXT,
            role TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_node_id TEXT NOT NULL,
            end_node_id TEXT
        );
        
        CREATE TABLE state (
            name TEXT PRIMARY KEY,
            head_id TEXT
        );
        
        CREATE INDEX idx_nodes_parent ON nodes(parent_id);
    """)
    
    # Insert nodes
    nodes = [
        ("node_1", "n1", "Content 1", None, "user"),
        ("node_2", "n2", "Content 2", "node_1", "assistant"),
        ("node_3", "n3", "Content 3", "node_2", "user"),
        ("node_4", "n4", "Content 4", "node_3", "assistant"),
        ("node_5", "n5", "Content 5", "node_4", "user"),
    ]
    conn.executemany("""
        INSERT INTO nodes (id, short_id, content, parent_id, role)
        VALUES (?, ?, ?, ?, ?)
    """, nodes)
    
    # Insert overlapping topics
    # Topic 1 has lower id, so should win for overlapping nodes
    topics = [
        ("topic-first", "node_1", "node_3"),   # id=1
        ("topic-second", "node_2", "node_4"),  # id=2, overlaps at node_2, node_3
    ]
    conn.executemany("""
        INSERT INTO topics (name, start_node_id, end_node_id)
        VALUES (?, ?, ?)
    """, topics)
    
    conn.execute("INSERT INTO state (name, head_id) VALUES ('head', 'node_5')")
    conn.commit()
    return conn


def create_hits(exchange_ids: List[str], similarities: List[float]) -> List[Dict]:
    """Create hit dicts for testing."""
    return [
        {
            'exchange_id': eid,
            'similarity': sim,
            'relevance_score': sim,
            'metadata': {'user_id': eid}
        }
        for eid, sim in zip(exchange_ids, similarities)
    ]
