"""
Test Fixture Management for Episodic.

Provides infrastructure for injecting and cleaning up test data
in a separate test database. Enables end-to-end testing of
query understanding and retrieval systems.

Usage:
    from episodic.test_fixtures import FixtureManager
    
    manager = FixtureManager()
    manager.initialize_test_db()
    manager.inject_fixtures()
    # ... run tests ...
    manager.cleanup()
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo
import uuid


# Test database location
TEST_DB_PATH = Path.home() / ".episodic" / "episodic_test.db"
PROD_DB_PATH = Path.home() / ".episodic" / "episodic.db"


def generate_node_id() -> str:
    """Generate a UUID for a node."""
    return str(uuid.uuid4())


def generate_short_id(index: int) -> str:
    """Generate a deterministic short ID for testing."""
    # Use base-36 encoding for short IDs
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if index < 36:
        return f"t{chars[index]}"
    else:
        return f"{chars[index // 36]}{chars[index % 36]}"


class TestFixture:
    """A single test fixture representing a conversation exchange."""
    
    def __init__(
        self,
        user_content: str,
        assistant_content: str,
        timestamp: datetime,
        topic_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        self.user_content = user_content
        self.assistant_content = assistant_content
        self.timestamp = timestamp
        self.topic_name = topic_name
        self.tags = tags or []
        
        # Generated IDs (set during injection)
        self.user_node_id: Optional[str] = None
        self.assistant_node_id: Optional[str] = None
        self.user_short_id: Optional[str] = None
        self.assistant_short_id: Optional[str] = None


class FixtureManager:
    """
    Manages test database and fixture injection.
    
    Provides complete isolation from production data by using
    a separate database file.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or TEST_DB_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self._node_counter = 0
        self._topic_counter = 0
        
    def initialize_test_db(self, clean: bool = True) -> None:
        """
        Initialize the test database.
        
        Args:
            clean: If True, delete existing test DB and start fresh
        """
        if clean and self.db_path.exists():
            self.db_path.unlink()
            
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_schema()
        
    def _create_schema(self) -> None:
        """Create the database schema for testing."""
        self.conn.executescript("""
            -- Core conversation storage
            CREATE TABLE IF NOT EXISTS nodes (
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
            
            -- Metadata storage
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            -- Topic tracking
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                start_node_id TEXT NOT NULL,
                end_node_id TEXT,
                confidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(start_node_id) REFERENCES nodes(id),
                FOREIGN KEY(end_node_id) REFERENCES nodes(id)
            );
            
            -- Topic node cache for efficient retrieval
            CREATE TABLE IF NOT EXISTS topic_node_cache (
                topic_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                PRIMARY KEY (topic_id, node_id)
            );
            
            -- Compression tracking
            CREATE TABLE IF NOT EXISTS compressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                compressed_node_id TEXT NOT NULL,
                original_branch_head TEXT NOT NULL,
                original_node_count INTEGER NOT NULL,
                original_words INTEGER NOT NULL,
                compressed_words INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                strategy TEXT NOT NULL,
                duration_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(compressed_node_id) REFERENCES nodes(id),
                FOREIGN KEY(original_branch_head) REFERENCES nodes(id)
            );
            
            -- Topic detection scores
            CREATE TABLE IF NOT EXISTS topic_detection_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_node_short_id TEXT NOT NULL UNIQUE,
                window_size INTEGER NOT NULL,
                detection_method TEXT DEFAULT 'sliding_window',
                window_a_start_short_id TEXT,
                window_a_end_short_id TEXT,
                window_a_size INTEGER NOT NULL,
                window_b_start_short_id TEXT,
                window_b_end_short_id TEXT,
                window_b_size INTEGER NOT NULL,
                drift_score REAL NOT NULL,
                keyword_score REAL NOT NULL,
                combined_score REAL NOT NULL,
                is_boundary BOOLEAN NOT NULL,
                transition_phrase TEXT,
                detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                threshold_used REAL
            );
            
            -- Schema migrations
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_short_id ON nodes(short_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON nodes(created_at);
            CREATE INDEX IF NOT EXISTS idx_topics_boundaries ON topics(start_node_id, end_node_id);
            CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);
        """)
        self.conn.commit()
        
    def inject_fixture(
        self,
        fixture: TestFixture,
        parent_node_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Inject a single fixture into the test database.
        
        Args:
            fixture: The fixture to inject
            parent_node_id: Parent node ID for threading
            
        Returns:
            Tuple of (user_node_id, assistant_node_id)
        """
        # Generate IDs
        fixture.user_node_id = generate_node_id()
        fixture.assistant_node_id = generate_node_id()
        fixture.user_short_id = generate_short_id(self._node_counter)
        self._node_counter += 1
        fixture.assistant_short_id = generate_short_id(self._node_counter)
        self._node_counter += 1
        
        # Format timestamp for SQLite
        ts_str = fixture.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert user node
        self.conn.execute("""
            INSERT INTO nodes (id, short_id, content, parent_id, role, provider, model, created_at)
            VALUES (?, ?, ?, ?, 'user', 'test', 'test-model', ?)
        """, (
            fixture.user_node_id,
            fixture.user_short_id,
            fixture.user_content,
            parent_node_id,
            ts_str
        ))
        
        # Insert assistant node
        self.conn.execute("""
            INSERT INTO nodes (id, short_id, content, parent_id, role, provider, model, created_at)
            VALUES (?, ?, ?, ?, 'assistant', 'test', 'test-model', ?)
        """, (
            fixture.assistant_node_id,
            fixture.assistant_short_id,
            fixture.assistant_content,
            fixture.user_node_id,
            ts_str
        ))
        
        self.conn.commit()
        return fixture.user_node_id, fixture.assistant_node_id
    
    def create_topic(
        self,
        name: str,
        start_node_id: str,
        end_node_id: Optional[str] = None,
        node_ids: Optional[List[str]] = None
    ) -> int:
        """
        Create a topic in the test database.
        
        Args:
            name: Topic name
            start_node_id: First node in topic
            end_node_id: Last node in topic (None for ongoing)
            node_ids: List of all node IDs in the topic
            
        Returns:
            Topic ID
        """
        cursor = self.conn.execute("""
            INSERT INTO topics (name, start_node_id, end_node_id, confidence)
            VALUES (?, ?, ?, 'high')
        """, (name, start_node_id, end_node_id))
        
        topic_id = cursor.lastrowid
        
        # Populate topic_node_cache
        if node_ids:
            for node_id in node_ids:
                self.conn.execute("""
                    INSERT OR IGNORE INTO topic_node_cache (topic_id, node_id)
                    VALUES (?, ?)
                """, (str(topic_id), node_id))
        
        self.conn.commit()
        return topic_id
    
    def inject_standard_fixtures(self, reference_time: Optional[datetime] = None) -> dict:
        """
        Inject a standard set of test fixtures for comprehensive testing.
        
        Creates conversations at known temporal offsets for testing:
        - yesterday
        - 3 days ago
        - last week
        - last month
        
        Args:
            reference_time: Reference time (defaults to now UTC)
            
        Returns:
            Dictionary mapping fixture names to their details
        """
        if reference_time is None:
            reference_time = datetime.now(ZoneInfo("UTC"))
            
        fixtures = {}
        last_node_id = None
        
        # === TOPIC 1: Quantum Computing (last month) ===
        topic1_nodes = []
        
        f1 = TestFixture(
            user_content="What is quantum computing and how does it differ from classical computing?",
            assistant_content="Quantum computing uses quantum bits (qubits) that can exist in superposition, unlike classical bits which are strictly 0 or 1. This enables quantum computers to process certain types of problems exponentially faster through quantum parallelism and entanglement.",
            timestamp=reference_time - timedelta(days=35),
            topic_name="quantum-computing"
        )
        user_id, asst_id = self.inject_fixture(f1, last_node_id)
        topic1_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["quantum_intro"] = f1
        
        f2 = TestFixture(
            user_content="What are the main challenges in building quantum computers?",
            assistant_content="The main challenges include: maintaining quantum coherence (qubits are extremely sensitive to environmental interference), error correction (quantum errors accumulate rapidly), and scaling (current systems have limited qubits). Temperatures near absolute zero are often required.",
            timestamp=reference_time - timedelta(days=34),
            topic_name="quantum-computing"
        )
        user_id, asst_id = self.inject_fixture(f2, last_node_id)
        topic1_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["quantum_challenges"] = f2
        
        # Create topic 1
        self.create_topic(
            "quantum-computing",
            topic1_nodes[0],
            topic1_nodes[-1],
            topic1_nodes
        )
        
        # === TOPIC 2: Database Indexing (last week) ===
        topic2_nodes = []
        
        f3 = TestFixture(
            user_content="How do B-tree indexes work in databases?",
            assistant_content="B-trees are self-balancing tree structures where each node can have multiple children. They keep data sorted and allow searches, insertions, and deletions in O(log n) time. The 'B' stands for 'balanced'. Each node contains multiple keys and pointers, making them efficient for disk-based storage.",
            timestamp=reference_time - timedelta(days=8),
            topic_name="database-indexing"
        )
        user_id, asst_id = self.inject_fixture(f3, last_node_id)
        topic2_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["btree_intro"] = f3
        
        f4 = TestFixture(
            user_content="When should I use a hash index versus a B-tree index?",
            assistant_content="Use hash indexes for exact equality lookups (WHERE id = 5) as they're O(1). Use B-tree indexes for range queries (WHERE date > '2024-01-01'), ordering (ORDER BY), and prefix searches (LIKE 'abc%'). Hash indexes don't support range scans.",
            timestamp=reference_time - timedelta(days=7),
            topic_name="database-indexing"
        )
        user_id, asst_id = self.inject_fixture(f4, last_node_id)
        topic2_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["index_comparison"] = f4
        
        # Create topic 2
        self.create_topic(
            "database-indexing",
            topic2_nodes[0],
            topic2_nodes[-1],
            topic2_nodes
        )
        
        # === TOPIC 3: Python Asyncio (3 days ago) ===
        topic3_nodes = []
        
        f5 = TestFixture(
            user_content="Can you explain Python's asyncio and when to use it?",
            assistant_content="Asyncio is Python's framework for writing concurrent code using async/await syntax. Use it for I/O-bound tasks like network requests, file operations, or database queries where you're waiting on external resources. It's NOT suitable for CPU-bound tasks - use multiprocessing for those.",
            timestamp=reference_time - timedelta(days=3),
            topic_name="python-asyncio"
        )
        user_id, asst_id = self.inject_fixture(f5, last_node_id)
        topic3_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["asyncio_intro"] = f5
        
        f6 = TestFixture(
            user_content="What's the difference between asyncio.gather and asyncio.wait?",
            assistant_content="asyncio.gather() runs awaitables concurrently and returns results in order. It raises the first exception by default. asyncio.wait() returns two sets: done and pending tasks, giving you more control. Use gather for simple concurrent execution, wait when you need to handle partial completion or timeouts.",
            timestamp=reference_time - timedelta(days=3, hours=1),
            topic_name="python-asyncio"
        )
        user_id, asst_id = self.inject_fixture(f6, last_node_id)
        topic3_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["asyncio_gather_wait"] = f6
        
        # Create topic 3
        self.create_topic(
            "python-asyncio",
            topic3_nodes[0],
            topic3_nodes[-1],
            topic3_nodes
        )
        
        # === TOPIC 4: Machine Learning Basics (yesterday) ===
        topic4_nodes = []
        
        f7 = TestFixture(
            user_content="What's the difference between supervised and unsupervised learning?",
            assistant_content="Supervised learning trains on labeled data (input-output pairs) to predict outputs for new inputs - examples include classification and regression. Unsupervised learning finds patterns in unlabeled data - examples include clustering and dimensionality reduction. Semi-supervised learning combines both approaches.",
            timestamp=reference_time - timedelta(days=1),
            topic_name="machine-learning-basics"
        )
        user_id, asst_id = self.inject_fixture(f7, last_node_id)
        topic4_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["ml_supervised_unsupervised"] = f7
        
        f8 = TestFixture(
            user_content="How do I know if my model is overfitting?",
            assistant_content="Signs of overfitting: training accuracy much higher than validation accuracy, model performs poorly on new data, very complex model with many parameters. Solutions: cross-validation, regularization (L1/L2), dropout, early stopping, getting more training data, or simplifying the model.",
            timestamp=reference_time - timedelta(days=1, hours=-2),
            topic_name="machine-learning-basics"
        )
        user_id, asst_id = self.inject_fixture(f8, last_node_id)
        topic4_nodes.extend([user_id, asst_id])
        last_node_id = asst_id
        fixtures["ml_overfitting"] = f8
        
        # Create topic 4 (ongoing - no end node)
        self.create_topic(
            "machine-learning-basics",
            topic4_nodes[0],
            None,  # Ongoing topic
            topic4_nodes
        )
        
        return fixtures
    
    def get_connection(self) -> sqlite3.Connection:
        """Get the test database connection."""
        if self.conn is None:
            raise RuntimeError("Test database not initialized. Call initialize_test_db() first.")
        return self.conn
    
    def cleanup(self) -> None:
        """Close connection and optionally remove test database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def destroy(self) -> None:
        """Close connection and delete the test database file."""
        self.cleanup()
        if self.db_path.exists():
            self.db_path.unlink()


# === Convenience functions for test scripts ===

def setup_test_environment(reference_time: Optional[datetime] = None) -> FixtureManager:
    """
    Set up a complete test environment with standard fixtures.
    
    Args:
        reference_time: Reference time for temporal fixtures
        
    Returns:
        Configured FixtureManager
    """
    manager = FixtureManager()
    manager.initialize_test_db(clean=True)
    manager.inject_standard_fixtures(reference_time)
    return manager


def teardown_test_environment(manager: FixtureManager, delete_db: bool = False) -> None:
    """
    Clean up test environment.
    
    Args:
        manager: The fixture manager to clean up
        delete_db: If True, delete the test database file
    """
    if delete_db:
        manager.destroy()
    else:
        manager.cleanup()


if __name__ == "__main__":
    # Quick verification
    print("Setting up test fixtures...")
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # Use a fixed reference time for reproducibility
    ref_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    
    manager = setup_test_environment(ref_time)
    
    print(f"Test database created at: {manager.db_path}")
    
    # Verify fixtures
    cursor = manager.conn.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    print(f"Nodes created: {node_count}")
    
    cursor = manager.conn.execute("SELECT COUNT(*) FROM topics")
    topic_count = cursor.fetchone()[0]
    print(f"Topics created: {topic_count}")
    
    cursor = manager.conn.execute("SELECT name, start_node_id, end_node_id FROM topics")
    for row in cursor.fetchall():
        status = "ongoing" if row[2] is None else "closed"
        print(f"  - {row[0]} ({status})")
    
    manager.cleanup()
    print("Done.")
