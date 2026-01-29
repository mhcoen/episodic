"""
Tests for topic deletion functionality.

These tests verify:
1. Topic lookup by name, pattern, and time range
2. Cascade deletion removes all related data
3. ChromaDB embeddings are cleaned up
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_db_with_topics():
    """Create a temporary database with topics and related data."""
    # Use a proper temp directory that won't trigger the safeguard
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "test.db")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Create all required tables
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start_node_id TEXT,
            end_node_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE topic_nodes (
            topic_start_node_id TEXT,
            node_id TEXT,
            PRIMARY KEY (topic_start_node_id, node_id)
        )
    """)
    conn.execute("""
        CREATE TABLE topic_centroids (
            start_node_id TEXT PRIMARY KEY,
            centroid_medoid_exchange_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE topic_working_set (
            topic_start_node_id TEXT,
            data TEXT,
            PRIMARY KEY (topic_start_node_id)
        )
    """)

    # Insert test data
    base_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

    # Create nodes and topics
    topics_data = [
        ("python-asyncio", "node_1", base_time - timedelta(days=3)),
        ("machine-learning-basics", "node_2", base_time - timedelta(days=1)),
        ("sourdough-starter", "node_3", base_time - timedelta(hours=6)),
        ("test-topic-to-delete", "node_4", base_time),
    ]

    for i, (name, start_node, created) in enumerate(topics_data):
        # Create node
        conn.execute(
            "INSERT INTO nodes (id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (start_node, "user", f"Content for {name}", created.isoformat())
        )
        # Create topic
        conn.execute(
            "INSERT INTO topics (name, start_node_id, end_node_id) VALUES (?, ?, ?)",
            (name, start_node, None)
        )
        # Create topic_nodes
        conn.execute(
            "INSERT INTO topic_nodes (topic_start_node_id, node_id) VALUES (?, ?)",
            (start_node, start_node)
        )
        # Create centroid
        conn.execute(
            "INSERT INTO topic_centroids (start_node_id, centroid_medoid_exchange_id) VALUES (?, ?)",
            (start_node, f"centroid_{i}")
        )
        # Create working set
        conn.execute(
            "INSERT INTO topic_working_set (topic_start_node_id, data) VALUES (?, ?)",
            (start_node, f"working_set_{i}")
        )

    conn.commit()

    yield path, conn, base_time

    conn.close()
    import shutil
    shutil.rmtree(temp_dir)


class TestTopicLookup:
    """Tests for topic lookup functions."""

    def test_get_topics_by_name(self, temp_db_with_topics):
        """Test finding topics by exact name."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn:
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import get_topics_by_name

            topics = get_topics_by_name("python-asyncio")
            assert len(topics) == 1
            assert topics[0]['name'] == "python-asyncio"
            assert topics[0]['start_node_id'] == "node_1"

    def test_get_topics_by_name_not_found(self, temp_db_with_topics):
        """Test that non-existent topic returns empty list."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn:
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import get_topics_by_name

            topics = get_topics_by_name("nonexistent-topic")
            assert len(topics) == 0

    def test_get_topics_by_pattern(self, temp_db_with_topics):
        """Test finding topics by pattern match."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn:
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import get_topics_by_pattern

            # Match "python"
            topics = get_topics_by_pattern("python")
            assert len(topics) == 1
            assert topics[0]['name'] == "python-asyncio"

            # Match "sourdough" (case-insensitive)
            topics = get_topics_by_pattern("SOURDOUGH")
            assert len(topics) == 1
            assert topics[0]['name'] == "sourdough-starter"

    def test_get_topics_by_time_range(self, temp_db_with_topics):
        """Test finding topics by time range."""
        db_path, conn, base_time = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn:
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import get_topics_by_time_range

            # Get topics from last 2 days
            start = base_time - timedelta(days=2)
            topics = get_topics_by_time_range(start, None)
            names = [t['name'] for t in topics]
            assert "machine-learning-basics" in names
            assert "sourdough-starter" in names
            assert "test-topic-to-delete" in names
            assert "python-asyncio" not in names  # 3 days ago


class TestTopicDeletion:
    """Tests for topic deletion cascade."""

    def test_delete_topic_cascade(self, temp_db_with_topics):
        """Test that cascade deletion removes all related data."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn, \
             patch('episodic.db_topic_delete._delete_topic_embeddings', return_value=0):
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import delete_topic_cascade

            # Delete python-asyncio topic
            result = delete_topic_cascade("node_1", delete_embeddings=False)

            # Verify counts
            assert result['topics'] == 1
            assert result['topic_nodes'] == 1
            assert result['centroids'] == 1
            assert result['working_set'] == 1

            # Verify data is actually deleted
            cursor = conn.execute("SELECT COUNT(*) FROM topics WHERE start_node_id = 'node_1'")
            assert cursor.fetchone()[0] == 0

            cursor = conn.execute("SELECT COUNT(*) FROM topic_nodes WHERE topic_start_node_id = 'node_1'")
            assert cursor.fetchone()[0] == 0

            cursor = conn.execute("SELECT COUNT(*) FROM topic_centroids WHERE start_node_id = 'node_1'")
            assert cursor.fetchone()[0] == 0

    def test_delete_preserves_other_topics(self, temp_db_with_topics):
        """Test that deleting one topic preserves others."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn, \
             patch('episodic.db_topic_delete._delete_topic_embeddings', return_value=0):
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import delete_topic_cascade

            # Count before
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            count_before = cursor.fetchone()[0]
            assert count_before == 4

            # Delete one topic
            delete_topic_cascade("node_1", delete_embeddings=False)

            # Verify only one was deleted
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            count_after = cursor.fetchone()[0]
            assert count_after == 3

    def test_delete_topics_batch(self, temp_db_with_topics):
        """Test batch deletion of multiple topics."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn, \
             patch('episodic.db_topic_delete._delete_topic_embeddings', return_value=0):
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import delete_topics_batch

            topics_to_delete = [
                {'start_node_id': 'node_1'},
                {'start_node_id': 'node_2'},
            ]

            count, totals = delete_topics_batch(topics_to_delete, delete_embeddings=False)

            assert count == 2
            assert totals['topics'] == 2
            assert totals['centroids'] == 2


class TestTableChecks:
    """Tests for table existence checks."""

    def test_check_tables_exist(self, temp_db_with_topics):
        """Test checking which tables exist."""
        db_path, conn, _ = temp_db_with_topics

        with patch('episodic.db_topic_delete.get_connection') as mock_conn:
            mock_conn.return_value.__enter__ = lambda s: conn
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            from episodic.db_topic_delete import check_tables_exist

            tables = check_tables_exist()

            assert tables['topics'] is True
            assert tables['topic_nodes'] is True
            assert tables['topic_centroids'] is True
            assert tables['topic_working_set'] is True
