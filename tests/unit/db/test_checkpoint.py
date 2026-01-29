"""
Tests for checkpoint-based incremental indexing.

These tests verify that:
1. Checkpoints persist across function calls
2. get_nodes_after_checkpoint returns only new nodes
3. Incremental backfill uses checkpoints correctly
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_db():
    """Create a temporary database with test data."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Create tables
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE configuration (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Insert test nodes
    for i in range(10):
        conn.execute(
            "INSERT INTO nodes (id, role, content) VALUES (?, ?, ?)",
            (f"node_{i}", "user" if i % 2 == 0 else "assistant", f"Test content {i}")
        )
    conn.commit()

    yield path, conn

    conn.close()
    os.unlink(path)


def test_get_embedding_checkpoint_default(temp_db):
    """Test that checkpoint returns 0 when not set."""
    db_path, conn = temp_db

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import get_embedding_checkpoint
        checkpoint = get_embedding_checkpoint()
        assert checkpoint == 0


def test_set_and_get_checkpoint(temp_db):
    """Test setting and retrieving checkpoint."""
    db_path, conn = temp_db

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import set_embedding_checkpoint, get_embedding_checkpoint

        # Set checkpoint
        set_embedding_checkpoint(5)

        # Verify it was stored
        cursor = conn.execute("SELECT value FROM configuration WHERE key = 'embedding_checkpoint_rowid'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "5"

        # Verify we can retrieve it
        checkpoint = get_embedding_checkpoint()
        assert checkpoint == 5


def test_get_nodes_after_checkpoint(temp_db):
    """Test that get_nodes_after_checkpoint returns only new nodes."""
    db_path, conn = temp_db

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import get_nodes_after_checkpoint

        # Get all nodes (checkpoint 0)
        all_nodes = get_nodes_after_checkpoint(0)
        assert len(all_nodes) == 10

        # Get nodes after rowid 5
        new_nodes = get_nodes_after_checkpoint(5)
        assert len(new_nodes) == 5

        # Verify all returned nodes have rowid > 5
        for node in new_nodes:
            assert node['rowid'] > 5


def test_get_max_node_rowid(temp_db):
    """Test getting the maximum node rowid."""
    db_path, conn = temp_db

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import get_max_node_rowid

        max_rowid = get_max_node_rowid()
        assert max_rowid == 10


def test_checkpoint_survives_reconnection(temp_db):
    """Test that checkpoint persists across reconnections."""
    db_path, conn = temp_db

    # Set checkpoint
    conn.execute(
        "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
        ('embedding_checkpoint_rowid', '7')
    )
    conn.commit()

    # Close and reopen
    conn.close()
    conn2 = sqlite3.connect(db_path)
    conn2.row_factory = sqlite3.Row

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn2
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import get_embedding_checkpoint
        checkpoint = get_embedding_checkpoint()
        assert checkpoint == 7

    conn2.close()


def test_ensure_configuration_table_idempotent(temp_db):
    """Test that ensure_configuration_table is safe to call multiple times."""
    db_path, conn = temp_db

    with patch('episodic.db_checkpoint.get_connection') as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        from episodic.db_checkpoint import ensure_configuration_table

        # Call multiple times - should not raise
        ensure_configuration_table()
        ensure_configuration_table()
        ensure_configuration_table()

        # Table should still exist with data
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='configuration'"
        )
        assert cursor.fetchone() is not None
