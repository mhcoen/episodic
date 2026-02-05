"""
Fixtures for persistence tests.

These tests use temporary file-based databases to verify
state persists across session restarts.
"""

import os
import pytest
import tempfile


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for persistence tests."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="episodic_test_")
    os.close(fd)
    # Remove the empty file - SQLite will create it
    os.unlink(path)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_episodic_home(tmp_path):
    """Create a temporary EPISODIC_HOME directory."""
    home = tmp_path / ".episodic"
    home.mkdir(parents=True, exist_ok=True)
    return str(home)
