"""
Unit test specific fixtures and configuration.

Unit tests should be:
- Fast (< 1 second each)
- Isolated (no external dependencies)
- Deterministic (same result every run)

This conftest provides fixtures that enforce isolation.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def no_external_calls():
    """
    Automatically applied to all unit tests.

    Patches common external calls to ensure tests are isolated.
    Tests that need real external calls should be in integration/.
    """
    patches = [
        patch('episodic.llm.litellm.completion', side_effect=RuntimeError(
            "Unit tests should not make real LLM calls. Use mock_llm fixture."
        )),
    ]

    for p in patches:
        try:
            p.start()
        except (ImportError, ModuleNotFoundError):
            # Module not available, skip this patch
            pass

    yield

    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            # Patch wasn't started
            pass


@pytest.fixture
def mock_db_connection():
    """
    Provide a mock database connection for unit tests.

    Use this when you don't need a real database.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []
    return mock_conn, mock_cursor


@pytest.fixture
def mock_config():
    """
    Provide a mock config object without touching global state.

    Use for testing config-dependent code in isolation.
    """
    config = MagicMock()
    config.get.return_value = None
    config._config = {}

    def mock_get(key, default=None):
        return config._config.get(key, default)

    def mock_set(key, value):
        config._config[key] = value

    config.get = mock_get
    config.set = mock_set
    return config


@pytest.fixture
def fast_embeddings():
    """
    Provide fast fake embeddings for unit tests.

    Returns consistent embeddings based on input hash.
    """
    import hashlib

    def get_embedding(text):
        # Create deterministic fake embedding based on text
        hash_bytes = hashlib.md5(text.encode()).digest()
        # Convert to list of floats between 0 and 1
        return [b / 255.0 for b in hash_bytes] * 24  # 384 dimensions

    return get_embedding
