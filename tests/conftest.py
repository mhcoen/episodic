"""
Pytest configuration and shared fixtures for Episodic test suite.

This file provides:
- Database fixtures for testing persistence
- Configuration fixtures for isolated testing
- LLM mocking fixtures
- RAG testing fixtures
- CLI testing fixtures
- Test conversation data fixtures
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set a test HOME before importing episodic config to avoid touching user files.
_ORIGINAL_HOME = os.environ.get("HOME")
_ORIGINAL_EPISODIC_HOME = os.environ.get("EPISODIC_HOME")
_TEST_HOME_DIR = tempfile.mkdtemp(prefix="episodic_test_home_")
os.environ["HOME"] = _TEST_HOME_DIR
os.environ["EPISODIC_HOME"] = _TEST_HOME_DIR

# Snapshot base config once for deterministic resets.
from episodic.config import config as _base_config
_BASE_CONFIG = dict(_base_config._template_defaults)

# Resolve a test DB path before any database imports.
# The DB path is read once when the connection pool is first created.
_ORIGINAL_DB_PATH = os.environ.get("EPISODIC_DB_PATH")
_ORIGINAL_DISABLE_POOL = os.environ.get("EPISODIC_DISABLE_POOL")
_TEST_DB_DIR = tempfile.mkdtemp(prefix="episodic_test_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "episodic_test.db")
os.environ["EPISODIC_DB_PATH"] = _TEST_DB_PATH
os.environ["EPISODIC_DISABLE_POOL"] = "true"


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_db_env():
    """
    Set a temp database path for the entire test session.

    The DB path is read once when the connection pool is first created.
    Changing EPISODIC_DB_PATH mid-process has no effect.
    """
    yield _TEST_DB_PATH

    # Cleanup after all tests complete
    try:
        from episodic.db_connection import close_pool
        close_pool()
    except Exception:
        pass

    if os.path.exists(_TEST_DB_PATH):
        try:
            os.remove(_TEST_DB_PATH)
        except Exception:
            pass
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)

    if _ORIGINAL_DB_PATH is not None:
        os.environ["EPISODIC_DB_PATH"] = _ORIGINAL_DB_PATH
    else:
        os.environ.pop("EPISODIC_DB_PATH", None)

    if _ORIGINAL_DISABLE_POOL is not None:
        os.environ["EPISODIC_DISABLE_POOL"] = _ORIGINAL_DISABLE_POOL
    else:
        os.environ.pop("EPISODIC_DISABLE_POOL", None)

    if _ORIGINAL_HOME is not None:
        os.environ["HOME"] = _ORIGINAL_HOME
    else:
        os.environ.pop("HOME", None)

    if _ORIGINAL_EPISODIC_HOME is not None:
        os.environ["EPISODIC_HOME"] = _ORIGINAL_EPISODIC_HOME
    else:
        os.environ.pop("EPISODIC_HOME", None)

    shutil.rmtree(_TEST_HOME_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons/state before each test."""
    from episodic.config import config
    from episodic import rag
    from episodic.db_connection import close_pool
    from episodic.topics import strategy_registry

    def _reset_rag_system():
        if rag._rag_system is None:
            return
        try:
            rag._rag_system.clear_documents()
        except Exception:
            pass
        rag._rag_system = None

    config.config.clear()
    config.config.update(_BASE_CONFIG)
    _reset_rag_system()
    strategy_registry.reset_strategy()
    close_pool()
    db_path = os.environ.get("EPISODIC_DB_PATH")
    if db_path and os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    yield

    config.config.clear()
    config.config.update(_BASE_CONFIG)
    _reset_rag_system()
    strategy_registry.reset_strategy()
    close_pool()
    db_path = os.environ.get("EPISODIC_DB_PATH")
    if db_path and os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


@pytest.fixture
def temp_db_path():
    """Provide the session-scoped temporary database path."""
    return _TEST_DB_PATH


@pytest.fixture
def temp_database(temp_db_path):
    """
    Create and initialize a temporary database for testing.

    Yields the database path after initialization.
    Automatically cleans up after test completion.
    """
    try:
        # Ensure a clean database file for each test.
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        from episodic.db import initialize_db
        initialize_db()
        yield temp_db_path
    finally:
        try:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
        except Exception:
            pass


@pytest.fixture
def initialized_db(temp_database):
    """
    Provide a fully initialized database with connection.

    Returns tuple of (db_path, connection).
    """
    import sqlite3
    conn = sqlite3.connect(temp_database)
    conn.row_factory = sqlite3.Row
    yield temp_database, conn
    conn.close()


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def isolated_config():
    """
    Provide isolated configuration that doesn't affect global state.

    Saves current config, provides clean config for test,
    restores original after test completes.
    """
    from episodic.config import config

    # Save current config
    original_config = dict(config.config)

    # Reset to test defaults
    config.config.clear()
    config.config.update({
        'model': 'test-model',
        'debug': False,
        'show_cost': False,
        'automatic_topic_detection': True,
        'min_messages_before_topic_change': 8,
        'drift_threshold': 0.9,
    })

    try:
        yield config
    finally:
        config.config.clear()
        config.config.update(original_config)


@pytest.fixture
def debug_config(isolated_config):
    """Provide isolated config with debug mode enabled."""
    isolated_config.set('debug', True)
    return isolated_config


# =============================================================================
# LLM Mocking Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """
    Mock LLM responses for testing.

    Returns a factory function to create mock responses.
    """
    def create_mock(response="Test response", model="test-model",
                    prompt_tokens=10, completion_tokens=20):
        return {
            "choices": [{
                "message": {"content": response}
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "model": model
        }
    return create_mock


@pytest.fixture
def mock_llm(mock_llm_response):
    """
    Patch the LLM query function with mock responses.

    Use mock_llm.return_value to set the response.
    """
    mock = Mock(return_value=mock_llm_response())
    with patch('episodic.llm.query_llm', mock):
        yield mock


@pytest.fixture
def mock_streaming_response():
    """
    Mock streaming LLM responses for testing.

    Returns a generator that yields chunks.
    """
    def create_stream(chunks=None):
        if chunks is None:
            chunks = ["Hello", " ", "World", "!"]
        for chunk in chunks:
            yield {
                "choices": [{
                    "delta": {"content": chunk}
                }]
            }
    return create_stream


# =============================================================================
# RAG Fixtures
# =============================================================================

@pytest.fixture
def temp_rag_dir(tmp_path):
    """Provide a temporary directory for RAG data."""
    rag_dir = tmp_path / "rag_data"
    rag_dir.mkdir()
    return str(rag_dir)


@pytest.fixture
def mock_rag_system():
    """
    Mock the RAG system for testing.

    Returns a mock RAG manager with common methods stubbed.
    """
    mock_rag = MagicMock()
    mock_rag.search.return_value = []
    mock_rag.add_document.return_value = True
    mock_rag.is_initialized.return_value = True
    return mock_rag


@pytest.fixture
def mock_embeddings():
    """Mock embedding generation for fast testing."""
    import numpy as np

    def create_embedding(*args, **kwargs):
        # Return a consistent fake embedding
        return np.random.rand(384).tolist()

    with patch('episodic.ml.embeddings.get_embedding', create_embedding):
        yield create_embedding


# =============================================================================
# CLI Fixtures
# =============================================================================

@pytest.fixture
def cli_runner():
    """
    Provide a CLI test runner using typer's testing utilities.
    """
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def capture_output():
    """
    Capture stdout and stderr during test execution.

    Yields (stdout_capture, stderr_capture) StringIO objects.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def three_topics_conversation():
    """
    Provide test conversation with three distinct topics.

    Topics: Mars exploration, Italian cooking, Neural networks
    """
    from tests.fixtures.conversations import THREE_TOPICS_CONVERSATION
    return THREE_TOPICS_CONVERSATION


@pytest.fixture
def gradual_drift_conversation():
    """
    Provide test conversation with gradual topic drift.

    Drifts from general programming to advanced Python topics.
    """
    from tests.fixtures.conversations import GRADUAL_DRIFT_CONVERSATION
    return GRADUAL_DRIFT_CONVERSATION


@pytest.fixture
def single_topic_conversation():
    """
    Provide test conversation focused on single topic.

    Topic: Machine learning basics
    """
    from tests.fixtures.conversations import SINGLE_TOPIC_CONVERSATION
    return SINGLE_TOPIC_CONVERSATION


@pytest.fixture
def sample_messages():
    """Provide sample message pairs for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What would you like to know?"},
    ]


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers based on test location and name.

    - Tests in 'integration/' directory get @integration marker
    - Tests in 'unit/' directory get @unit marker
    - Tests with 'llm' in name get @llm marker
    - Tests with 'slow' in name get @slow marker
    """
    for item in items:
        # Add markers based on test location
        test_path = str(item.fspath)
        if "integration" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "unit" in test_path:
            item.add_marker(pytest.mark.unit)

        # Add markers based on test name
        if "llm" in item.name.lower():
            item.add_marker(pytest.mark.llm)
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        if "rag" in item.name.lower():
            item.add_marker(pytest.mark.rag)
        if "db" in item.name.lower() or "database" in item.name.lower():
            item.add_marker(pytest.mark.db)


def pytest_report_header(config):
    """Add custom header to pytest output."""
    return "Episodic Test Suite - pytest infrastructure"
