"""
Integration test specific fixtures and configuration.

Integration tests may:
- Use real database connections
- Take longer to run
- Test component interactions
- Require setup/teardown of resources
"""

import pytest
import os
import tempfile
import shutil


@pytest.fixture(scope="session")
def integration_db_dir():
    """
    Provide a session-scoped temporary directory for integration test databases.

    This directory persists across all integration tests in a session,
    allowing tests to share database state when appropriate.
    """
    temp_dir = tempfile.mkdtemp(prefix="episodic_integration_")
    yield temp_dir
    # Cleanup after all integration tests complete
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def integration_db(integration_db_dir):
    """
    Provide a fresh database for each integration test.

    Each test gets its own database file to avoid interference.
    """
    import sqlite3

    # Use the session-scoped test DB path; path is resolved once at startup.
    db_path = os.environ['EPISODIC_DB_PATH']

    try:
        # Ensure a clean database for each test.
        if os.path.exists(db_path):
            os.remove(db_path)
        from episodic.db import initialize_db
        initialize_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        yield db_path, conn
        conn.close()
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


@pytest.fixture
def integration_config(isolated_config):
    """
    Provide configuration suitable for integration testing.

    Extends isolated_config with integration-specific defaults.
    """
    isolated_config.set('model', 'gpt-3.5-turbo')
    isolated_config.set('automatic_topic_detection', True)
    return isolated_config


@pytest.fixture
def real_llm_available():
    """
    Check if real LLM API is available for integration tests.

    Skip tests that require real LLM if no API key is configured.
    """
    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        pytest.skip("No LLM API key available for integration test")
    return True


@pytest.fixture
def rag_integration_dir(integration_db_dir, request):
    """
    Provide a directory for RAG integration test data.

    Each test gets its own RAG directory.
    """
    test_name = request.node.name.replace("/", "_").replace(":", "_")
    rag_dir = os.path.join(integration_db_dir, f"rag_{test_name}")
    os.makedirs(rag_dir, exist_ok=True)
    return rag_dir
