"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides
shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test fixtures
from tests.fixtures.test_utils import (
    temp_database,
    mock_llm_response,
    isolated_config,
    capture_cli_output
)
from tests.fixtures.conversations import (
    THREE_TOPICS_CONVERSATION,
    GRADUAL_DRIFT_CONVERSATION,
    SINGLE_TOPIC_CONVERSATION,
    create_test_conversation,
    create_test_node
)


# Make fixtures available to all tests
@pytest.fixture
def test_db():
    """Provide a temporary test database."""
    with temp_database() as db_path:
        yield db_path


@pytest.fixture
def test_config():
    """Provide isolated test configuration."""
    with isolated_config() as config:
        yield config


@pytest.fixture
def mock_llm():
    """Provide mock LLM responses."""
    with mock_llm_response() as mock:
        yield mock


@pytest.fixture
def three_topics_conversation():
    """Provide three-topic test conversation."""
    return THREE_TOPICS_CONVERSATION


@pytest.fixture
def gradual_drift_conversation():
    """Provide gradual drift test conversation."""
    return GRADUAL_DRIFT_CONVERSATION


@pytest.fixture
def single_topic_conversation():
    """Provide single topic test conversation."""
    return SINGLE_TOPIC_CONVERSATION


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM mocking"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            
        # Add markers based on test name
        if "llm" in item.name.lower():
            item.add_marker(pytest.mark.llm)
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)