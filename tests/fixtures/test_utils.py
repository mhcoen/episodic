"""
Common test utilities and helpers.

Provides utilities for setting up test environments, mocking,
and common test operations.
"""

import os
from contextlib import contextmanager
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional
import sqlite3


@contextmanager
def temp_database():
    """Create a temporary database for testing."""
    db_path = os.environ["EPISODIC_DB_PATH"]

    try:
        # Initialize the database
        from episodic.db import initialize_db
        if os.path.exists(db_path):
            os.remove(db_path)
        initialize_db()
        yield db_path
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception:
            pass


def mock_llm_response(response: str = "Test response", model: str = "test-model"):
    """Mock LLM responses for testing."""
    cost_info = {
        "model": model,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
    return response, cost_info


@contextmanager
def isolated_config():
    """Create an isolated configuration for testing."""
    from episodic.config import config
    
    # Save current config
    original_config = dict(config.config)

    # Reset to defaults
    config.config.clear()
    config.config.update({
        'model': 'test-model',
        'debug': False,
        'show_cost': False,
        'automatic_topic_detection': True,
        'min_messages_before_topic_change': 8
    })

    try:
        yield config
    finally:
        # Restore original config
        config.config.clear()
        config.config.update(original_config)


def create_test_topics(count: int = 3) -> List[Dict]:
    """Create test topic entries."""
    topics = []
    for i in range(count):
        topics.append({
            'id': f'topic-{i}',
            'name': f'Test Topic {i}',
            'start_node_id': f'node-{i*10}',
            'end_node_id': f'node-{i*10 + 9}' if i < count - 1 else None,
            'created_at': f'2024-01-0{i+1}T00:00:00',
            'message_count': 10
        })
    return topics


def insert_test_nodes(db_path: str, nodes: List[Dict]):
    """Insert test nodes directly into database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    seen_short_ids = set()
    
    for idx, node in enumerate(nodes):
        content = node.get('content', node.get('message', ''))
        short_id = node.get('short_id')
        if not short_id or short_id in seen_short_ids:
            short_id = f"t{idx}"
        seen_short_ids.add(short_id)
        cursor.execute("""
            INSERT INTO nodes (
                id, short_id, parent_id, content, role, provider, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            node['id'],
            short_id,
            node.get('parent_id'),
            content,
            node.get('role', 'user'),
            node.get('provider'),
            node.get('model', node.get('model_name', 'test-model'))
        ))
    
    conn.commit()
    conn.close()


def assert_command_output(output: str, expected_patterns: List[str]):
    """Assert that output contains expected patterns."""
    for pattern in expected_patterns:
        assert pattern in output, f"Expected '{pattern}' in output:\n{output}"


def mock_topic_detection(should_detect: bool = True, confidence: float = 0.8):
    """Create a mock for topic detection."""
    return Mock(
        return_value={
            'topic_changed': should_detect,
            'confidence': confidence,
            'reason': 'Test detection',
            'new_topic_hint': 'New Test Topic' if should_detect else None
        }
    )


class TestOutputStream:
    """Capture output for testing."""
    
    def __init__(self):
        self.lines = []
        
    def write(self, text):
        self.lines.append(text)
        
    def get_output(self):
        return ''.join(self.lines)
        
    def clear(self):
        self.lines.clear()


@contextmanager
def capture_cli_output():
    """Capture CLI output for testing."""
    import sys
    from io import StringIO
    
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
