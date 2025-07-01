"""
Common test utilities and helpers.

Provides utilities for setting up test environments, mocking,
and common test operations.
"""

import os
import tempfile
import shutil
from contextlib import contextmanager
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional
import sqlite3


@contextmanager
def temp_database():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_episodic.db')
    
    # Set the environment variable
    old_path = os.environ.get('EPISODIC_DB_PATH')
    os.environ['EPISODIC_DB_PATH'] = db_path
    
    try:
        # Initialize the database
        from episodic.db import initialize_db
        initialize_db()
        yield db_path
    finally:
        # Restore environment and cleanup
        if old_path:
            os.environ['EPISODIC_DB_PATH'] = old_path
        else:
            os.environ.pop('EPISODIC_DB_PATH', None)
        shutil.rmtree(temp_dir)


@contextmanager
def mock_llm_response(response: str = "Test response", model: str = "test-model"):
    """Mock LLM responses for testing."""
    mock_response = {
        "choices": [{
            "message": {"content": response}
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        },
        "model": model
    }
    
    with patch('episodic.llm.query_llm', return_value=mock_response):
        yield mock_response


@contextmanager
def isolated_config():
    """Create an isolated configuration for testing."""
    from episodic.config import config
    
    # Save current config
    original_config = dict(config._config)
    
    # Reset to defaults
    config._config.clear()
    config._config.update({
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
        config._config.clear()
        config._config.update(original_config)


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
    
    for node in nodes:
        cursor.execute("""
            INSERT INTO nodes (
                id, short_id, message, role, parent_id,
                timestamp, model_name, system_prompt, response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node['id'], node['short_id'], node['message'],
            node.get('role', 'user'), node.get('parent_id'),
            node['timestamp'], node.get('model_name', 'test-model'),
            node.get('system_prompt', 'Test prompt'),
            node.get('response')
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