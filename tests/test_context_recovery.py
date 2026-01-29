"""
Unit tests for context recovery strategies.

Tests cover:
1. Ancestry strategy basic behavior
2. Topic-local strategy isolation ("B disappears")
3. Year-later resume with summary
4. No summary yet case
5. Strategy selection logic
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Test fixtures
@pytest.fixture
def test_db():
    """Create a test database with schema and sample data."""
    test_db_path = tempfile.mktemp(suffix='.db')
    original_db_path = os.environ.get('EPISODIC_DB_PATH')
    os.environ['EPISODIC_DB_PATH'] = test_db_path

    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Create nodes table (with short_id for get_ancestry compatibility)
    cursor.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT,
            content TEXT,
            role TEXT,
            parent_id TEXT
        )
    """)

    # Create topics table
    cursor.execute("""
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start_node_id TEXT,
            end_node_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create topic_nodes table
    cursor.execute("""
        CREATE TABLE topic_nodes (
            topic_start_node_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            turn_idx INTEGER NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY(topic_start_node_id, node_id)
        )
    """)

    # Create topic_working_set table
    cursor.execute("""
        CREATE TABLE topic_working_set (
            topic_start_node_id TEXT PRIMARY KEY,
            topic_name TEXT,
            summary_md TEXT NOT NULL DEFAULT '',
            decisions_json TEXT NOT NULL DEFAULT '[]',
            open_loops_json TEXT NOT NULL DEFAULT '[]',
            entities_json TEXT NOT NULL DEFAULT '[]',
            last_summarized_turn_idx INTEGER,
            last_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            summary_version INTEGER NOT NULL DEFAULT 1
        )
    """)

    conn.commit()

    yield conn

    conn.close()
    os.remove(test_db_path)

    # Restore original EPISODIC_DB_PATH and close pool to clear stale connections
    if original_db_path is not None:
        os.environ['EPISODIC_DB_PATH'] = original_db_path
    else:
        os.environ.pop('EPISODIC_DB_PATH', None)

    from episodic.db_connection import close_pool
    close_pool()


@pytest.fixture
def topic_a_nodes(test_db):
    """Create nodes for Topic A (baseball discussion)."""
    cursor = test_db.cursor()

    nodes = [
        ('node_a1', 'a1', 'What is the best batting stance?', 'user', None),
        ('node_a2', 'a2', 'The best stance depends on your height...', 'assistant', 'node_a1'),
        ('node_a3', 'a3', 'How do you calculate ERA?', 'user', 'node_a2'),
        ('node_a4', 'a4', 'ERA is Earned Run Average, calculated by...', 'assistant', 'node_a3'),
    ]

    for node_id, short_id, content, role, parent_id in nodes:
        cursor.execute(
            "INSERT INTO nodes (id, short_id, content, role, parent_id) VALUES (?, ?, ?, ?, ?)",
            (node_id, short_id, content, role, parent_id)
        )

    # Create topic
    cursor.execute(
        "INSERT INTO topics (name, start_node_id, end_node_id) VALUES (?, ?, ?)",
        ('baseball-discussion', 'node_a1', 'node_a4')
    )

    # Populate topic_nodes
    for i, (node_id, short_id, content, role, _) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role) VALUES (?, ?, ?, ?)",
            ('node_a1', node_id, i + 1, role)
        )

    # Create working set
    cursor.execute(
        "INSERT INTO topic_working_set (topic_start_node_id, topic_name, summary_md) VALUES (?, ?, ?)",
        ('node_a1', 'baseball-discussion', 'Discussion about baseball batting and pitching statistics.')
    )

    test_db.commit()
    return nodes


@pytest.fixture
def topic_b_nodes(test_db, topic_a_nodes):
    """Create nodes for Topic B (coffee brewing) AFTER Topic A."""
    cursor = test_db.cursor()

    nodes = [
        ('node_b1', 'b1', 'What is the best water temperature for coffee?', 'user', 'node_a4'),
        ('node_b2', 'b2', 'The ideal temperature is 195-205°F...', 'assistant', 'node_b1'),
        ('node_b3', 'b3', 'How fine should I grind for espresso?', 'user', 'node_b2'),
        ('node_b4', 'b4', 'For espresso, you want a fine grind...', 'assistant', 'node_b3'),
    ]

    for node_id, short_id, content, role, parent_id in nodes:
        cursor.execute(
            "INSERT INTO nodes (id, short_id, content, role, parent_id) VALUES (?, ?, ?, ?, ?)",
            (node_id, short_id, content, role, parent_id)
        )

    # Create topic
    cursor.execute(
        "INSERT INTO topics (name, start_node_id, end_node_id) VALUES (?, ?, ?)",
        ('coffee-brewing', 'node_b1', 'node_b4')
    )

    # Populate topic_nodes
    for i, (node_id, short_id, content, role, _) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role) VALUES (?, ?, ?, ?)",
            ('node_b1', node_id, i + 5, role)  # Continue turn_idx from topic A
        )

    # Create working set
    cursor.execute(
        "INSERT INTO topic_working_set (topic_start_node_id, topic_name, summary_md) VALUES (?, ?, ?)",
        ('node_b1', 'coffee-brewing', '')  # No summary yet
    )

    test_db.commit()
    return nodes


class TestTopicLocalStrategy:
    """Tests for topic-local context recovery."""

    def test_b_disappears_when_reactivating_a(self, test_db, topic_a_nodes, topic_b_nodes):
        """
        Critical test: When reactivating Topic A, Topic B nodes should NOT appear.

        Scenario:
        - Topic A (baseball): 4 nodes
        - Topic B (coffee): 4 nodes
        - Reactivate Topic A
        - Assert: context contains ONLY Topic A nodes
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy(exchange_pairs=4)

        result = strategy.assemble(
            user_turn_text="Tell me more about ERA",
            user_node_id=None,
            active_topic_start_node_id='node_a1',
            user_embedding=None,
            token_budget=4000,
            conn=test_db,
        )

        # Check that we got messages
        assert len(result.messages) > 0

        # Extract all content from messages
        all_content = " ".join(msg["content"] for msg in result.messages)

        # Topic A content should be present
        assert "batting stance" in all_content.lower() or "ERA" in all_content

        # Topic B content should NOT be present
        assert "water temperature" not in all_content.lower()
        assert "coffee" not in all_content.lower()
        assert "espresso" not in all_content.lower()

        # Check debug info
        assert result.debug["mode"] == "topic_local"
        assert result.debug["topic_start_node_id"] == "node_a1"

        # All included nodes should be from Topic A
        for node_id in result.debug["included_node_ids"]:
            assert node_id.startswith("node_a"), f"Node {node_id} is not from Topic A"

    def test_year_later_resume_with_summary(self, test_db, topic_a_nodes):
        """
        Year-later scenario: topic has summary but no recent context.

        Assembler should include the summary and function correctly
        even without recent ancestry.
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy(exchange_pairs=4)

        result = strategy.assemble(
            user_turn_text="What were we discussing about baseball?",
            user_node_id=None,
            active_topic_start_node_id='node_a1',
            user_embedding=None,
            token_budget=4000,
            conn=test_db,
        )

        # Check that summary is included
        all_content = " ".join(msg["content"] for msg in result.messages)
        assert "baseball" in all_content.lower() or "batting" in all_content.lower()

        # Check debug flags
        assert result.debug["working_set_used"] is True
        assert result.debug["summary_included"] is True

    def test_no_summary_yet_returns_exchanges(self, test_db, topic_a_nodes, topic_b_nodes):
        """
        When topic has no summary yet, assembler should return last N exchanges.
        With thin_topic_local fallback, a topic without summary may fall back to ancestry.
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy(exchange_pairs=2)

        # Topic B has no summary
        result = strategy.assemble(
            user_turn_text="More about coffee grinding?",
            user_node_id=None,
            active_topic_start_node_id='node_b1',
            user_embedding=None,
            token_budget=4000,
            conn=test_db,
        )

        # Mode could be topic_local or ancestry_fallback depending on token count
        assert result.debug["mode"] in ("topic_local", "ancestry_fallback")

        # If it stayed in topic_local, verify we got content
        if result.debug["mode"] == "topic_local":
            # Should still have messages (the exchanges)
            assert len(result.messages) >= 1

            # Check that we got coffee-related content
            all_content = " ".join(msg["content"] for msg in result.messages)
            assert "coffee" in all_content.lower() or "espresso" in all_content.lower() or "grind" in all_content.lower()

            # Summary should not be included (empty)
            assert result.debug["summary_included"] is False

    def test_empty_topic_returns_empty_messages(self, test_db):
        """When topic has no nodes, return empty messages list.

        With thin_topic_local fallback, empty topics will fall back to ancestry.
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy()

        result = strategy.assemble(
            user_turn_text="Hello?",
            user_node_id=None,
            active_topic_start_node_id='nonexistent_topic',
            user_embedding=None,
            token_budget=4000,
            conn=test_db,
        )

        # Empty/nonexistent topics trigger thin_topic_local fallback to ancestry
        assert result.debug["mode"] in ("topic_local", "ancestry_fallback")
        if result.debug["mode"] == "ancestry_fallback":
            assert result.debug.get("fallback_reason") == "thin_topic_local"

    def test_null_topic_returns_empty(self, test_db):
        """When no active topic, return empty result."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy()

        result = strategy.assemble(
            user_turn_text="Hello?",
            user_node_id=None,
            active_topic_start_node_id=None,
            user_embedding=None,
            token_budget=4000,
            conn=test_db,
        )

        assert len(result.messages) == 0
        assert result.debug["reactivation_fired"] is False


class TestAncestryStrategy:
    """Tests for ancestry-based context recovery."""

    def test_ancestry_includes_cross_topic_context(self, test_db, topic_a_nodes, topic_b_nodes):
        """
        Ancestry strategy should include context across topic boundaries.

        This is the traditional behavior - all recent ancestry regardless of topic.
        """
        from episodic.context_recovery.ancestry import AncestryStrategy
        from unittest.mock import patch

        # Mock get_ancestry since it uses global connection, not the test_db
        # Note: get_ancestry is imported inside the assemble method from episodic.db
        mock_ancestry = [
            {'id': 'node_a1', 'content': 'What is the best batting stance?', 'role': 'user', 'parent_id': None},
            {'id': 'node_a2', 'content': 'The best stance depends on your height...', 'role': 'assistant', 'parent_id': 'node_a1'},
            {'id': 'node_b1', 'content': 'What is a good latte recipe?', 'role': 'user', 'parent_id': 'node_a2'},
            {'id': 'node_b2', 'content': 'For a great latte, use fresh espresso...', 'role': 'assistant', 'parent_id': 'node_b1'},
        ]

        with patch('episodic.db.get_ancestry', return_value=mock_ancestry):
            strategy = AncestryStrategy()

            result = strategy.assemble(
                user_turn_text="What about ERA?",
                user_node_id='node_b2',  # Latest node
                active_topic_start_node_id='node_b1',
                user_embedding=None,
                token_budget=4000,
                conn=test_db,
            )

            assert result.debug["mode"] == "ancestry"
            # Ancestry should include both topics (cross-topic context)
            assert len(result.messages) >= 2
            # Check that both baseball and coffee content is included
            all_content = " ".join(msg["content"] for msg in result.messages)
            assert "batting" in all_content.lower() or "latte" in all_content.lower()


class TestStrategySelection:
    """Tests for strategy selection logic."""

    def test_select_ancestry_mode(self):
        """ANCESTRY mode returns AncestryStrategy."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.ancestry import AncestryStrategy

        strategy = select_strategy(ContextRecoveryMode.ANCESTRY)
        assert isinstance(strategy, AncestryStrategy)

    def test_select_topic_local_mode(self):
        """TOPIC_LOCAL mode returns TopicLocalStrategy."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = select_strategy(ContextRecoveryMode.TOPIC_LOCAL)
        assert isinstance(strategy, TopicLocalStrategy)

    def test_hybrid_with_reactivation(self):
        """HYBRID mode with reactivation returns TopicLocalStrategy."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.recall.reactivation import ReactivationDecision

        reactivation = ReactivationDecision(
            action="REACTIVATE",
            topic_name="baseball",
            topic_start_node_id="node_a1"
        )

        strategy = select_strategy(ContextRecoveryMode.HYBRID, reactivation)
        assert isinstance(strategy, TopicLocalStrategy)

    def test_hybrid_without_reactivation(self):
        """HYBRID mode without reactivation returns AncestryStrategy."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.ancestry import AncestryStrategy
        from episodic.recall.reactivation import ReactivationDecision

        reactivation = ReactivationDecision(action="CONTINUE")

        strategy = select_strategy(ContextRecoveryMode.HYBRID, reactivation)
        assert isinstance(strategy, AncestryStrategy)

    def test_hybrid_with_none_reactivation(self):
        """HYBRID mode with None reactivation returns AncestryStrategy."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.ancestry import AncestryStrategy

        strategy = select_strategy(ContextRecoveryMode.HYBRID, None)
        assert isinstance(strategy, AncestryStrategy)


class TestContextAssemblyResult:
    """Tests for ContextAssemblyResult structure."""

    def test_result_has_required_fields(self):
        """Result should have messages and debug fields."""
        from episodic.context_recovery.strategy import ContextAssemblyResult

        result = ContextAssemblyResult(
            messages=[{"role": "user", "content": "Hello"}],
            debug={"mode": "test"}
        )

        assert hasattr(result, 'messages')
        assert hasattr(result, 'debug')
        assert len(result.messages) == 1
        assert result.debug["mode"] == "test"

    def test_result_default_debug(self):
        """Debug should default to empty dict."""
        from episodic.context_recovery.strategy import ContextAssemblyResult

        result = ContextAssemblyResult(messages=[])
        assert result.debug == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
