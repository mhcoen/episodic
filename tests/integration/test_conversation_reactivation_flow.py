"""
Integration tests for the full conversation flow with reactivation.

Tests the complete flow from topic creation through reactivation probe
to context assembly, verifying persistence and correctness.
"""

import sqlite3
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal

import pytest
import numpy as np

from episodic.context_recovery.imports import (
    detect_import_intent,
    resolve_import_target,
    ImportIntent,
)


def create_full_test_db() -> sqlite3.Connection:
    """Create in-memory database with all required tables for integration tests."""
    conn = sqlite3.connect(":memory:")

    # Core node table
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            role TEXT,
            parent_id TEXT,
            is_meta_query INTEGER DEFAULT 0
        )
    """)

    # Topics table
    conn.execute("""
        CREATE TABLE topics (
            name TEXT,
            start_node_id TEXT PRIMARY KEY,
            end_node_id TEXT
        )
    """)

    # Topic nodes junction table
    conn.execute("""
        CREATE TABLE topic_nodes (
            topic_start_node_id TEXT,
            node_id TEXT,
            turn_idx INTEGER,
            PRIMARY KEY (topic_start_node_id, node_id)
        )
    """)

    # Topic working set
    conn.execute("""
        CREATE TABLE topic_working_set (
            topic_start_node_id TEXT PRIMARY KEY,
            summary_md TEXT,
            last_summarized_turn_idx INTEGER,
            summary_version INTEGER DEFAULT 0,
            last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Topic centroids
    conn.execute("""
        CREATE TABLE topic_centroids (
            start_node_id TEXT PRIMARY KEY,
            centroid_medoid_exchange_id TEXT
        )
    """)

    # Reactivation decisions
    conn.execute("""
        CREATE TABLE reactivation_decisions (
            user_node_id TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            reason TEXT,
            confidence REAL,
            topic_name TEXT,
            topic_start_node_id TEXT,
            candidates_json TEXT NOT NULL DEFAULT '[]',
            support_counts_json TEXT NOT NULL DEFAULT '{}',
            gates_json TEXT NOT NULL DEFAULT '{"passed": [], "failed": []}',
            best_similarity REAL,
            best_support_count INTEGER,
            dormancy_turns INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Reactivation labels
    conn.execute("""
        CREATE TABLE reactivation_labels (
            user_node_id TEXT PRIMARY KEY,
            ground_truth TEXT NOT NULL,
            labeler TEXT,
            notes TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def setup_topic_with_exchanges(
    conn: sqlite3.Connection,
    topic_name: str,
    topic_start_id: str,
    exchanges: List[Dict[str, str]]
) -> None:
    """Create a topic with exchanges for testing."""
    # Create topic
    conn.execute("""
        INSERT INTO topics (name, start_node_id, end_node_id)
        VALUES (?, ?, NULL)
    """, (topic_name, topic_start_id))

    # Create working set
    conn.execute("""
        INSERT INTO topic_working_set (topic_start_node_id)
        VALUES (?)
    """, (topic_start_id,))

    # Create nodes and topic_nodes entries
    turn_idx = 0
    parent_id = None

    for exchange in exchanges:
        user_id = f"{topic_start_id}_u{turn_idx}"
        asst_id = f"{topic_start_id}_a{turn_idx}"

        # User node
        conn.execute("""
            INSERT INTO nodes (id, content, role, parent_id)
            VALUES (?, ?, 'user', ?)
        """, (user_id, exchange['user'], parent_id))

        conn.execute("""
            INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx)
            VALUES (?, ?, ?)
        """, (topic_start_id, user_id, turn_idx * 2))

        # Assistant node
        conn.execute("""
            INSERT INTO nodes (id, content, role, parent_id)
            VALUES (?, ?, 'assistant', ?)
        """, (asst_id, exchange['assistant'], user_id))

        conn.execute("""
            INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx)
            VALUES (?, ?, ?)
        """, (topic_start_id, asst_id, turn_idx * 2 + 1))

        parent_id = asst_id
        turn_idx += 1

    conn.commit()


class TestTopicCreationAndIsolation:
    """Tests for topic creation and isolation."""

    def test_create_separate_topics(self):
        """Test that separate topics are created correctly."""
        conn = create_full_test_db()

        # Create Topic A about Python
        setup_topic_with_exchanges(conn, "Python Programming", "topic_py", [
            {"user": "How do I create a list in Python?", "assistant": "Use square brackets: my_list = []"},
            {"user": "How do I add items?", "assistant": "Use append(): my_list.append('item')"},
            {"user": "What about dictionaries?", "assistant": "Use curly braces: my_dict = {}"},
        ])

        # Create Topic B about Coffee
        setup_topic_with_exchanges(conn, "Coffee Brewing", "topic_coffee", [
            {"user": "What's the best coffee brewing method?", "assistant": "Pour-over produces clean flavors."},
            {"user": "What grind size?", "assistant": "Medium-fine for pour-over, like table salt."},
        ])

        # Verify topics exist
        cursor = conn.execute("SELECT COUNT(*) FROM topics")
        assert cursor.fetchone()[0] == 2

        # Verify topic_nodes are isolated
        cursor = conn.execute("SELECT COUNT(*) FROM topic_nodes WHERE topic_start_node_id = ?", ("topic_py",))
        assert cursor.fetchone()[0] == 6  # 3 exchanges * 2 nodes each

        cursor = conn.execute("SELECT COUNT(*) FROM topic_nodes WHERE topic_start_node_id = ?", ("topic_coffee",))
        assert cursor.fetchone()[0] == 4  # 2 exchanges * 2 nodes each

    def test_topic_nodes_dont_cross_boundaries(self):
        """Test that topic_nodes don't cross topic boundaries."""
        conn = create_full_test_db()

        setup_topic_with_exchanges(conn, "Topic A", "topic_a", [
            {"user": "Question A1", "assistant": "Answer A1"},
        ])

        setup_topic_with_exchanges(conn, "Topic B", "topic_b", [
            {"user": "Question B1", "assistant": "Answer B1"},
        ])

        # Get all node IDs for each topic
        cursor = conn.execute("""
            SELECT node_id FROM topic_nodes WHERE topic_start_node_id = ?
        """, ("topic_a",))
        topic_a_nodes = {row[0] for row in cursor.fetchall()}

        cursor = conn.execute("""
            SELECT node_id FROM topic_nodes WHERE topic_start_node_id = ?
        """, ("topic_b",))
        topic_b_nodes = {row[0] for row in cursor.fetchall()}

        # No overlap
        assert topic_a_nodes.isdisjoint(topic_b_nodes)


class TestImportIntentWithTopics:
    """Tests for import intent detection with real topic data."""

    def test_import_intent_resolves_to_correct_topic(self):
        """Test that import intent resolves to the correct topic."""
        conn = create_full_test_db()

        # Create topics
        setup_topic_with_exchanges(conn, "Python Programming", "topic_py", [
            {"user": "How do I use Python?", "assistant": "Python is a versatile language."},
        ])

        setup_topic_with_exchanges(conn, "Machine Learning", "topic_ml", [
            {"user": "What is ML?", "assistant": "ML is a subset of AI."},
        ])

        # Test intent detection
        intent = detect_import_intent("as we discussed about Python earlier")
        assert intent.has_intent is True
        assert "python" in intent.topic_reference.lower()

        # Test resolution - currently in ML topic, want to import from Python
        target = resolve_import_target(
            topic_reference=intent.topic_reference,
            active_topic_start_node_id="topic_ml",
            user_embedding=None,
            conn=conn
        )

        assert target is not None
        assert target.topic_start_node_id == "topic_py"
        assert target.topic_name == "Python Programming"

    def test_import_intent_excludes_active_topic(self):
        """Test that active topic is excluded from import resolution."""
        conn = create_full_test_db()

        setup_topic_with_exchanges(conn, "Python", "topic_py", [
            {"user": "Python question", "assistant": "Python answer"},
        ])

        # Try to import Python while already in Python topic
        intent = detect_import_intent("as we discussed about Python")
        target = resolve_import_target(
            topic_reference=intent.topic_reference,
            active_topic_start_node_id="topic_py",  # Same topic
            user_embedding=None,
            conn=conn
        )

        # Should not find target (can't import from self)
        assert target is None


class TestReactivationDecisionFlow:
    """Tests for the reactivation decision persistence flow."""

    def test_decision_persisted_after_probe(self):
        """Test that reactivation decisions are persisted."""
        from episodic.db_reactivation_decisions import (
            persist_reactivation_decision,
            get_reactivation_decision,
        )
        from episodic.recall.reactivation import ReactivationDecision

        conn = create_full_test_db()

        # Create decision
        decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="Python Programming",
            topic_start_node_id="topic_py",
            debug={
                "confidence": 0.85,
                "candidates": [{"topic": "Python Programming", "sim": 0.85, "rank": 1}],
                "support_counts": {"Python Programming": 5},
                "gates_passed": ["dormancy", "support"],
                "gates_failed": [],
                "best_similarity": 0.85,
                "best_support_count": 5,
                "dormancy_turns": 10,
            }
        )

        # Persist - pass conn directly, no patching needed
        result = persist_reactivation_decision("user_node_123", decision, conn)

        assert result is True

        # Retrieve and verify
        retrieved = get_reactivation_decision("user_node_123", conn)

        assert retrieved is not None
        assert retrieved['decision'] == "REACTIVATE"
        assert retrieved['topic_name'] == "Python Programming"
        assert retrieved['confidence'] == 0.85
        assert len(retrieved['candidates']) == 1
        assert retrieved['best_similarity'] == 0.85

    def test_decision_can_be_labeled(self):
        """Test that persisted decisions can be labeled."""
        from episodic.db_reactivation_decisions import (
            persist_reactivation_decision,
            store_reactivation_label,
            get_labeled_decisions,
        )
        from episodic.recall.reactivation import ReactivationDecision

        conn = create_full_test_db()

        # Create and persist decision
        decision = ReactivationDecision(
            action="CONTINUE",
            debug={"confidence": 0.3, "exit_reason": "no_strong_match"}
        )

        # Pass conn directly
        persist_reactivation_decision("user_node_456", decision, conn)

        # Label it (human reviewer says it should have reactivated)
        store_reactivation_label(
            user_node_id="user_node_456",
            ground_truth="reactivate:Python",
            labeler="reviewer",
            notes="Missed resume - user was returning to Python topic",
            conn=conn
        )

        # Verify labeled decision
        labeled = get_labeled_decisions(conn)

        assert len(labeled) == 1
        assert labeled[0]['decision'] == "CONTINUE"
        assert labeled[0]['ground_truth'] == "reactivate:Python"
        assert labeled[0]['labeler'] == "reviewer"
        assert "Missed resume" in labeled[0]['notes']


class TestReplayIntegration:
    """Tests for replay harness with real data structures."""

    def test_replay_with_labeled_data(self):
        """Test replay harness with labeled decisions."""
        from episodic.db_reactivation_decisions import (
            persist_reactivation_decision,
            store_reactivation_label,
        )
        from episodic.evaluation.reactivation_replay import (
            ReplayResult,
            compute_metrics,
        )
        from episodic.recall.reactivation import ReactivationDecision

        conn = create_full_test_db()

        # Create multiple decisions with labels
        test_cases = [
            # True positive
            ("node_1", "REACTIVATE", "Python", "reactivate:Python", True),
            # True negative
            ("node_2", "CONTINUE", None, "continue", True),
            # False positive
            ("node_3", "REACTIVATE", "Python", "continue", False),
            # False negative
            ("node_4", "CONTINUE", None, "reactivate:Python", False),
        ]

        for node_id, action, topic, ground_truth, correct in test_cases:
            decision = ReactivationDecision(
                action=action,
                topic_name=topic,
                debug={"confidence": 0.7}
            )

            # Pass conn directly
            persist_reactivation_decision(node_id, decision, conn)

            store_reactivation_label(node_id, ground_truth, conn=conn)

        # Build replay results manually (simulating what replay_conversation would do)
        results = [
            ReplayResult("node_1", "", "reactivate:Python", "REACTIVATE", "Python", True),
            ReplayResult("node_2", "", "continue", "CONTINUE", None, True),
            ReplayResult("node_3", "", "continue", "REACTIVATE", "Python", False),
            ReplayResult("node_4", "", "reactivate:Python", "CONTINUE", None, False),
        ]

        # Compute metrics
        metrics = compute_metrics(results)

        assert metrics.total == 4
        assert metrics.correct == 2
        assert metrics.accuracy == 0.5
        assert metrics.true_positives == 1
        assert metrics.false_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_negatives == 1
        assert metrics.precision == 0.5  # 1 / (1 + 1)
        assert metrics.recall == 0.5     # 1 / (1 + 1)


class TestContextAssemblyWithImports:
    """Tests for context assembly with cross-topic imports."""

    def test_import_context_includes_topic_summary(self):
        """Test that imported context includes the source topic's summary."""
        from episodic.context_recovery.imports import fetch_import_context

        conn = create_full_test_db()

        # Create topic with summary
        setup_topic_with_exchanges(conn, "Python Programming", "topic_py", [
            {"user": "Python question", "assistant": "Python answer"},
        ])

        # Update summary
        conn.execute("""
            UPDATE topic_working_set SET summary_md = ?
            WHERE topic_start_node_id = ?
        """, ("We discussed Python basics including lists and dictionaries.", "topic_py"))
        conn.commit()

        # Fetch import context
        with patch('episodic.context_recovery.imports._get_import_anchors', return_value=""):
            context = fetch_import_context(
                source_topic_start_node_id="topic_py",
                user_input="Tell me about Python lists again",
                user_embedding=None,
                token_budget=500,
                conn=conn
            )

        assert "[Imported from: Python Programming]" in context.context_block
        assert "Python basics" in context.context_block
        assert context.debug['summary_included'] is True

    def test_full_import_flow_with_context_assembly(self):
        """Test the full flow: detect intent -> resolve -> fetch context."""
        from episodic.context_recovery.imports import (
            detect_import_intent,
            resolve_import_target,
            fetch_import_context,
        )

        conn = create_full_test_db()

        # Setup Python topic with summary
        setup_topic_with_exchanges(conn, "Python Programming", "topic_py", [
            {"user": "How do lists work?", "assistant": "Lists are ordered collections."},
        ])
        conn.execute("""
            UPDATE topic_working_set SET summary_md = ?
            WHERE topic_start_node_id = ?
        """, ("Covered Python list operations and indexing.", "topic_py"))

        # Setup Coffee topic (current context)
        setup_topic_with_exchanges(conn, "Coffee", "topic_coffee", [
            {"user": "Best beans?", "assistant": "Try Ethiopian Yirgacheffe."},
        ])
        conn.commit()

        # User in coffee topic references Python
        user_input = "Going back to what you said about Python, can you explain more?"

        # 1. Detect intent
        intent = detect_import_intent(user_input)
        assert intent.has_intent is True

        # 2. Resolve target
        target = resolve_import_target(
            topic_reference=intent.topic_reference,
            active_topic_start_node_id="topic_coffee",
            user_embedding=None,
            conn=conn
        )
        assert target is not None
        assert target.topic_start_node_id == "topic_py"

        # 3. Fetch import context
        with patch('episodic.context_recovery.imports._get_import_anchors', return_value=""):
            context = fetch_import_context(
                source_topic_start_node_id=target.topic_start_node_id,
                user_input=user_input,
                user_embedding=None,
                token_budget=300,
                conn=conn
            )

        # Verify the assembled import block
        assert "[Imported from: Python Programming]" in context.context_block
        assert "list operations" in context.context_block
        assert context.topic_name == "Python Programming"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_database(self):
        """Test behavior with empty database."""
        conn = create_full_test_db()

        # Try to resolve import with no topics
        target = resolve_import_target(
            topic_reference="Python",
            active_topic_start_node_id=None,
            user_embedding=None,
            conn=conn
        )

        assert target is None

    def test_single_topic(self):
        """Test with only one topic (no import target possible)."""
        conn = create_full_test_db()

        setup_topic_with_exchanges(conn, "Only Topic", "topic_only", [
            {"user": "Question", "assistant": "Answer"},
        ])

        # Can't import from self
        target = resolve_import_target(
            topic_reference="Only Topic",
            active_topic_start_node_id="topic_only",
            user_embedding=None,
            conn=conn
        )

        assert target is None

    def test_nonexistent_topic_fetch(self):
        """Test fetching context from nonexistent topic."""
        from episodic.context_recovery.imports import fetch_import_context

        conn = create_full_test_db()

        context = fetch_import_context(
            source_topic_start_node_id="nonexistent",
            user_input="test",
            user_embedding=None,
            token_budget=100,
            conn=conn
        )

        assert context.context_block == ""
        assert context.debug['error'] == 'topic_not_found'
