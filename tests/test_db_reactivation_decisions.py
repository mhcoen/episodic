"""
Tests for reactivation decision persistence functionality.

Tests save_reactivation_decision(), get_reactivation_decision(),
store_reactivation_label(), and related functions.
"""

import sqlite3
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal

import pytest

# Mark all tests in this module for the reactivation CI gate
pytestmark = pytest.mark.reactivation


@dataclass
class MockReactivationDecision:
    """Mock ReactivationDecision for testing without importing full module."""
    action: Literal["CONTINUE", "REACTIVATE", "DISAMBIGUATE"]
    topic_name: Optional[str] = None
    topic_start_node_id: Optional[str] = None
    options: Optional[List] = None
    debug: Dict[str, Any] = field(default_factory=dict)


def create_test_db() -> sqlite3.Connection:
    """Create in-memory database with required tables."""
    conn = sqlite3.connect(":memory:")

    # Create reactivation_decisions table
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

    # Create reactivation_labels table
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


class TestPersistReactivationDecision:
    """Tests for persist_reactivation_decision()."""

    def test_persist_continue_decision(self):
        """Test persisting a CONTINUE decision."""
        from episodic.db_reactivation_decisions import persist_reactivation_decision

        conn = create_test_db()

        decision = MockReactivationDecision(
            action="CONTINUE",
            debug={
                "exit_reason": "no_candidates",
                "confidence": 0.1,
                "candidates": [],
                "support_counts": {},
                "gates_passed": ["dormancy"],
                "gates_failed": [],
            }
        )

        # Pass connection directly - no need to patch context managers
        result = persist_reactivation_decision("node_123", decision, conn)

        assert result is True

        # Verify stored
        cursor = conn.execute("SELECT * FROM reactivation_decisions WHERE user_node_id = ?", ("node_123",))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "CONTINUE"

    def test_persist_reactivate_decision(self):
        """Test persisting a REACTIVATE decision."""
        from episodic.db_reactivation_decisions import persist_reactivation_decision

        conn = create_test_db()

        decision = MockReactivationDecision(
            action="REACTIVATE",
            topic_name="Python Programming",
            topic_start_node_id="topic_py",
            debug={
                "best_topic": "Python Programming",
                "confidence": 0.85,
                "candidates": [
                    {"topic": "Python Programming", "sim": 0.8, "rank": 1, "dormancy": 5}
                ],
                "support_counts": {"Python Programming": 3},
                "gates_passed": ["dormancy", "support", "rank_gap"],
                "gates_failed": [],
                "best_similarity": 0.8,
                "best_support_count": 3,
                "dormancy_turns": 5,
            }
        )

        # Pass connection directly
        result = persist_reactivation_decision("node_456", decision, conn)

        assert result is True

        # Verify stored
        cursor = conn.execute("SELECT * FROM reactivation_decisions WHERE user_node_id = ?", ("node_456",))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "REACTIVATE"
        assert row[4] == "Python Programming"
        assert row[5] == "topic_py"

    def test_persist_overwrites_existing(self):
        """Test that persisting overwrites existing decision."""
        from episodic.db_reactivation_decisions import persist_reactivation_decision

        conn = create_test_db()

        # Insert first decision
        decision1 = MockReactivationDecision(action="CONTINUE", debug={"confidence": 0.5})
        persist_reactivation_decision("node_123", decision1, conn)

        # Insert replacement decision
        decision2 = MockReactivationDecision(
            action="REACTIVATE",
            topic_name="New Topic",
            debug={"confidence": 0.9}
        )
        persist_reactivation_decision("node_123", decision2, conn)

        # Verify only one row exists with updated values
        cursor = conn.execute("SELECT COUNT(*) FROM reactivation_decisions WHERE user_node_id = ?", ("node_123",))
        assert cursor.fetchone()[0] == 1

        cursor = conn.execute("SELECT decision, topic_name FROM reactivation_decisions WHERE user_node_id = ?", ("node_123",))
        row = cursor.fetchone()
        assert row[0] == "REACTIVATE"
        assert row[1] == "New Topic"


class TestGetReactivationDecision:
    """Tests for get_reactivation_decision()."""

    def test_get_existing_decision(self):
        """Test retrieving an existing decision."""
        from episodic.db_reactivation_decisions import get_reactivation_decision

        conn = create_test_db()

        # Insert test data
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, confidence, topic_name, topic_start_node_id,
             candidates_json, support_counts_json, gates_json, best_similarity, dormancy_turns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "node_123", "REACTIVATE", 0.85, "Python", "topic_py",
            '[{"topic": "Python", "sim": 0.8}]',
            '{"Python": 3}',
            '{"passed": ["dormancy"], "failed": []}',
            0.8, 5
        ))
        conn.commit()

        result = get_reactivation_decision("node_123", conn)

        assert result is not None
        assert result['user_node_id'] == "node_123"
        assert result['decision'] == "REACTIVATE"
        assert result['confidence'] == 0.85
        assert result['topic_name'] == "Python"
        assert len(result['candidates']) == 1
        assert result['candidates'][0]['topic'] == "Python"
        assert result['support_counts']['Python'] == 3
        assert "dormancy" in result['gates']['passed']
        assert result['best_similarity'] == 0.8
        assert result['dormancy_turns'] == 5

    def test_get_nonexistent_decision(self):
        """Test retrieving a non-existent decision."""
        from episodic.db_reactivation_decisions import get_reactivation_decision

        conn = create_test_db()

        result = get_reactivation_decision("nonexistent", conn)

        assert result is None


class TestGetRecentReactivationDecisions:
    """Tests for get_recent_reactivation_decisions()."""

    def test_get_recent_with_limit(self):
        """Test getting recent decisions with limit."""
        from episodic.db_reactivation_decisions import get_recent_reactivation_decisions

        conn = create_test_db()

        # Insert multiple decisions
        for i in range(5):
            conn.execute("""
                INSERT INTO reactivation_decisions
                (user_node_id, decision, candidates_json, support_counts_json, gates_json, created_at)
                VALUES (?, ?, '[]', '{}', '{}', datetime('now', ? || ' minutes'))
            """, (f"node_{i}", "CONTINUE", f"-{i}"))
        conn.commit()

        results = get_recent_reactivation_decisions(limit=3, conn=conn)

        assert len(results) == 3

    def test_get_recent_with_filter(self):
        """Test filtering by decision type."""
        from episodic.db_reactivation_decisions import get_recent_reactivation_decisions

        conn = create_test_db()

        # Insert mixed decisions
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_1", "CONTINUE"))
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_2", "REACTIVATE"))
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_3", "CONTINUE"))
        conn.commit()

        results = get_recent_reactivation_decisions(decision_filter="REACTIVATE", conn=conn)

        assert len(results) == 1
        assert results[0]['decision'] == "REACTIVATE"

    def test_get_recent_empty(self):
        """Test getting recent from empty table."""
        from episodic.db_reactivation_decisions import get_recent_reactivation_decisions

        conn = create_test_db()

        results = get_recent_reactivation_decisions(conn=conn)

        assert results == []


class TestStoreReactivationLabel:
    """Tests for store_reactivation_label()."""

    def test_store_reactivate_label(self):
        """Test storing a reactivate label."""
        from episodic.db_reactivation_decisions import store_reactivation_label

        conn = create_test_db()

        result = store_reactivation_label(
            user_node_id="node_123",
            ground_truth="reactivate:Python",
            labeler="tester",
            notes="Clear topic return",
            conn=conn
        )

        assert result is True

        # Verify stored
        cursor = conn.execute("SELECT * FROM reactivation_labels WHERE user_node_id = ?", ("node_123",))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "reactivate:Python"
        assert row[2] == "tester"
        assert row[3] == "Clear topic return"

    def test_store_continue_label(self):
        """Test storing a continue label."""
        from episodic.db_reactivation_decisions import store_reactivation_label

        conn = create_test_db()

        result = store_reactivation_label(
            user_node_id="node_456",
            ground_truth="continue",
            conn=conn
        )

        assert result is True

        cursor = conn.execute("SELECT ground_truth FROM reactivation_labels WHERE user_node_id = ?", ("node_456",))
        assert cursor.fetchone()[0] == "continue"

    def test_store_new_topic_label(self):
        """Test storing a new_topic label."""
        from episodic.db_reactivation_decisions import store_reactivation_label

        conn = create_test_db()

        result = store_reactivation_label(
            user_node_id="node_789",
            ground_truth="new_topic",
            labeler="reviewer",
            conn=conn
        )

        assert result is True

        cursor = conn.execute("SELECT ground_truth, labeler FROM reactivation_labels WHERE user_node_id = ?", ("node_789",))
        row = cursor.fetchone()
        assert row[0] == "new_topic"
        assert row[1] == "reviewer"

    def test_store_label_overwrites_existing(self):
        """Test that storing label overwrites existing."""
        from episodic.db_reactivation_decisions import store_reactivation_label

        conn = create_test_db()

        # Store first label
        store_reactivation_label("node_123", "continue", conn=conn)

        # Store replacement label
        store_reactivation_label("node_123", "reactivate:Python", labeler="reviewer", conn=conn)

        # Verify only one row with updated value
        cursor = conn.execute("SELECT COUNT(*) FROM reactivation_labels WHERE user_node_id = ?", ("node_123",))
        assert cursor.fetchone()[0] == 1

        cursor = conn.execute("SELECT ground_truth, labeler FROM reactivation_labels WHERE user_node_id = ?", ("node_123",))
        row = cursor.fetchone()
        assert row[0] == "reactivate:Python"
        assert row[1] == "reviewer"


class TestGetLabeledDecisions:
    """Tests for get_labeled_decisions()."""

    def test_get_labeled_with_decisions_and_labels(self):
        """Test getting labeled decisions with both tables populated."""
        from episodic.db_reactivation_decisions import get_labeled_decisions

        conn = create_test_db()

        # Insert decision
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, confidence, topic_name, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, ?, ?, '[]', '{}', '{}')
        """, ("node_123", "REACTIVATE", 0.85, "Python"))

        # Insert label
        conn.execute("""
            INSERT INTO reactivation_labels
            (user_node_id, ground_truth, labeler, notes)
            VALUES (?, ?, ?, ?)
        """, ("node_123", "reactivate:Python", "tester", "Correct"))
        conn.commit()

        results = get_labeled_decisions(conn)

        assert len(results) == 1
        assert results[0]['user_node_id'] == "node_123"
        assert results[0]['decision'] == "REACTIVATE"
        assert results[0]['ground_truth'] == "reactivate:Python"
        assert results[0]['labeler'] == "tester"
        assert results[0]['notes'] == "Correct"

    def test_get_labeled_excludes_unlabeled(self):
        """Test that unlabeled decisions are excluded."""
        from episodic.db_reactivation_decisions import get_labeled_decisions

        conn = create_test_db()

        # Insert decision without label
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_unlabeled", "CONTINUE"))

        # Insert decision with label
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_labeled", "REACTIVATE"))
        conn.execute("""
            INSERT INTO reactivation_labels
            (user_node_id, ground_truth)
            VALUES (?, ?)
        """, ("node_labeled", "reactivate:Test"))
        conn.commit()

        results = get_labeled_decisions(conn)

        assert len(results) == 1
        assert results[0]['user_node_id'] == "node_labeled"

    def test_get_labeled_empty(self):
        """Test getting labeled when no labels exist."""
        from episodic.db_reactivation_decisions import get_labeled_decisions

        conn = create_test_db()

        # Insert decision without label
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_123", "CONTINUE"))
        conn.commit()

        results = get_labeled_decisions(conn)

        assert results == []


class TestGetUnlabeledDecisions:
    """Tests for get_unlabeled_decisions()."""

    def test_get_unlabeled_excludes_labeled(self):
        """Test that labeled decisions are excluded."""
        from episodic.db_reactivation_decisions import get_unlabeled_decisions

        conn = create_test_db()

        # Insert unlabeled decision
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_unlabeled", "CONTINUE"))

        # Insert labeled decision
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_labeled", "REACTIVATE"))
        conn.execute("""
            INSERT INTO reactivation_labels
            (user_node_id, ground_truth)
            VALUES (?, ?)
        """, ("node_labeled", "reactivate:Test"))
        conn.commit()

        results = get_unlabeled_decisions(conn=conn)

        assert len(results) == 1
        assert results[0]['user_node_id'] == "node_unlabeled"

    def test_get_unlabeled_respects_limit(self):
        """Test limit parameter is respected."""
        from episodic.db_reactivation_decisions import get_unlabeled_decisions

        conn = create_test_db()

        # Insert multiple unlabeled decisions
        for i in range(10):
            conn.execute("""
                INSERT INTO reactivation_decisions
                (user_node_id, decision, candidates_json, support_counts_json, gates_json)
                VALUES (?, ?, '[]', '{}', '{}')
            """, (f"node_{i}", "CONTINUE"))
        conn.commit()

        results = get_unlabeled_decisions(limit=5, conn=conn)

        assert len(results) == 5

    def test_get_unlabeled_empty(self):
        """Test when all decisions are labeled."""
        from episodic.db_reactivation_decisions import get_unlabeled_decisions

        conn = create_test_db()

        # Insert decision with label
        conn.execute("""
            INSERT INTO reactivation_decisions
            (user_node_id, decision, candidates_json, support_counts_json, gates_json)
            VALUES (?, ?, '[]', '{}', '{}')
        """, ("node_123", "CONTINUE"))
        conn.execute("""
            INSERT INTO reactivation_labels
            (user_node_id, ground_truth)
            VALUES (?, ?)
        """, ("node_123", "continue"))
        conn.commit()

        results = get_unlabeled_decisions(conn=conn)

        assert results == []


class TestIntegration:
    """Integration tests for the full decision + label workflow."""

    def test_full_persist_and_label_workflow(self):
        """Test persisting a decision, then labeling it."""
        from episodic.db_reactivation_decisions import (
            persist_reactivation_decision,
            store_reactivation_label,
            get_labeled_decisions,
            get_unlabeled_decisions,
        )

        conn = create_test_db()

        # Create and persist a decision
        decision = MockReactivationDecision(
            action="REACTIVATE",
            topic_name="Python",
            topic_start_node_id="topic_py",
            debug={
                "confidence": 0.75,
                "candidates": [{"topic": "Python", "sim": 0.75}],
                "support_counts": {"Python": 2},
                "gates_passed": ["dormancy"],
                "gates_failed": [],
            }
        )

        # Pass connection directly
        persist_reactivation_decision("node_test", decision, conn)

        # Verify it's unlabeled
        unlabeled = get_unlabeled_decisions(conn=conn)
        assert len(unlabeled) == 1
        assert unlabeled[0]['user_node_id'] == "node_test"

        labeled = get_labeled_decisions(conn)
        assert len(labeled) == 0

        # Add a label
        store_reactivation_label(
            user_node_id="node_test",
            ground_truth="reactivate:Python",
            labeler="test_user",
            notes="Confirmed correct",
            conn=conn
        )

        # Verify it's now labeled
        unlabeled = get_unlabeled_decisions(conn=conn)
        assert len(unlabeled) == 0

        labeled = get_labeled_decisions(conn)
        assert len(labeled) == 1
        assert labeled[0]['decision'] == "REACTIVATE"
        assert labeled[0]['ground_truth'] == "reactivate:Python"
        assert labeled[0]['labeler'] == "test_user"
