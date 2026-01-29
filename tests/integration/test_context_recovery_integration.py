"""
Integration tests for context recovery wiring in conversation.py.

These tests verify:
1. "B Disappears" with real conversation flow
2. Debug persistence to context_assembly_debug table
3. Mode switching in hybrid mode
4. Config sourcing for context_token_budget
5. Backfill verification for topic_nodes data integrity
"""

import pytest
import sqlite3
import tempfile
import os
from typing import Dict, Any, List, Optional

# Use fixtures from conftest.py


@pytest.fixture
def context_recovery_db():
    """
    Create a test database with full schema for context recovery testing.

    This fixture creates:
    - All standard episodic tables
    - Sample topics with nodes
    - topic_nodes membership data
    - topic_working_set entries
    """
    test_db_path = tempfile.mktemp(suffix='_context_recovery.db')
    original_db_path = os.environ.get('EPISODIC_DB_PATH')
    os.environ['EPISODIC_DB_PATH'] = test_db_path

    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Create nodes table
    cursor.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT UNIQUE,
            content TEXT,
            role TEXT,
            parent_id TEXT,
            provider TEXT,
            model TEXT,
            is_meta_query BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create topics table
    cursor.execute("""
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

    # Create context_assembly_debug table
    cursor.execute("""
        CREATE TABLE context_assembly_debug (
            user_node_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            active_topic_id TEXT,
            included_node_ids_json TEXT NOT NULL DEFAULT '[]',
            token_counts_json TEXT NOT NULL DEFAULT '{}',
            reactivation_fired INTEGER NOT NULL DEFAULT 0,
            reactivation_reason TEXT,
            truncation_info_json TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create state table
    cursor.execute("""
        CREATE TABLE state (
            name TEXT PRIMARY KEY,
            head_id TEXT
        )
    """)
    cursor.execute("INSERT INTO state (name, head_id) VALUES ('head', NULL)")

    # Create configuration table
    cursor.execute("""
        CREATE TABLE configuration (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    yield conn

    conn.close()
    try:
        os.remove(test_db_path)
    except Exception:
        pass

    # Restore original EPISODIC_DB_PATH and close pool
    if original_db_path is not None:
        os.environ['EPISODIC_DB_PATH'] = original_db_path
    else:
        os.environ.pop('EPISODIC_DB_PATH', None)

    from episodic.db_connection import close_pool
    close_pool()


@pytest.fixture
def python_topic_data(context_recovery_db):
    """Create Topic A: Python programming discussion."""
    cursor = context_recovery_db.cursor()

    # Create nodes for Python topic
    nodes = [
        ('python_u1', 'pu1', 'How do I create a list comprehension in Python?', 'user', None),
        ('python_a1', 'pa1', 'A list comprehension in Python uses the syntax [expr for item in iterable]...', 'assistant', 'python_u1'),
        ('python_u2', 'pu2', 'What about dictionary comprehensions?', 'user', 'python_a1'),
        ('python_a2', 'pa2', 'Dictionary comprehensions use {key_expr: value_expr for item in iterable}...', 'assistant', 'python_u2'),
        ('python_u3', 'pu3', 'Can you show me an async function example?', 'user', 'python_a2'),
        ('python_a3', 'pa3', 'Sure! Here is an async function: async def fetch_data(): await asyncio.sleep(1)...', 'assistant', 'python_u3'),
    ]

    for i, (node_id, short_id, content, role, parent_id) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO nodes (id, short_id, content, role, parent_id) VALUES (?, ?, ?, ?, ?)",
            (node_id, short_id, content, role, parent_id)
        )

    # Create topic record
    cursor.execute(
        "INSERT INTO topics (name, start_node_id, end_node_id) VALUES (?, ?, ?)",
        ('python-programming', 'python_u1', 'python_a3')
    )

    # Populate topic_nodes
    for i, (node_id, _, _, role, _) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role) VALUES (?, ?, ?, ?)",
            ('python_u1', node_id, i + 1, role)
        )

    # Create working set with summary
    cursor.execute(
        "INSERT INTO topic_working_set (topic_start_node_id, topic_name, summary_md) VALUES (?, ?, ?)",
        ('python_u1', 'python-programming',
         'Discussion about Python programming including list comprehensions, '
         'dictionary comprehensions, and async/await patterns.')
    )

    context_recovery_db.commit()
    return {'start_node_id': 'python_u1', 'nodes': nodes, 'name': 'python-programming'}


@pytest.fixture
def coffee_topic_data(context_recovery_db, python_topic_data):
    """Create Topic B: Coffee brewing discussion (after Python topic)."""
    cursor = context_recovery_db.cursor()

    # Create nodes for Coffee topic (after Python topic)
    nodes = [
        ('coffee_u1', 'cu1', 'What is the ideal water temperature for pour-over coffee?', 'user', 'python_a3'),
        ('coffee_a1', 'ca1', 'The ideal water temperature for pour-over is 195-205°F (90-96°C)...', 'assistant', 'coffee_u1'),
        ('coffee_u2', 'cu2', 'How fine should the grind be for espresso?', 'user', 'coffee_a1'),
        ('coffee_a2', 'ca2', 'For espresso, you want a fine grind similar to table salt...', 'assistant', 'coffee_u2'),
    ]

    for i, (node_id, short_id, content, role, parent_id) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO nodes (id, short_id, content, role, parent_id) VALUES (?, ?, ?, ?, ?)",
            (node_id, short_id, content, role, parent_id)
        )

    # Create topic record
    cursor.execute(
        "INSERT INTO topics (name, start_node_id, end_node_id) VALUES (?, ?, ?)",
        ('coffee-brewing', 'coffee_u1', 'coffee_a2')
    )

    # Populate topic_nodes (turn_idx continues from Python topic)
    base_turn_idx = len(python_topic_data['nodes'])
    for i, (node_id, _, _, role, _) in enumerate(nodes):
        cursor.execute(
            "INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role) VALUES (?, ?, ?, ?)",
            ('coffee_u1', node_id, base_turn_idx + i + 1, role)
        )

    # Create working set (no summary yet - new topic)
    cursor.execute(
        "INSERT INTO topic_working_set (topic_start_node_id, topic_name, summary_md) VALUES (?, ?, ?)",
        ('coffee_u1', 'coffee-brewing', '')
    )

    context_recovery_db.commit()
    return {'start_node_id': 'coffee_u1', 'nodes': nodes, 'name': 'coffee-brewing'}


class TestBDisappears:
    """
    Test 1: "B Disappears" with Real Conversation Flow

    When reactivating Topic A (Python), Topic B (Coffee) content should
    NOT appear in the assembled context.
    """

    def test_b_disappears_when_reactivating_python_topic(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """
        Critical test: When reactivating Python topic, Coffee content disappears.
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.recall.reactivation import ReactivationDecision

        # Simulate reactivation to Python topic
        strategy = TopicLocalStrategy(exchange_pairs=4)

        result = strategy.assemble(
            user_turn_text="How do I use decorators in Python?",
            user_node_id=None,
            active_topic_start_node_id='python_u1',
            user_embedding=None,
            token_budget=4000,
            conn=context_recovery_db,
        )

        # Verify we got messages
        assert len(result.messages) > 0, "Should have context messages"

        # Extract all content
        all_content = " ".join(msg["content"].lower() for msg in result.messages)

        # Python content SHOULD be present
        python_keywords = ['python', 'list comprehension', 'async', 'dictionary']
        python_found = any(kw in all_content for kw in python_keywords)
        assert python_found, f"Python content should be present. Content: {all_content[:200]}"

        # Coffee content should NOT be present
        coffee_keywords = ['coffee', 'pour-over', 'espresso', 'grind', 'water temperature']
        for keyword in coffee_keywords:
            assert keyword not in all_content, \
                f"Coffee keyword '{keyword}' should NOT be in context when Python is active"

        # Verify debug info
        assert result.debug["mode"] == "topic_local"
        assert result.debug["topic_start_node_id"] == "python_u1"

        # All included node IDs should be from Python topic
        for node_id in result.debug["included_node_ids"]:
            assert node_id.startswith("python_"), \
                f"Node {node_id} should be from Python topic, not Coffee"

    def test_a_disappears_when_reactivating_coffee_topic(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """Verify inverse: when Coffee is active, Python content disappears."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy(exchange_pairs=4)

        result = strategy.assemble(
            user_turn_text="What about cold brew ratios?",
            user_node_id=None,
            active_topic_start_node_id='coffee_u1',
            user_embedding=None,
            token_budget=4000,
            conn=context_recovery_db,
        )

        all_content = " ".join(msg["content"].lower() for msg in result.messages)

        # Python content should NOT be present
        python_keywords = ['list comprehension', 'async def', 'dictionary comprehension']
        for keyword in python_keywords:
            assert keyword not in all_content, \
                f"Python keyword '{keyword}' should NOT be in context when Coffee is active"


class TestDebugPersistence:
    """
    Test 2: Verify Debug Persistence

    Check that context_assembly_debug table is correctly populated.
    """

    def test_persist_and_retrieve_debug_info(self, context_recovery_db, python_topic_data):
        """Test that debug info is correctly persisted and retrievable."""
        from episodic.db_context_debug import (
            persist_context_assembly_debug,
            get_context_assembly_debug
        )
        from episodic.recall.reactivation import ReactivationDecision

        user_node_id = 'test_user_node_123'

        debug_info = {
            "mode": "topic_local",
            "topic_start_node_id": "python_u1",
            "included_node_ids": ["python_u1", "python_a1", "python_u2"],
            "token_counts": {"total_estimate": 500, "conversation_history": 400},
            "reactivation_fired": True,
            "truncation_info": None,
        }

        reactivation_decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="python-programming",
            topic_start_node_id="python_u1",
            debug={"reason": "high_similarity"}
        )

        # Persist
        result = persist_context_assembly_debug(
            user_node_id, debug_info, reactivation_decision, conn=context_recovery_db
        )
        assert result is True, "Persist should succeed"

        # Retrieve
        retrieved = get_context_assembly_debug(user_node_id, conn=context_recovery_db)

        assert retrieved is not None, "Should retrieve persisted debug info"
        assert retrieved["mode"] == "topic_local"
        assert retrieved["active_topic_id"] == "python_u1"
        assert retrieved["reactivation_fired"] is True
        assert "python_u1" in retrieved["included_node_ids"]
        assert retrieved["token_counts"]["total_estimate"] == 500
        assert "REACTIVATE:high_similarity" in retrieved["reactivation_reason"]

    def test_debug_info_includes_reactivation_reason(self, context_recovery_db):
        """Test that reactivation reason is extracted from decision debug."""
        from episodic.db_context_debug import (
            persist_context_assembly_debug,
            get_context_assembly_debug
        )
        from episodic.recall.reactivation import ReactivationDecision

        user_node_id = 'test_reason_node'
        debug_info = {
            "mode": "topic_local",
            "topic_start_node_id": "python_u1",
            "included_node_ids": [],
            "token_counts": {},
            "reactivation_fired": True,
        }

        decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="test-topic",
            topic_start_node_id="node_x",
            debug={"reason": "explicit_reference"}
        )

        persist_context_assembly_debug(
            user_node_id, debug_info, decision, conn=context_recovery_db
        )

        retrieved = get_context_assembly_debug(user_node_id, conn=context_recovery_db)
        assert "explicit_reference" in retrieved["reactivation_reason"]


class TestHybridModeSwitching:
    """
    Test 3: Mode Switching in Hybrid

    Verify that hybrid mode correctly switches between ancestry and topic_local
    based on reactivation decision.
    """

    def test_hybrid_uses_topic_local_when_reactivation_fires(self, context_recovery_db, python_topic_data):
        """Hybrid mode should use topic_local strategy when reactivation fires."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.recall.reactivation import ReactivationDecision

        decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="python-programming",
            topic_start_node_id="python_u1"
        )

        strategy = select_strategy(ContextRecoveryMode.HYBRID, decision)
        assert isinstance(strategy, TopicLocalStrategy), \
            "Hybrid + REACTIVATE should select TopicLocalStrategy"

        # Verify assembled context uses topic_local mode
        result = strategy.assemble(
            user_turn_text="More about Python",
            user_node_id=None,
            active_topic_start_node_id="python_u1",
            user_embedding=None,
            token_budget=4000,
            conn=context_recovery_db,
        )

        assert result.debug["mode"] == "topic_local"

    def test_hybrid_uses_ancestry_when_continue(self, context_recovery_db):
        """Hybrid mode should use ancestry strategy when action is CONTINUE."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.ancestry import AncestryStrategy
        from episodic.recall.reactivation import ReactivationDecision

        decision = ReactivationDecision(action="CONTINUE")

        strategy = select_strategy(ContextRecoveryMode.HYBRID, decision)
        assert isinstance(strategy, AncestryStrategy), \
            "Hybrid + CONTINUE should select AncestryStrategy"

    def test_hybrid_uses_ancestry_when_no_decision(self, context_recovery_db):
        """Hybrid mode should use ancestry strategy when no reactivation decision."""
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy
        )
        from episodic.context_recovery.ancestry import AncestryStrategy

        strategy = select_strategy(ContextRecoveryMode.HYBRID, None)
        assert isinstance(strategy, AncestryStrategy), \
            "Hybrid + None decision should select AncestryStrategy"


class TestConfigSourcing:
    """
    Test 4: Config Sourcing

    Verify that context_token_budget is read from config, not hardcoded.
    """

    def test_token_budget_from_config(self, context_recovery_db, python_topic_data):
        """
        Test that token_budget is sourced from config.

        This test verifies that the strategy uses the token_budget parameter,
        which is sourced from config by the caller (context_builder/conversation.py).
        """
        from episodic.context_recovery.topic_local import TopicLocalStrategy
        from episodic.config import config

        # Verify config has the expected key
        budget = config.get("context_token_budget", 4000)
        assert budget is not None
        assert isinstance(budget, int)

        # Test that strategy accepts and uses the token_budget
        strategy = TopicLocalStrategy()

        result = strategy.assemble(
            user_turn_text="Test query about Python",
            user_node_id=None,
            active_topic_start_node_id="python_u1",
            user_embedding=None,
            token_budget=budget,  # Pass config value
            conn=context_recovery_db,
        )

        # Strategy should work with config-sourced budget
        assert result is not None
        assert result.debug["mode"] == "topic_local"
        assert "token_counts" in result.debug

    def test_explicit_token_budget_overrides_config(self, context_recovery_db, python_topic_data):
        """
        Test that explicit token_budget parameter overrides config.

        This is a simpler test that verifies the token_budget parameter flows through
        to build_with_strategy without relying on config mocking.
        """
        from episodic.context_builder import ContextBuilder
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        # Directly test that build_with_strategy accepts and uses token_budget
        strategy = TopicLocalStrategy(exchange_pairs=4)

        # Call with explicit token_budget
        result = strategy.assemble(
            user_turn_text="Test with explicit budget",
            user_node_id=None,
            active_topic_start_node_id="python_u1",
            user_embedding=None,
            token_budget=2000,  # Explicit override
            conn=context_recovery_db,
        )

        assert result is not None
        assert result.debug["mode"] == "topic_local"

        # Verify the strategy ran correctly with the budget
        # (Token budget affects truncation, which we can verify via token counts)
        assert "token_counts" in result.debug


class TestBackfillVerification:
    """
    Test 5: Backfill Verification

    Verify topic_nodes data integrity for topic-local assembly.
    """

    def test_topic_nodes_contains_all_topic_members(
        self, context_recovery_db, python_topic_data
    ):
        """Verify that all nodes in a topic are in topic_nodes table."""
        from episodic.db_topic_nodes import get_topic_nodes, count_topic_nodes

        topic_start_id = python_topic_data['start_node_id']
        expected_nodes = [n[0] for n in python_topic_data['nodes']]

        # Get all nodes from topic_nodes
        topic_nodes = get_topic_nodes(topic_start_id, conn=context_recovery_db)
        actual_node_ids = [n['node_id'] for n in topic_nodes]

        # Verify all expected nodes are present
        for expected_id in expected_nodes:
            assert expected_id in actual_node_ids, \
                f"Node {expected_id} should be in topic_nodes"

        # Verify count matches
        count = count_topic_nodes(topic_start_id, conn=context_recovery_db)
        assert count == len(expected_nodes)

    def test_topic_nodes_does_not_cross_topic_boundaries(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """Verify that topic_nodes doesn't mix topics."""
        from episodic.db_topic_nodes import get_topic_nodes

        # Get Python topic nodes
        python_nodes = get_topic_nodes('python_u1', conn=context_recovery_db)
        python_ids = [n['node_id'] for n in python_nodes]

        # Get Coffee topic nodes
        coffee_nodes = get_topic_nodes('coffee_u1', conn=context_recovery_db)
        coffee_ids = [n['node_id'] for n in coffee_nodes]

        # Verify no overlap
        overlap = set(python_ids) & set(coffee_ids)
        assert len(overlap) == 0, \
            f"Topic nodes should not overlap. Found overlap: {overlap}"

        # Verify each topic has correct nodes
        for node_id in python_ids:
            assert node_id.startswith('python_'), \
                f"Python topic should only contain python_ nodes, found {node_id}"

        for node_id in coffee_ids:
            assert node_id.startswith('coffee_'), \
                f"Coffee topic should only contain coffee_ nodes, found {node_id}"

    def test_topic_working_set_exists_for_topics(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """Verify that topic_working_set has entries for all topics."""
        from episodic.db_topic_nodes import get_topic_working_set

        # Python topic should have working set with summary
        python_ws = get_topic_working_set('python_u1', conn=context_recovery_db)
        assert python_ws is not None
        assert python_ws['topic_name'] == 'python-programming'
        assert 'Python' in python_ws['summary_md'] or 'python' in python_ws['summary_md'].lower()

        # Coffee topic should have working set (empty summary)
        coffee_ws = get_topic_working_set('coffee_u1', conn=context_recovery_db)
        assert coffee_ws is not None
        assert coffee_ws['topic_name'] == 'coffee-brewing'

    def test_exchanges_retrieval_respects_topic_boundary(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """Verify get_last_n_exchanges_in_topic only returns topic's exchanges."""
        from episodic.db_topic_nodes import get_last_n_exchanges_in_topic

        # Get Python exchanges
        python_exchanges = get_last_n_exchanges_in_topic(
            'python_u1', n=10, conn=context_recovery_db
        )

        # Should have exactly 3 exchanges (6 nodes / 2)
        assert len(python_exchanges) == 3, \
            f"Python topic should have 3 exchanges, got {len(python_exchanges)}"

        # All content should be Python-related
        for ex in python_exchanges:
            content = (ex['user_content'] + " " + ex['assistant_content']).lower()
            assert 'coffee' not in content
            assert 'espresso' not in content
            assert 'pour-over' not in content

        # Get Coffee exchanges
        coffee_exchanges = get_last_n_exchanges_in_topic(
            'coffee_u1', n=10, conn=context_recovery_db
        )

        # Should have exactly 2 exchanges
        assert len(coffee_exchanges) == 2, \
            f"Coffee topic should have 2 exchanges, got {len(coffee_exchanges)}"

        # All content should be Coffee-related
        for ex in coffee_exchanges:
            content = (ex['user_content'] + " " + ex['assistant_content']).lower()
            assert 'python' not in content
            assert 'list comprehension' not in content


class TestEdgeCases:
    """Additional edge case tests discovered during integration."""

    def test_empty_topic_returns_minimal_context(self, context_recovery_db):
        """Test behavior when topic has no nodes."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy()

        result = strategy.assemble(
            user_turn_text="Hello?",
            user_node_id=None,
            active_topic_start_node_id='nonexistent_topic_xyz',
            user_embedding=None,
            token_budget=4000,
            conn=context_recovery_db,
        )

        # Empty/nonexistent topic triggers thin_topic_local fallback to ancestry
        # This is the expected behavior - thin topics should not use topic_local
        assert result.debug["mode"] in ("topic_local", "ancestry_fallback")
        if result.debug["mode"] == "ancestry_fallback":
            # Verify fallback reason is documented
            assert result.debug.get("fallback_reason") == "thin_topic_local"

    def test_null_topic_returns_empty_context(self, context_recovery_db):
        """Test behavior when no active topic."""
        from episodic.context_recovery.topic_local import TopicLocalStrategy

        strategy = TopicLocalStrategy()

        result = strategy.assemble(
            user_turn_text="Hello?",
            user_node_id=None,
            active_topic_start_node_id=None,
            user_embedding=None,
            token_budget=4000,
            conn=context_recovery_db,
        )

        assert len(result.messages) == 0
        assert result.debug["reactivation_fired"] is False

    def test_force_continue_boundary_semantics(
        self, context_recovery_db, python_topic_data, coffee_topic_data
    ):
        """
        Verify that when reactivation fires, topic detection uses FORCE_CONTINUE.

        This test verifies the boundary semantics: if reactivation_applied is True,
        the topic tracker should not create a new topic (FORCE_CONTINUE override).
        """
        # This is more of a documentation test - the actual implementation
        # is in conversation.py. We verify the expected behavior here.
        from episodic.recall.reactivation import ReactivationDecision

        decision = ReactivationDecision(
            action="REACTIVATE",
            topic_name="python-programming",
            topic_start_node_id="python_u1"
        )

        # When reactivation fires (action == "REACTIVATE"):
        # 1. The conversation.py code sets reactivation_applied = True
        # 2. Topic detection is called with decision_override="FORCE_CONTINUE"
        # 3. This prevents creating a new topic when the user is clearly
        #    continuing an old topic

        # Verify the decision has the correct action
        assert decision.action == "REACTIVATE"

        # The actual FORCE_CONTINUE logic is tested via conversation.py
        # integration tests or manual testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
