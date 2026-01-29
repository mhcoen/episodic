"""
End-to-end integration tests for topic reactivation.

Tests the full reactivation flow: A → B → resume A.
This is the core UX that must work.
"""

import numpy as np
import pytest
import sqlite3
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from episodic.config import config
from episodic.recall.reactivation import (
    probe_reactivation,
    ReactivationDecision,
)


# Mark all tests in this module for the reactivation CI gate
pytestmark = pytest.mark.reactivation


@pytest.fixture
def isolated_db(tmp_path):
    """Create an isolated database for testing."""
    import episodic.db_connection as db_conn

    db_path = tmp_path / "test_reactivation.db"

    # Reset the cached db path to use our test path
    original_resolved = db_conn._resolved_db_path
    db_conn._resolved_db_path = None

    # Set up the test database
    with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(db_path)}):
        from episodic.db import initialize_db

        initialize_db()

        yield db_path

    # Restore original
    db_conn._resolved_db_path = original_resolved


@pytest.fixture
def mock_chroma():
    """Mock Chroma collection that returns embeddings."""
    # Create embeddings that are somewhat similar (for support check)
    # Python embeddings cluster around [1, 0, 0, ...]
    # Sourdough embeddings cluster around [0, 1, 0, ...]

    python_base = np.array([1.0] + [0.0] * 383)
    sourdough_base = np.array([0.0, 1.0] + [0.0] * 382)

    embeddings_store = {}

    def add_embedding(node_id: str, category: str):
        """Add an embedding for a node."""
        if category == "python":
            # Add small noise to base
            emb = python_base + np.random.randn(384) * 0.1
        else:  # sourdough
            emb = sourdough_base + np.random.randn(384) * 0.1
        emb = emb / np.linalg.norm(emb)  # Normalize
        embeddings_store[node_id] = emb

    def get_embedding(text: str):
        """Get embedding for input text."""
        text_lower = text.lower()
        if "python" in text_lower or "retry" in text_lower or "tenacity" in text_lower:
            base = python_base
        elif "sourdough" in text_lower or "bread" in text_lower or "proof" in text_lower:
            base = sourdough_base
        else:
            base = np.random.randn(384)
        emb = base + np.random.randn(384) * 0.05
        return (emb / np.linalg.norm(emb)).tolist()

    mock_collection = MagicMock()

    def mock_get(ids, include=None):
        """Mock collection.get()."""
        result_ids = []
        result_embeddings = []
        for node_id in ids:
            if node_id in embeddings_store:
                result_ids.append(node_id)
                result_embeddings.append(embeddings_store[node_id])
        return {
            'ids': result_ids,
            'embeddings': result_embeddings,
        }

    mock_collection.get = mock_get
    mock_collection._embedding_function = lambda texts: [get_embedding(t) for t in texts]

    mock_rag = MagicMock()
    mock_rag.get_collection.return_value = mock_collection

    return mock_rag, add_embedding, get_embedding


class TestReactivationE2E:
    """End-to-end tests for topic reactivation."""

    def test_reactivation_full_flow(self, isolated_db, mock_chroma):
        """
        End-to-end test: A → B → resume A
        This is the core UX that must work.
        """
        mock_rag, add_embedding, get_embedding = mock_chroma

        # 1. Create topic A (Python retry) and topic B (sourdough)
        with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
            from episodic.db_connection import get_connection

            with get_connection() as conn:
                # Create nodes for Python topic (topic A)
                python_start_id = str(uuid.uuid4())[:36]
                python_user_ids = []
                for i in range(3):
                    user_id = str(uuid.uuid4())[:36]
                    asst_id = str(uuid.uuid4())[:36]
                    python_user_ids.append(user_id)
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'user', ?)
                    """, (user_id, f"py_u{i}", f"Tell me about Python retry logic {i}",
                          python_start_id if i == 0 else python_user_ids[i-1]))
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'assistant', ?)
                    """, (asst_id, f"py_a{i}", f"Python retry response {i}", user_id))
                    add_embedding(user_id, "python")

                python_start_id = python_user_ids[0]

                # Create Python topic
                conn.execute("""
                    INSERT INTO topics (name, start_node_id, created_at)
                    VALUES (?, ?, ?)
                """, ("python-retry-patterns", python_start_id, datetime.now().isoformat()))

                # Create nodes for Sourdough topic (topic B)
                sourdough_start_id = str(uuid.uuid4())[:36]
                sourdough_user_ids = []
                for i in range(3):
                    user_id = str(uuid.uuid4())[:36]
                    asst_id = str(uuid.uuid4())[:36]
                    sourdough_user_ids.append(user_id)
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'user', ?)
                    """, (user_id, f"sd_u{i}", f"Tell me about sourdough {i}",
                          sourdough_start_id if i == 0 else sourdough_user_ids[i-1]))
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'assistant', ?)
                    """, (asst_id, f"sd_a{i}", f"Sourdough response {i}", user_id))
                    add_embedding(user_id, "sourdough")

                sourdough_start_id = sourdough_user_ids[0]

                # Close Python topic (set end_node_id)
                conn.execute("""
                    UPDATE topics SET end_node_id = ?
                    WHERE start_node_id = ?
                """, (python_user_ids[-1], python_start_id))

                # Create sourdough topic (currently active)
                conn.execute("""
                    INSERT INTO topics (name, start_node_id, created_at)
                    VALUES (?, ?, ?)
                """, ("sourdough-baking", sourdough_start_id, datetime.now().isoformat()))

                conn.commit()

                # 2. Create topic_centroids entries
                # Python topic centroid
                conn.execute("""
                    INSERT INTO topic_centroids
                    (start_node_id, centroid_medoid_exchange_id, exchange_count, last_active_turn_idx, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (python_start_id, python_user_ids[1], 3, 6, datetime.now().isoformat()))

                # Sourdough topic centroid (more recent)
                conn.execute("""
                    INSERT INTO topic_centroids
                    (start_node_id, centroid_medoid_exchange_id, exchange_count, last_active_turn_idx, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (sourdough_start_id, sourdough_user_ids[1], 3, 12, datetime.now().isoformat()))

                conn.commit()

        # 3. Run probe with "Back to that Python retry thing"
        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
                user_input = "Back to that Python retry thing - should I use tenacity?"
                user_embedding = np.array(get_embedding(user_input))

                decision = probe_reactivation(
                    user_input=user_input,
                    user_embedding=user_embedding,
                    active_topic_start_node_id=sourdough_start_id,  # Currently in sourdough
                    cooldown_turns=0,
                    now=datetime.now(),
                    recent_nodes=[]
                )

        # 4. Assert: Topic A should be reactivated
        assert decision.action == "REACTIVATE", f"Expected REACTIVATE, got {decision.action}. Debug: {decision.debug}"
        assert decision.topic_name == "python-retry-patterns", f"Expected python topic, got {decision.topic_name}"
        assert decision.topic_start_node_id == python_start_id

        # No errors in debug
        assert "error" not in decision.debug or decision.debug.get("error") is None
        assert "exit_reason" not in decision.debug or decision.debug.get("exit_reason") is None

    def test_reactivation_respects_cooldown(self, isolated_db, mock_chroma):
        """Test that reactivation respects cooldown period."""
        mock_rag, add_embedding, get_embedding = mock_chroma

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
                user_input = "Back to that Python retry thing"
                user_embedding = np.array(get_embedding(user_input))

                # With cooldown active, should CONTINUE
                decision = probe_reactivation(
                    user_input=user_input,
                    user_embedding=user_embedding,
                    active_topic_start_node_id=None,
                    cooldown_turns=2,  # Cooldown active
                    now=datetime.now(),
                    recent_nodes=[]
                )

                assert decision.action == "CONTINUE"
                assert "cooldown" in decision.debug.get("gates_failed", [])

    def test_reactivation_short_input_continues(self, isolated_db, mock_chroma):
        """Test that very short inputs don't trigger reactivation."""
        mock_rag, add_embedding, get_embedding = mock_chroma

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
                user_input = "Hi there"  # Too short (< 4 words)
                user_embedding = np.array(get_embedding(user_input))

                decision = probe_reactivation(
                    user_input=user_input,
                    user_embedding=user_embedding,
                    active_topic_start_node_id=None,
                    cooldown_turns=0,
                    now=datetime.now(),
                    recent_nodes=[]
                )

                assert decision.action == "CONTINUE"
                assert "input_length" in decision.debug.get("gates_failed", [])

    def test_reactivation_via_alias_matching(self, isolated_db, mock_chroma):
        """
        Test that alias matching (channel B) can trigger reactivation
        even when semantic similarity is low.

        This tests the two-channel gate: referential queries like
        "Back to that Python thing" have low embedding similarity but
        should match via alias tokens (python, retry).
        """
        mock_rag, add_embedding, _ = mock_chroma

        # Create embeddings that return LOW similarity for referential queries
        # but topic has aliases like "python", "retry"
        def get_low_sim_embedding(text: str):
            """Return embeddings with low similarity for referential queries."""
            text_lower = text.lower()
            # Referential queries get a random direction (low sim to any topic)
            if "back to" in text_lower or "that thing" in text_lower:
                base = np.random.randn(384)
            elif "python" in text_lower or "retry" in text_lower:
                base = np.array([1.0] + [0.0] * 383)
            else:
                base = np.random.randn(384)
            emb = base + np.random.randn(384) * 0.05
            return (emb / np.linalg.norm(emb)).tolist()

        # Override the embedding function in mock
        mock_collection = mock_rag.get_collection.return_value
        mock_collection._embedding_function = lambda texts: [get_low_sim_embedding(t) for t in texts]

        with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
            from episodic.db_connection import get_connection

            with get_connection() as conn:
                # Create Python topic with distinctive aliases
                python_start_id = str(uuid.uuid4())[:36]
                python_user_ids = []
                for i in range(3):
                    user_id = str(uuid.uuid4())[:36]
                    asst_id = str(uuid.uuid4())[:36]
                    python_user_ids.append(user_id)
                    # Content contains alias terms
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'user', ?)
                    """, (user_id, f"py_u{i}", f"How do I implement Python retry with tenacity?",
                          python_start_id if i == 0 else python_user_ids[i-1]))
                    conn.execute("""
                        INSERT INTO nodes (id, short_id, content, role, parent_id)
                        VALUES (?, ?, ?, 'assistant', ?)
                    """, (asst_id, f"py_a{i}", f"Use tenacity for Python retry logic.", user_id))
                    add_embedding(user_id, "python")

                python_start_id = python_user_ids[0]

                # Create topic with name containing aliases
                conn.execute("""
                    INSERT INTO topics (name, start_node_id, created_at, end_node_id)
                    VALUES (?, ?, ?, ?)
                """, ("python-retry-patterns", python_start_id,
                      datetime.now().isoformat(), python_user_ids[-1]))

                # Add topic_nodes entries for alias extraction
                for idx, user_id in enumerate(python_user_ids):
                    conn.execute("""
                        INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role)
                        VALUES (?, ?, ?, 'user')
                    """, (python_start_id, user_id, idx))

                # Create centroid entry - set last_active_turn_idx LOW so topic is dormant
                # (needs dormancy >= 4 turns to be considered for reactivation)
                conn.execute("""
                    INSERT INTO topic_centroids
                    (start_node_id, centroid_medoid_exchange_id, exchange_count,
                     last_active_turn_idx, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (python_start_id, python_user_ids[1], 3, 1, datetime.now().isoformat()))

                conn.commit()

        # Run probe with referential query (low semantic sim, but has alias matches)
        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            with patch.dict('os.environ', {'EPISODIC_DB_PATH': str(isolated_db)}):
                # This query has resume cues and alias tokens (python, retry)
                user_input = "Back to that Python retry thing - should I use it?"
                user_embedding = np.array(get_low_sim_embedding(user_input))

                decision = probe_reactivation(
                    user_input=user_input,
                    user_embedding=user_embedding,
                    active_topic_start_node_id=None,  # No active topic
                    cooldown_turns=0,
                    now=datetime.now(),
                    recent_nodes=[]
                )

        # Assert: Should detect resume cues and use two-channel matching
        assert decision.debug.get('resume_cues_detected') is True, \
            f"Resume cues not detected. Debug: {decision.debug}"

        # The probe should use lower threshold (0.25) and check aliases
        assert decision.debug.get('sim_threshold_used') == 0.25, \
            f"Wrong threshold used: {decision.debug.get('sim_threshold_used')}"

        # Should have computed alias scores
        alias_scores = decision.debug.get('alias_scores', {})
        assert len(alias_scores) > 0, f"No alias scores computed. Debug: {decision.debug}"

        # At least the Python topic should have alias hits
        python_alias_score = alias_scores.get('python-retry-patterns', 0)
        assert python_alias_score >= 2, \
            f"Expected alias score >= 2, got {python_alias_score}. Scores: {alias_scores}"
