"""
CLI integration tests for topic reactivation.

These tests verify the actual flow that runs when a user types in episodic:
1. ConversationManager.probe_topic_reactivation gets called
2. probe_reactivation (the core logic) returns REACTIVATE
3. The "🔄 Resuming topic:" message would appear

This is a CI gate - if these tests fail, reactivation is broken.
"""

import os
import tempfile
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch
import uuid

import numpy as np
import pytest


# Mark all tests in this module as reactivation CI gate tests
pytestmark = pytest.mark.reactivation


class TestReactivationCLIFlow:
    """Integration tests for CLI reactivation flow."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.db_path = str(tmp_path / "test.db")
        self.chroma_path = str(tmp_path / "chroma")

        # Set environment
        os.environ['EPISODIC_DB_PATH'] = self.db_path
        os.environ['EPISODIC_CHROMA_PATH'] = self.chroma_path

        # Initialize database
        from episodic.db import initialize_db
        initialize_db()

        # Create embedding mock
        self.embedding_store = {}
        self.python_base = np.array([1.0] + [0.0] * 383)
        self.sourdough_base = np.array([0.0, 1.0] + [0.0] * 382)

    def get_embedding(self, text: str):
        """Generate embeddings based on content."""
        text_lower = text.lower()
        if "python" in text_lower or "retry" in text_lower or "tenacity" in text_lower:
            base = self.python_base
        elif "sourdough" in text_lower or "bread" in text_lower:
            base = self.sourdough_base
        else:
            base = np.random.randn(384)
        emb = base + np.random.randn(384) * 0.05
        return (emb / np.linalg.norm(emb)).tolist()

    def create_mock_rag(self):
        """Create mock RAG with embedding support."""
        mock_collection = MagicMock()

        def mock_add(ids, documents, embeddings=None, metadatas=None):
            for i, node_id in enumerate(ids):
                if embeddings:
                    self.embedding_store[node_id] = np.array(embeddings[i])
                elif documents:
                    self.embedding_store[node_id] = np.array(self.get_embedding(documents[i]))

        def mock_get(ids, include=None):
            result_ids = []
            result_embeddings = []
            for node_id in ids:
                if node_id in self.embedding_store:
                    result_ids.append(node_id)
                    result_embeddings.append(self.embedding_store[node_id])
            return {'ids': result_ids, 'embeddings': result_embeddings}

        def mock_query(query_embeddings, n_results=10, include=None, where=None):
            if not self.embedding_store:
                return {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}

            query_emb = np.array(query_embeddings[0])
            results = []
            for node_id, emb in self.embedding_store.items():
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
                results.append((node_id, 1 - sim))
            results.sort(key=lambda x: x[1])
            results = results[:n_results]
            return {
                'ids': [[r[0] for r in results]],
                'distances': [[r[1] for r in results]],
                'metadatas': [[{}] * len(results)]
            }

        mock_collection.add = mock_add
        mock_collection.get = mock_get
        mock_collection.query = mock_query
        mock_collection._embedding_function = lambda texts: [self.get_embedding(t) for t in texts]

        mock_rag = MagicMock()
        mock_rag.get_collection.return_value = mock_collection
        return mock_rag

    def add_dummy_nodes(self, conn: sqlite3.Connection, count: int = 10):
        """Add dummy nodes to increase turn index for dormancy testing."""
        for i in range(count):
            node_id = str(uuid.uuid4())[:36]
            short_id = str(uuid.uuid4())[:8]
            conn.execute("""
                INSERT INTO nodes (id, short_id, content, role, parent_id)
                VALUES (?, ?, ?, 'user', NULL)
            """, (node_id, f"dm_{short_id}", f"Dummy node {i}"))
        conn.commit()

    def create_topic_with_centroid(
        self,
        conn: sqlite3.Connection,
        name: str,
        content: str,
        is_dormant: bool = True,
        exchange_count: int = 3
    ) -> str:
        """Create a topic with centroid entry and multiple exchanges."""
        user_node_ids = []
        prev_node_id = None

        # Create multiple exchanges for support check
        for i in range(exchange_count):
            node_id = str(uuid.uuid4())[:36]
            asst_node_id = str(uuid.uuid4())[:36]
            exchange_short_id = str(uuid.uuid4())[:8]

            # Create user node with related content
            related_content = f"{content} (exchange {i+1})"
            conn.execute("""
                INSERT INTO nodes (id, short_id, content, role, parent_id)
                VALUES (?, ?, ?, 'user', ?)
            """, (node_id, f"u_{exchange_short_id}", related_content, prev_node_id))

            conn.execute("""
                INSERT INTO nodes (id, short_id, content, role, parent_id)
                VALUES (?, ?, ?, 'assistant', ?)
            """, (asst_node_id, f"a_{exchange_short_id}", f"Response about {name} {i+1}", node_id))

            # Store embedding
            self.embedding_store[node_id] = np.array(self.get_embedding(related_content))

            # Get rowid
            cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,))
            rowid = cursor.fetchone()[0]

            user_node_ids.append((node_id, rowid))
            prev_node_id = asst_node_id

        # First node is topic start
        start_node_id = user_node_ids[0][0]
        start_rowid = user_node_ids[0][1]

        # Create topic
        end_node_id = user_node_ids[-1][0] if is_dormant else None
        conn.execute("""
            INSERT INTO topics (name, start_node_id, created_at, end_node_id)
            VALUES (?, ?, ?, ?)
        """, (name, start_node_id, datetime.now().isoformat(), end_node_id))

        # Create topic_nodes for all exchanges
        for node_id, rowid in user_node_ids:
            conn.execute("""
                INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role)
                VALUES (?, ?, ?, 'user')
            """, (start_node_id, node_id, rowid))

        # Create centroid
        if is_dormant:
            last_active = start_rowid
        else:
            cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
            max_rowid = cursor.fetchone()[0] or start_rowid
            last_active = max_rowid + 10

        # Use middle exchange as medoid
        medoid_idx = len(user_node_ids) // 2
        medoid_node_id = user_node_ids[medoid_idx][0]

        conn.execute("""
            INSERT INTO topic_centroids
            (start_node_id, centroid_medoid_exchange_id, exchange_count,
             last_active_turn_idx, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (start_node_id, medoid_node_id, exchange_count, last_active, datetime.now().isoformat()))

        conn.commit()
        return start_node_id

    def test_probe_reactivation_returns_reactivate(self):
        """
        CI GATE: probe_reactivation must return REACTIVATE for resume queries.

        This is the core logic. If this fails, the feature is broken.
        """
        mock_rag = self.create_mock_rag()

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            from episodic.db_connection import get_connection
            from episodic.recall.reactivation import probe_reactivation, DORMANCY_MIN

            with get_connection() as conn:
                # Create dormant Python topic FIRST
                python_node_id = self.create_topic_with_centroid(
                    conn, "python-retry-patterns",
                    "How do I implement Python retry with tenacity?",
                    is_dormant=True
                )

                # Add dummy nodes to create enough turn gap for dormancy
                self.add_dummy_nodes(conn, count=DORMANCY_MIN + 2)

                # Create active sourdough topic
                sourdough_node_id = self.create_topic_with_centroid(
                    conn, "sourdough-baking",
                    "What's the best way to proof sourdough bread?",
                    is_dormant=False
                )

            # Query "Back to Python retry"
            user_input = "Back to that Python retry thing - should I use tenacity?"
            user_embedding = np.array(self.get_embedding(user_input))

            decision = probe_reactivation(
                user_input=user_input,
                user_embedding=user_embedding,
                active_topic_start_node_id=sourdough_node_id,
                cooldown_turns=0,
                now=datetime.now(),
                recent_nodes=[]
            )

            # MUST return REACTIVATE for Python topic
            assert decision.action == "REACTIVATE", \
                f"Expected REACTIVATE, got {decision.action}. Debug: {decision.debug}"
            assert "python" in decision.topic_name.lower(), \
                f"Expected Python topic, got {decision.topic_name}"

            # Verify all expected gates passed
            assert 'support' in decision.debug.get('gates_passed', []), \
                "Support gate should pass"

    def test_conversation_manager_probe_returns_true(self):
        """
        CI GATE: ConversationManager.probe_topic_reactivation must return True.

        This tests the actual method called in chat().
        """
        mock_rag = self.create_mock_rag()

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            from episodic.db_connection import get_connection
            from episodic.config import config
            from episodic.conversation import ConversationManager
            from episodic.recall.reactivation import DORMANCY_MIN

            # Enable reactivation
            config.set("enable_topic_reactivation", True)

            with get_connection() as conn:
                # Create dormant Python topic
                python_node_id = self.create_topic_with_centroid(
                    conn, "python-retry-patterns",
                    "How do I implement Python retry with tenacity?",
                    is_dormant=True
                )

                # Add dummy nodes
                self.add_dummy_nodes(conn, count=DORMANCY_MIN + 2)

                # Create active sourdough topic
                sourdough_node_id = self.create_topic_with_centroid(
                    conn, "sourdough-baking",
                    "What's the best way to proof sourdough bread?",
                    is_dormant=False
                )

            # Create conversation manager
            conv = ConversationManager()
            conv.set_current_topic("sourdough-baking", sourdough_node_id)

            # Probe with resume query
            user_input = "Back to that Python retry thing - should I use tenacity?"

            should_reactivate, topic_name, topic_start_node_id = conv.probe_topic_reactivation(
                user_input=user_input,
                recent_nodes=[],
                is_meta_query=False,
                is_recall_intent=False
            )

            # MUST reactivate
            assert should_reactivate, \
                f"Expected should_reactivate=True. Decision: {getattr(conv, '_last_reactivation_decision', None)}"
            assert topic_name and "python" in topic_name.lower(), \
                f"Expected Python topic, got {topic_name}"

    def test_two_channel_gate_alias_matching(self):
        """
        CI GATE: Two-channel gate must detect alias matches.

        Verifies that "Back to Python retry" triggers alias matching
        even if semantic similarity is moderate.
        """
        mock_rag = self.create_mock_rag()

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            from episodic.db_connection import get_connection
            from episodic.recall.reactivation import probe_reactivation, DORMANCY_MIN

            with get_connection() as conn:
                python_node_id = self.create_topic_with_centroid(
                    conn, "python-retry-patterns",
                    "How do I implement Python retry with tenacity?",
                    is_dormant=True
                )
                self.add_dummy_nodes(conn, count=DORMANCY_MIN + 2)
                sourdough_node_id = self.create_topic_with_centroid(
                    conn, "sourdough-baking",
                    "What's the best way to proof sourdough bread?",
                    is_dormant=False
                )

            user_input = "Back to that Python retry thing - should I use tenacity?"
            user_embedding = np.array(self.get_embedding(user_input))

            decision = probe_reactivation(
                user_input=user_input,
                user_embedding=user_embedding,
                active_topic_start_node_id=sourdough_node_id,
                cooldown_turns=0,
                now=datetime.now(),
                recent_nodes=[]
            )

            # Verify resume cues detected
            assert decision.debug.get('resume_cues_detected') is True, \
                "Resume cues should be detected"

            # Verify alias matching worked
            alias_scores = decision.debug.get('alias_scores', {})
            assert 'python-retry-patterns' in alias_scores, \
                f"Python topic should have alias score. Scores: {alias_scores}"
            assert alias_scores['python-retry-patterns'] >= 2, \
                f"Expected alias score >= 2, got {alias_scores['python-retry-patterns']}"

            # Verify channel B passed
            assert decision.debug.get('channel_b_pass') is True, \
                "Channel B (alias matching) should pass"
