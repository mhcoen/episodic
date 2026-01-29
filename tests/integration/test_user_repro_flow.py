"""
User repro test for topic reactivation.

This tests the EXACT flow the user reported:
1. "How do I handle retry logic in Python with exponential backoff?"
2. "Let's talk about sourdough bread. What's a good starter ratio?"
3. "How long should I proof the dough?"
4. "Back to that Python retry thing - should I use tenacity?"

Step 4 must trigger reactivation.
"""

import os
import tempfile
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch
import uuid

import numpy as np
import pytest


pytestmark = pytest.mark.reactivation


class TestUserReproFlow:
    """Test the exact user repro scenario."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.db_path = str(tmp_path / "test.db")
        self.chroma_path = str(tmp_path / "chroma")

        os.environ['EPISODIC_DB_PATH'] = self.db_path
        os.environ['EPISODIC_CHROMA_PATH'] = self.chroma_path

        from episodic.db import initialize_db
        initialize_db()

        self.embedding_store = {}
        self.python_base = np.array([1.0] + [0.0] * 383)
        self.sourdough_base = np.array([0.0, 1.0] + [0.0] * 382)

    def get_embedding(self, text: str):
        """Generate embeddings based on content."""
        text_lower = text.lower()
        if "python" in text_lower or "retry" in text_lower or "tenacity" in text_lower or "backoff" in text_lower:
            base = self.python_base
        elif "sourdough" in text_lower or "bread" in text_lower or "proof" in text_lower or "dough" in text_lower or "starter" in text_lower:
            base = self.sourdough_base
        else:
            base = np.random.randn(384)
        emb = base + np.random.randn(384) * 0.05
        return (emb / np.linalg.norm(emb)).tolist()

    def create_mock_rag(self):
        """Create mock RAG with embedding support."""
        mock_collection = MagicMock()

        def mock_get(ids, include=None):
            result_ids = []
            result_embeddings = []
            for node_id in ids:
                if node_id in self.embedding_store:
                    result_ids.append(node_id)
                    result_embeddings.append(self.embedding_store[node_id])
            return {'ids': result_ids, 'embeddings': result_embeddings}

        mock_collection.get = mock_get
        mock_collection._embedding_function = lambda texts: [self.get_embedding(t) for t in texts]

        mock_rag = MagicMock()
        mock_rag.get_collection.return_value = mock_collection
        return mock_rag

    def create_single_exchange_topic(
        self,
        conn: sqlite3.Connection,
        name: str,
        user_content: str,
        is_closed: bool = True
    ) -> str:
        """Create a topic with a SINGLE exchange (realistic scenario)."""
        short_id = str(uuid.uuid4())[:8]
        node_id = str(uuid.uuid4())[:36]
        asst_id = str(uuid.uuid4())[:36]

        conn.execute("""
            INSERT INTO nodes (id, short_id, content, role, parent_id)
            VALUES (?, ?, ?, 'user', NULL)
        """, (node_id, f"u_{short_id}", user_content))

        conn.execute("""
            INSERT INTO nodes (id, short_id, content, role, parent_id)
            VALUES (?, ?, ?, 'assistant', ?)
        """, (asst_id, f"a_{short_id}", f"Response about {name}", node_id))

        # Store embedding
        self.embedding_store[node_id] = np.array(self.get_embedding(user_content))

        # Get rowid
        cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,))
        rowid = cursor.fetchone()[0]

        # Create topic
        end_node_id = node_id if is_closed else None
        conn.execute("""
            INSERT INTO topics (name, start_node_id, created_at, end_node_id)
            VALUES (?, ?, ?, ?)
        """, (name, node_id, datetime.now().isoformat(), end_node_id))

        # Create topic_nodes
        conn.execute("""
            INSERT INTO topic_nodes (topic_start_node_id, node_id, turn_idx, role)
            VALUES (?, ?, ?, 'user')
        """, (node_id, node_id, rowid))

        # Create centroid - set last_active to this rowid
        conn.execute("""
            INSERT INTO topic_centroids
            (start_node_id, centroid_medoid_exchange_id, exchange_count,
             last_active_turn_idx, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (node_id, node_id, 1, rowid, datetime.now().isoformat()))

        conn.commit()
        return node_id

    def test_exact_user_repro(self):
        """
        Test the EXACT user repro scenario:

        1. "How do I handle retry logic in Python with exponential backoff?"
        2. "Let's talk about sourdough bread. What's a good starter ratio?"
        3. "How long should I proof the dough?"
        4. "Back to that Python retry thing - should I use tenacity?"

        Step 4 MUST trigger reactivation to Python topic.
        """
        mock_rag = self.create_mock_rag()

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            from episodic.db_connection import get_connection
            from episodic.recall.reactivation import probe_reactivation, DORMANCY_MIN

            with get_connection() as conn:
                # Step 1: Python question (creates python topic, then closes it)
                python_node_id = self.create_single_exchange_topic(
                    conn, "python-retry-patterns",
                    "How do I handle retry logic in Python with exponential backoff?",
                    is_closed=True
                )

                # Step 2: Sourdough question (new topic)
                sourdough_node_id = self.create_single_exchange_topic(
                    conn, "sourdough-baking",
                    "Let's talk about sourdough bread. What's a good starter ratio?",
                    is_closed=False  # Still active
                )

                # Step 3: Proofing follow-up (same sourdough topic, adds turn)
                # Add this as another exchange in the sourdough topic
                follow_up_id = str(uuid.uuid4())[:36]
                follow_up_asst = str(uuid.uuid4())[:36]
                short_id = str(uuid.uuid4())[:8]

                conn.execute("""
                    INSERT INTO nodes (id, short_id, content, role, parent_id)
                    VALUES (?, ?, ?, 'user', ?)
                """, (follow_up_id, f"u_{short_id}", "How long should I proof the dough?", sourdough_node_id))

                conn.execute("""
                    INSERT INTO nodes (id, short_id, content, role, parent_id)
                    VALUES (?, ?, ?, 'assistant', ?)
                """, (follow_up_asst, f"a_{short_id}", "Proof for 4-6 hours at room temp.", follow_up_id))

                self.embedding_store[follow_up_id] = np.array(self.get_embedding("How long should I proof the dough?"))

                # Get the current max rowid (this is the "current turn")
                cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
                current_turn = cursor.fetchone()[0]

                # Update sourdough topic centroid to current
                conn.execute("""
                    UPDATE topic_centroids SET last_active_turn_idx = ?
                    WHERE start_node_id = ?
                """, (current_turn, sourdough_node_id))

                conn.commit()

                # Debug: Check dormancy
                cursor = conn.execute("""
                    SELECT start_node_id, last_active_turn_idx FROM topic_centroids
                """)
                print("\n=== Topic Centroids ===")
                for row in cursor.fetchall():
                    print(f"  {row[0][:20]}... last_active={row[1]}")
                print(f"  Current turn: {current_turn}")

            # Step 4: Resume Python
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

            print(f"\n=== Reactivation Decision ===")
            print(f"  action: {decision.action}")
            print(f"  topic_name: {decision.topic_name}")
            print(f"  debug: {decision.debug}")

            # Key assertions for the user's exact repro
            assert decision.debug.get('resume_cues_detected') is True, \
                "Resume cues should be detected in 'Back to that Python retry thing'"

            # Check dormancy passed
            assert 'no_dormant_topics' not in decision.debug.get('exit_reason', ''), \
                f"Python topic should be dormant. Debug: {decision.debug}"

            # Check support passed (with lowered threshold for channel B)
            assert 'insufficient_support' not in decision.debug.get('exit_reason', ''), \
                f"Support should pass with 1 exchange when channel B passes. Debug: {decision.debug}"

            # The key assertion: MUST reactivate
            assert decision.action == "REACTIVATE", \
                f"Expected REACTIVATE, got {decision.action}. Debug: {decision.debug}"

            assert "python" in decision.topic_name.lower(), \
                f"Expected Python topic, got {decision.topic_name}"

            print("\n✓ User repro passed! Reactivation fires correctly.")
