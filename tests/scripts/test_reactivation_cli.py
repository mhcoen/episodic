#!/usr/bin/env python3
"""
Integration test for topic reactivation CLI flow.

This script tests the ACTUAL end-to-end flow by directly exercising
the ConversationManager's probe_topic_reactivation method.

Run with: python tests/scripts/test_reactivation_cli.py
"""

import os
import sys
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReactivationCLIFlow:
    """Integration tests for CLI reactivation flow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.chroma_path = os.path.join(self.tmpdir, "chroma")

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
        """Create a topic with centroid entry and multiple exchanges for reactivation testing."""
        # Create unique short_id using uuid
        short_id = str(uuid.uuid4())[:8]

        # Create first user node (topic start)
        start_node_id = str(uuid.uuid4())[:36]
        asst_id = str(uuid.uuid4())[:36]
        prev_node_id = None
        user_node_ids = []

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

            # Store embedding (similar to topic content)
            self.embedding_store[node_id] = np.array(self.get_embedding(related_content))

            # Get rowid
            cursor = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_id,))
            rowid = cursor.fetchone()[0]

            # Track first node as topic start
            if i == 0:
                start_node_id = node_id
                start_rowid = rowid

            user_node_ids.append((node_id, rowid))
            prev_node_id = asst_node_id

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
        # For dormant topic: set last_active_turn_idx to the start (so it's old)
        # For active topic: get current max turn and set to that (recent)
        if is_dormant:
            last_active = start_rowid
        else:
            cursor = conn.execute("SELECT MAX(rowid) FROM nodes")
            max_rowid = cursor.fetchone()[0] or start_rowid
            last_active = max_rowid + 10  # Very recent

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

    def test_probe_reactivation_direct(self):
        """
        Test that probe_reactivation returns REACTIVATE for resume queries.

        This tests the core logic that must work.
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
                # Need at least DORMANCY_MIN (4) turns between Python and current
                self.add_dummy_nodes(conn, count=DORMANCY_MIN + 2)

                # Create active sourdough topic (will have higher turn index)
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

            print(f"\n=== probe_reactivation result ===")
            print(f"  action: {decision.action}")
            print(f"  topic_name: {decision.topic_name}")
            print(f"  debug: {decision.debug}")

            # MUST return REACTIVATE for Python topic
            assert decision.action == "REACTIVATE", \
                f"Expected REACTIVATE, got {decision.action}. Debug: {decision.debug}"
            assert "python" in decision.topic_name.lower(), \
                f"Expected Python topic, got {decision.topic_name}"

            print("✓ probe_reactivation works correctly!")
            return True

    def test_conversation_manager_probe(self):
        """
        Test ConversationManager.probe_topic_reactivation flow.

        This tests the actual method that gets called in chat().
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

            # Create conversation manager
            conv = ConversationManager()

            # Set current topic to sourdough (simulating user is in sourdough conversation)
            conv.set_current_topic("sourdough-baking", sourdough_node_id)

            # Probe with resume query
            user_input = "Back to that Python retry thing - should I use tenacity?"

            should_reactivate, topic_name, topic_start_node_id = conv.probe_topic_reactivation(
                user_input=user_input,
                recent_nodes=[],
                is_meta_query=False,
                is_recall_intent=False
            )

            print(f"\n=== probe_topic_reactivation result ===")
            print(f"  should_reactivate: {should_reactivate}")
            print(f"  topic_name: {topic_name}")
            print(f"  topic_start_node_id: {topic_start_node_id}")

            # Check the stored decision
            decision = getattr(conv, '_last_reactivation_decision', None)
            if decision:
                print(f"  decision.action: {decision.action}")
                print(f"  decision.debug: {decision.debug}")

            # MUST reactivate
            assert should_reactivate, \
                f"Expected should_reactivate=True, got False. Decision: {decision.debug if decision else 'None'}"
            assert topic_name and "python" in topic_name.lower(), \
                f"Expected Python topic, got {topic_name}"

            print("✓ ConversationManager.probe_topic_reactivation works correctly!")
            return True


def run_tests():
    """Run all tests."""
    test = TestReactivationCLIFlow()

    print("=" * 60)
    print("REACTIVATION CLI INTEGRATION TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: Direct probe_reactivation
    print("\n[1] Testing probe_reactivation directly...")
    try:
        test.setup_method()
        test.test_probe_reactivation_direct()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: ConversationManager probe
    print("\n[2] Testing ConversationManager.probe_topic_reactivation...")
    try:
        test.setup_method()
        test.test_conversation_manager_probe()
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
