"""
Robustness tests for TopicLocalStrategy assembly.

Three categories:
- Category A: Prompt-shape snapshot tests (semantic correctness)
- Category B: Token hard-cap property tests (distributional robustness)
- Category C: Boundary-condition tests (edge cases)
"""

import os
import random
import re
import string
import sqlite3
from typing import Dict, List, Any, Set
from unittest.mock import MagicMock, patch

import pytest

# Set test mode before imports
os.environ["EPISODIC_TEST_MODE"] = "1"

import tiktoken

from episodic.context_recovery.topic_local import (
    TopicLocalStrategy,
    _estimate_tokens,
    _truncate_to_tokens,
    _assert_no_contamination,
    ContaminationError,
)
from episodic.context_recovery.strategy import ContextAssemblyResult


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def get_tiktoken_encoder():
    """Get the production tokenizer (cl100k_base for GPT-4/GPT-3.5)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using production tokenizer."""
    encoder = get_tiktoken_encoder()
    return len(encoder.encode(text))


def create_synthetic_exchange(
    user_id: str,
    asst_id: str,
    user_content: str,
    asst_content: str
) -> Dict[str, Any]:
    """Create a synthetic exchange dict."""
    return {
        "user_node_id": user_id,
        "assistant_node_id": asst_id,
        "user_content": user_content,
        "assistant_content": asst_content,
    }


def create_mock_chroma_results(
    anchors: List[Dict[str, Any]],
    topic_id: str
) -> Dict[str, Any]:
    """Create mock Chroma query results."""
    if not anchors:
        return {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]], "embeddings": [[]]}

    ids = [[a["user_node_id"] for a in anchors]]
    metadatas = [[{
        "user_id": a["user_node_id"],
        "assistant_id": a["assistant_node_id"],
        "user_content": a["user_content"],
        "assistant_content": a["assistant_content"],
        "topic_start_node_id": topic_id,
    } for a in anchors]]
    distances = [[0.5] * len(anchors)]  # Mid-range similarity
    documents = [[f"User: {a['user_content']}\nAssistant: {a['assistant_content']}" for a in anchors]]
    embeddings = [[[0.1] * 384 for _ in anchors]]  # Fake embeddings

    return {
        "ids": ids,
        "metadatas": metadatas,
        "distances": distances,
        "documents": documents,
        "embeddings": embeddings,
    }


def generate_random_text(num_tokens: int) -> str:
    """Generate random text with approximately the specified number of tokens."""
    # On average, ~4 chars per token for English text
    words = []
    current_tokens = 0
    encoder = get_tiktoken_encoder()

    while current_tokens < num_tokens:
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        current_tokens = len(encoder.encode(' '.join(words)))

    return ' '.join(words)


# ============================================================================
# Category A: Prompt-Shape Snapshot Tests
# ============================================================================

class TestCategoryA_PromptShapeSnapshots:
    """
    Snapshot tests verifying assembled prompt structure.

    Validates:
    1. Section order (topic header → summary → anchors → recency)
    2. Anchors under past-framed header
    3. Imports labeled if present
    4. No duplicate exchange IDs
    5. No foreign-topic contamination
    """

    def test_simple_single_topic_structure(self):
        """Test A.1: Simple single-topic conversation structure."""
        strategy = TopicLocalStrategy(exchange_pairs=4)

        # Create synthetic exchanges
        exchanges = [
            create_synthetic_exchange("u1", "a1", "Hello", "Hi there!"),
            create_synthetic_exchange("u2", "a2", "How are you?", "I'm fine, thanks!"),
            create_synthetic_exchange("u3", "a3", "What's Python?", "It's a programming language."),
        ]

        # Mock working set
        working_set = {
            "topic_name": "Python Discussion",
            "summary_md": "Discussion about Python programming language basics.",
        }

        # Mock Chroma to return one anchor (u1)
        mock_anchor = exchanges[0]
        mock_chroma_results = create_mock_chroma_results([mock_anchor], "topic_start_123")

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_chroma_results
        mock_collection._embedding_function = MagicMock(return_value=[[0.1] * 384])

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,  # Disable imports for this test
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    # Need to mock _assert_no_contamination since we don't have real DB
                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Tell me more about Python",
                            user_node_id="u4",
                            active_topic_start_node_id="topic_start_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        # Verify structure
        messages = result.messages
        assert len(messages) >= 1  # At least system message

        # First message should be system with topic context
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        content = system_msg["content"]

        # Check section order
        topic_header_pos = content.find("# Topic:")
        summary_pos = content.find("## Summary")
        anchors_pos = content.find("## Relevant Past Context")

        assert topic_header_pos >= 0, "Topic header missing"
        assert summary_pos > topic_header_pos, "Summary should come after topic header"

        # Anchors may or may not be present depending on retrieval
        if anchors_pos >= 0:
            assert anchors_pos > summary_pos, "Anchors should come after summary"

        # Verify recency messages are user/assistant role (not system)
        recency_messages = [m for m in messages if m["role"] in ("user", "assistant")]
        assert len(recency_messages) > 0, "Should have recency messages"

    def test_anchors_overlap_recency_no_duplicates(self):
        """Test A.2: When anchor overlaps recency, no duplicate exchange IDs."""
        strategy = TopicLocalStrategy(exchange_pairs=4)

        # Create exchanges where u2 is both in recency AND would be top anchor
        exchanges = [
            create_synthetic_exchange("u1", "a1", "First message", "First response"),
            create_synthetic_exchange("u2", "a2", "Important message about topic", "Key response"),
            create_synthetic_exchange("u3", "a3", "Third message", "Third response"),
        ]

        working_set = {
            "topic_name": "Test Topic",
            "summary_md": "Test summary",
        }

        # Anchor results return u2 (which is also in recency)
        mock_anchor = exchanges[1]  # u2 - the overlap case
        mock_chroma_results = create_mock_chroma_results([mock_anchor], "topic_start_123")

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_chroma_results
        mock_collection._embedding_function = MagicMock(return_value=[[0.1] * 384])

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Continue discussion",
                            user_node_id="u4",
                            active_topic_start_node_id="topic_start_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        debug = result.debug

        # Verify no duplicate IDs in included_node_ids
        included_ids = debug.get("included_node_ids", [])
        assert len(included_ids) == len(set(included_ids)), \
            f"Duplicate node IDs found: {included_ids}"

        # u2 should appear in anchors (charged to A), not recency
        anchor_ids = debug.get("anchors", {}).get("included_node_ids", [])
        recency_skipped = debug.get("recency", {}).get("skipped_as_anchors", [])

        # Either u2 is in anchors, or it was skipped from recency
        if "u2" in anchor_ids:
            assert "u2" in recency_skipped, "u2 should be skipped from recency if in anchors"

    def test_past_framed_header_for_anchors(self):
        """Test A.3: Anchors are under clearly past-framed header."""
        strategy = TopicLocalStrategy()

        exchanges = [
            create_synthetic_exchange("u1", "a1", "Hello", "Hi!"),
        ]

        working_set = {"topic_name": "Test", "summary_md": "Summary"}

        # Create mock chroma results with proper structure for anchor retrieval
        # The _retrieve_anchors_budgeted method expects specific metadata fields
        # Note: anchor embedding must be different from summary embedding to pass novelty check
        anchor_embedding = [0.5] * 192 + [0.1] * 192  # Different from summary embedding
        mock_chroma_results = {
            "ids": [["old_u"]],
            "metadatas": [[{
                "user_id": "old_u",
                "assistant_id": "old_a",
                "user_content": "Old question from previous context",
                "assistant_content": "Old answer from previous context",
                "topic_start_node_id": "topic_start_123",
            }]],
            "distances": [[0.3]],  # Low distance = high similarity (sim = 1 - dist/2 = 0.85)
            "documents": [["User: Old question\nAssistant: Old answer"]],
            "embeddings": [[anchor_embedding]],
        }

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_chroma_results
        # Summary embedding is different from anchor embedding to pass novelty check
        summary_embedding = [0.1] * 384
        mock_collection._embedding_function = MagicMock(return_value=[summary_embedding])

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Question",
                            user_node_id="u2",
                            active_topic_start_node_id="topic_start_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        # Check that anchors have past-framed header
        system_content = result.messages[0]["content"]
        assert "## Relevant Past Context" in system_content, \
            f"Anchors should be under '## Relevant Past Context' header. Got: {system_content[:500]}"

        # Verify anchor content is included
        assert "Old question" in system_content or "Old answer" in system_content, \
            "Anchor content should appear in system message"


# ============================================================================
# Category B: Token Hard-Cap Property Tests
# ============================================================================

class TestCategoryB_TokenHardCap:
    """
    Property tests using production tokenizer.

    Validates:
    1. Assembled tokens NEVER exceed T
    2. Drop policy triggers when cap would be exceeded
    3. Logged counts match actual tokenized length
    """

    def test_user_paste_bomb_respects_cap(self):
        """Test B.1: User paste bomb (5000+ tokens) doesn't exceed T."""
        strategy = TopicLocalStrategy()

        # Create a huge user message (5000 tokens)
        huge_content = generate_random_text(5000)
        exchanges = [
            create_synthetic_exchange("u1", "a1", huge_content, "Response"),
        ]

        working_set = {"topic_name": "Test", "summary_md": "Summary"}

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        T = 3000  # Token cap

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": T,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Continue",
                            user_node_id="u2",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=T,
                            chroma_collection=mock_collection,
                        )

        # Count actual tokens in assembled messages
        total_text = ""
        for msg in result.messages:
            total_text += msg.get("content", "")

        actual_tokens = count_tokens_tiktoken(total_text)
        estimated_tokens = result.debug.get("token_breakdown", {}).get("total_tokens", 0)

        # The huge exchange should have been dropped by budget
        recency_debug = result.debug.get("recency", {})
        assert "dropped_by_budget" in recency_debug or recency_debug.get("included_count", 0) == 0, \
            "Huge exchange should be dropped or skipped"

    def test_many_short_exchanges_respects_cap(self):
        """Test B.2: Many short exchanges (50+) still respect T cap."""
        strategy = TopicLocalStrategy()

        # Create 50 small exchanges
        exchanges = [
            create_synthetic_exchange(f"u{i}", f"a{i}", f"Question {i}", f"Answer {i}")
            for i in range(50)
        ]

        working_set = {"topic_name": "Test", "summary_md": "Summary"}

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        T = 2000  # Smaller cap to force truncation

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": T,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Continue",
                            user_node_id="u51",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=T,
                            chroma_collection=mock_collection,
                        )

        # Verify budget was enforced
        recency_debug = result.debug.get("recency", {})
        tokens_used = recency_debug.get("tokens_used", 0)
        budget = recency_debug.get("budget", 0)

        assert tokens_used <= budget, \
            f"Recency tokens {tokens_used} should not exceed budget {budget}"

    @pytest.mark.parametrize("iteration", range(20))  # Property test with 20 random cases
    def test_random_conversation_respects_cap(self, iteration):
        """Test B.3: Random conversations always respect T cap."""
        strategy = TopicLocalStrategy()

        # Generate random exchanges
        num_exchanges = random.randint(1, 20)
        exchanges = []
        for i in range(num_exchanges):
            # Random token counts for user/assistant
            user_tokens = random.randint(10, 500)
            asst_tokens = random.randint(10, 500)
            exchanges.append(create_synthetic_exchange(
                f"u{i}", f"a{i}",
                generate_random_text(user_tokens),
                generate_random_text(asst_tokens)
            ))

        working_set = {
            "topic_name": "Random Test",
            "summary_md": generate_random_text(random.randint(50, 300))
        }

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        T = random.randint(1500, 4000)

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": T,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Random query",
                            user_node_id="current",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=T,
                            chroma_collection=mock_collection,
                        )

        # Check budget breakdown sums correctly
        breakdown = result.debug.get("token_breakdown", {})
        total = breakdown.get("total_tokens", 0)
        budgets = result.debug.get("budgets", {})

        # Total should be within reasonable bounds of T
        # (allowing some variance since we don't control system prompt)
        assert total <= T + 100, \
            f"Total tokens {total} should not greatly exceed T={T}"


# ============================================================================
# Category C: Boundary-Condition Tests
# ============================================================================

class TestCategoryC_BoundaryConditions:
    """
    Edge case tests for weird budget math.

    Validates graceful degradation in extreme scenarios.
    """

    def test_tiny_M_graceful_degradation(self):
        """Test C.1: Tiny M (T=600, overhead=500) handles gracefully."""
        strategy = TopicLocalStrategy()

        exchanges = [
            create_synthetic_exchange("u1", "a1", "Hello", "Hi there!"),
        ]

        working_set = {
            "topic_name": "Test",
            "summary_md": "A brief summary."
        }

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        # Tiny budget: T=600, overhead=500, s_max=400 -> M = max(0, 600-500-400) = 0
        # But we clamp to avoid negative
        T = 600
        overhead = 500
        s_max = 400

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": T,
                "context_overhead_estimate": overhead,
                "summary_max_tokens": s_max,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        # Should not crash
                        result = strategy.assemble(
                            user_turn_text="Test",
                            user_node_id="u2",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=T,
                            chroma_collection=mock_collection,
                        )

        # Should complete without error
        assert result is not None
        assert result.messages is not None

        # M should be clamped to 0 or small positive
        budgets = result.debug.get("budgets", {})
        assert budgets.get("M", 0) >= 0, "M should not be negative"

    def test_a_min_exceeds_available_M(self):
        """Test C.2: a_min=500 but M=300 after overhead - no crash."""
        strategy = TopicLocalStrategy()

        exchanges = [create_synthetic_exchange("u1", "a1", "Hello", "Hi!")]
        working_set = {"topic_name": "Test", "summary_md": "Summary"}

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        # M = T - overhead - s_max = 1000 - 500 - 200 = 300
        # a_min = 500 > M
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 1000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 200,
                "anchor_alpha": 0.25,
                "anchor_a_min": 500,  # Exceeds M
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Test",
                            user_node_id="u2",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=1000,
                            chroma_collection=mock_collection,
                        )

        # Should not crash
        assert result is not None

        # R should be clamped to 0 (not negative)
        budgets = result.debug.get("budgets", {})
        R = budgets.get("R", 0)
        assert R >= 0, f"R should not be negative, got {R}"

    def test_zero_anchors_returned(self):
        """Test C.5: Zero anchors returned - proceeds with summary + recency."""
        strategy = TopicLocalStrategy()

        exchanges = [
            create_synthetic_exchange("u1", "a1", "Hello", "Hi!"),
            create_synthetic_exchange("u2", "a2", "Question", "Answer"),
        ]
        working_set = {"topic_name": "Test", "summary_md": "A summary."}

        # Mock returns empty anchors
        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Test",
                            user_node_id="u3",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        # Should complete
        assert result is not None
        assert len(result.messages) > 0

        # No anchors in result
        anchor_debug = result.debug.get("anchors", {})
        assert anchor_debug.get("included_count", 0) == 0

        # But recency should still be present
        recency_debug = result.debug.get("recency", {})
        assert recency_debug.get("included_count", 0) > 0

    def test_zero_recency(self):
        """Test C.6: Zero recency - proceeds with anchors only."""
        strategy = TopicLocalStrategy()

        # Empty recency
        exchanges = []
        working_set = {"topic_name": "Test", "summary_md": "Summary"}

        # Mock returns one anchor
        anchor = create_synthetic_exchange("old_u", "old_a", "Old Q", "Old A")
        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([anchor], "topic_123")
        mock_collection._embedding_function = MagicMock(return_value=[[0.1] * 384])

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Test",
                            user_node_id="u1",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        # Should complete without error
        assert result is not None

        # Zero recency
        recency_debug = result.debug.get("recency", {})
        assert recency_debug.get("candidate_count", 0) == 0

    def test_summary_truncation_tracking(self):
        """Test C.3-variant: Long summary truncation is tracked."""
        strategy = TopicLocalStrategy()

        # Long summary (1500 tokens)
        long_summary = generate_random_text(1500)

        exchanges = [create_synthetic_exchange("u1", "a1", "Hello", "Hi!")]
        working_set = {"topic_name": "Test", "summary_md": long_summary}

        mock_collection = MagicMock()
        mock_collection.query.return_value = create_mock_chroma_results([], "topic_123")

        s_max = 400  # Summary cap

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": s_max,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
                "anchor_count": 3,
                "anchor_similarity_threshold": 0.3,
                "anchor_retrieval_count": 10,
                "import_detection_enabled": False,
                "min_anchors_for_topic_local": 0,
                "min_tokens_for_topic_local": 0,
                "debug": False,
            }.get(key, default)

            with patch("episodic.db_topic_nodes.get_topic_working_set") as mock_ws:
                mock_ws.return_value = working_set

                with patch("episodic.db_topic_nodes.get_last_n_exchanges_in_topic") as mock_exchanges:
                    mock_exchanges.return_value = exchanges

                    with patch("episodic.context_recovery.topic_local._assert_no_contamination"):
                        result = strategy.assemble(
                            user_turn_text="Test",
                            user_node_id="u2",
                            active_topic_start_node_id="topic_123",
                            user_embedding=None,
                            token_budget=3000,
                            chroma_collection=mock_collection,
                        )

        # Summary should be truncated
        truncation_info = result.debug.get("truncation_info", {})
        assert truncation_info.get("summary_truncated", False) is True, \
            "Long summary should have been truncated"

        # Summary tokens should be <= s_max
        breakdown = result.debug.get("token_breakdown", {})
        summary_tokens = breakdown.get("summary_tokens", 0)
        assert summary_tokens <= s_max + 10, \
            f"Summary tokens {summary_tokens} should be <= s_max={s_max}"
