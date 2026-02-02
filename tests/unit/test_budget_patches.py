"""
Acceptance tests for TopicLocalStrategy budget patches.

Tests verify:
- Patch B: Token-based recency truncation
- Patch C: Summary bounding with s_max
- Patch D: Anchors-first selection order (overlap charged to A)
- Patch E: Structured assembly logging
- Patch A: Configurable alpha budget split
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

# Set test mode before imports
os.environ["EPISODIC_TEST_MODE"] = "1"

from episodic.context_recovery.topic_local import (
    TopicLocalStrategy,
    _estimate_tokens,
    _truncate_to_tokens,
    DEFAULT_EXCHANGE_PAIRS,
)


class TestPatchB_RecencyTokenBudget:
    """Patch B: Token-based recency truncation."""

    def test_huge_exchange_truncates_by_tokens(self):
        """Verify that recency truncates by tokens, not just count."""
        strategy = TopicLocalStrategy()

        # Create exchanges where one is huge (2000+ tokens worth)
        huge_content = "x" * 8000  # ~2000 tokens
        small_content = "y" * 400  # ~100 tokens

        all_exchanges = [
            {
                "user_node_id": "exchange1_user",
                "assistant_node_id": "exchange1_asst",
                "user_content": huge_content,
                "assistant_content": huge_content,
            },
            {
                "user_node_id": "exchange2_user",
                "assistant_node_id": "exchange2_asst",
                "user_content": small_content,
                "assistant_content": small_content,
            },
            {
                "user_node_id": "exchange3_user",
                "assistant_node_id": "exchange3_asst",
                "user_content": small_content,
                "assistant_content": small_content,
            },
        ]

        # Budget of 500 tokens - huge exchange should be dropped
        recency_budget = 500
        anchor_node_ids = set()  # No anchors

        messages, node_ids, debug = strategy._build_recency_budgeted(
            all_exchanges=all_exchanges,
            anchor_node_ids=anchor_node_ids,
            recency_budget=recency_budget,
        )

        # The huge exchange should be dropped
        assert "exchange1_user" not in node_ids
        assert debug["tokens_used"] <= recency_budget

        # Smaller exchanges should fit
        included_count = debug["included_count"]
        assert included_count >= 1

    def test_total_assembled_under_target_T(self):
        """Verify total assembled tokens stay under T."""
        strategy = TopicLocalStrategy()

        # Mock config imported inside _compute_budgets
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 1000,
                "context_overhead_estimate": 200,
                "summary_max_tokens": 100,
                "anchor_alpha": 0.25,
                "anchor_a_min": 100,
                "anchor_a_max": 500,
            }.get(key, default)

            budgets = strategy._compute_budgets(token_budget=1000)

            # T=1000, overhead=200, s_max=100 -> M=700
            # A = clamp(100, 0.25*700=175, 500) = 175
            # R = 700 - 175 = 525
            assert budgets["T"] == 1000
            assert budgets["M"] == 700
            assert budgets["A"] == 175
            assert budgets["R"] == 525


class TestPatchC_SummaryBounding:
    """Patch C: Summary truncation at injection."""

    def test_overlong_summary_truncates_at_injection(self):
        """Verify injected summary is truncated to s_max."""
        # Create overlong summary (1500 tokens = 6000 chars)
        long_summary = "This is a test summary. " * 400  # ~1500+ tokens
        s_max = 500

        truncated, was_truncated = _truncate_to_tokens(long_summary.strip(), s_max)

        # Should be truncated
        assert was_truncated is True
        assert _estimate_tokens(truncated) <= s_max + 1  # Allow small margin for "..."

    def test_short_summary_not_truncated(self):
        """Verify short summaries are not truncated."""
        short_summary = "This is a brief summary."
        s_max = 500

        truncated, was_truncated = _truncate_to_tokens(short_summary, s_max)

        assert was_truncated is False
        assert truncated == short_summary


class TestPatchD_AnchorsFirstOrdering:
    """Patch D: Anchors selected first, recency backfills non-anchors."""

    def test_overlap_charged_to_anchors(self):
        """Verify overlap is charged to anchor budget, recency backfills."""
        strategy = TopicLocalStrategy()

        # Create exchanges where one is both recent AND an anchor
        all_exchanges = [
            {
                "user_node_id": "overlap_user",  # This one is both anchor and recent
                "assistant_node_id": "overlap_asst",
                "user_content": "Overlapping exchange content",
                "assistant_content": "Overlapping response",
            },
            {
                "user_node_id": "recency_only_user",
                "assistant_node_id": "recency_only_asst",
                "user_content": "Recency only content",
                "assistant_content": "Recency only response",
            },
        ]

        # Simulate anchor_node_ids containing the overlap
        anchor_node_ids = {"overlap_user", "overlap_asst"}

        messages, node_ids, debug = strategy._build_recency_budgeted(
            all_exchanges=all_exchanges,
            anchor_node_ids=anchor_node_ids,
            recency_budget=1000,
        )

        # Overlap should be skipped (charged to anchors)
        assert "overlap_user" in debug["skipped_as_anchors"]
        assert "overlap_user" not in node_ids

        # Recency should backfill with non-anchor exchange
        assert "recency_only_user" in node_ids

    def test_overlap_appears_exactly_once(self):
        """Verify overlapping exchange appears only once in final context."""
        strategy = TopicLocalStrategy()

        # Overlap node
        overlap_id = "overlap_exchange"

        # Anchors already have the overlap
        anchor_node_ids = {overlap_id}

        # Exchanges include the overlap
        all_exchanges = [
            {
                "user_node_id": overlap_id,
                "assistant_node_id": f"{overlap_id}_asst",
                "user_content": "Overlap content",
                "assistant_content": "Overlap response",
            }
        ]

        messages, node_ids, debug = strategy._build_recency_budgeted(
            all_exchanges=all_exchanges,
            anchor_node_ids=anchor_node_ids,
            recency_budget=1000,
        )

        # Overlap should NOT appear in recency (already in anchors)
        assert overlap_id not in node_ids
        assert debug["included_count"] == 0


class TestPatchE_StructuredLogging:
    """Patch E: Single structured log per assembly."""

    def test_log_record_contains_required_fields(self, caplog):
        """Verify structured log includes all required fields."""
        import logging

        caplog.set_level(logging.INFO)

        strategy = TopicLocalStrategy()

        # Create a mock debug dict with all fields populated
        debug = {
            "mode": "topic_local",
            "topic_start_node_id": "test_topic_123",
            "budgets": {"T": 3000, "M": 2100, "A": 525, "R": 1575},
            "anchors": {
                "included_node_ids": ["anchor1", "anchor2"],
                "tokens_used": 200,
            },
            "recency": {
                "included_node_ids": ["recency1", "recency2"],
                "tokens_used": 500,
                "skipped_as_anchors": ["overlap1"],
            },
            "token_breakdown": {
                "overhead_tokens": 100,
                "summary_tokens": 200,
                "anchor_tokens": 200,
                "recency_tokens": 500,
                "import_tokens": 0,
                "total_tokens": 1000,
            },
            "truncation_info": {"summary_truncated": False},
            "timing": {"context_assembly_ms": 15.5},
        }

        with patch("episodic.config.config") as mock_config:
            mock_config.get.return_value = True  # debug=True

            strategy._emit_assembly_log(debug)

        # Find the assembly log
        assembly_logs = [r for r in caplog.records if "ASSEMBLY:" in r.message]
        assert len(assembly_logs) == 1

        log_msg = assembly_logs[0].message
        assert "ASSEMBLY:" in log_msg

        # Parse the JSON from the log
        json_str = log_msg.split("ASSEMBLY: ")[1]
        log_record = json.loads(json_str)

        # Verify required fields
        assert log_record["strategy"] == "topic_local"
        assert log_record["topic_id"] == "test_topic_123"
        assert "budgets" in log_record
        assert "anchors" in log_record
        assert log_record["anchors"]["exchange_ids"] == ["anchor1", "anchor2"]
        assert "recency" in log_record
        assert log_record["recency"]["exchange_ids"] == ["recency1", "recency2"]
        assert log_record["recency"]["skipped_as_anchors"] == ["overlap1"]
        assert "token_breakdown" in log_record
        assert "truncations" in log_record


class TestPatchA_ConfigurableAlpha:
    """Patch A: Configurable alpha for budget split."""

    def test_alpha_changes_allocation(self):
        """Verify changing alpha changes A and R allocation."""
        strategy = TopicLocalStrategy()

        # Test with alpha=0.25
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
            }.get(key, default)

            budgets_25 = strategy._compute_budgets(token_budget=3000)

        # Test with alpha=0.50
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.50,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
            }.get(key, default)

            budgets_50 = strategy._compute_budgets(token_budget=3000)

        # Alpha change should change A and R
        assert budgets_50["A"] > budgets_25["A"]
        assert budgets_50["R"] < budgets_25["R"]

        # T should remain constant
        assert budgets_25["T"] == budgets_50["T"] == 3000

    def test_alpha_clamp_respects_bounds(self):
        """Verify A is clamped between a_min and a_max."""
        strategy = TopicLocalStrategy()

        # Very small alpha (should hit a_min)
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.01,  # Very small
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
            }.get(key, default)

            budgets = strategy._compute_budgets(token_budget=3000)

        # Should be clamped to a_min
        assert budgets["A"] == 200

        # Very large alpha (should hit a_max)
        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 3000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.99,  # Very large
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
            }.get(key, default)

            budgets = strategy._compute_budgets(token_budget=3000)

        # Should be clamped to a_max
        assert budgets["A"] == 1000

    def test_current_behavior_matches_alpha_025(self):
        """Verify current behavior corresponds to alpha ~0.25."""
        strategy = TopicLocalStrategy()

        with patch("episodic.config.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "context_target_T": 4000,
                "context_overhead_estimate": 500,
                "summary_max_tokens": 400,
                "anchor_alpha": 0.25,
                "anchor_a_min": 200,
                "anchor_a_max": 1000,
            }.get(key, default)

            budgets = strategy._compute_budgets(token_budget=4000)

        # With T=4000, overhead=500, s_max=400 -> M=3100
        # A = clamp(200, 0.25*3100=775, 1000) = 775
        # Old behavior was token_budget // 4 = 1000 (close but not exact)
        # New behavior with alpha=0.25 gives 775
        assert budgets["alpha"] == 0.25
        assert budgets["M"] == 3100
        assert budgets["A"] == 775


class TestTokenEstimation:
    """Test token estimation helpers."""

    def test_estimate_tokens_basic(self):
        """Basic token estimation (chars / 4)."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("a" * 100) == 25

    def test_truncate_within_budget(self):
        """Text within budget should not be truncated."""
        text = "Short text"
        result, truncated = _truncate_to_tokens(text, 100)
        assert result == text
        assert truncated is False

    def test_truncate_over_budget(self):
        """Text over budget should be truncated with ellipsis."""
        text = "a" * 1000  # 250 tokens
        result, truncated = _truncate_to_tokens(text, 50)  # 50 tokens = 200 chars
        assert truncated is True
        assert len(result) <= 200
        assert result.endswith("...")
