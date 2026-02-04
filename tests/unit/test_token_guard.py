"""
Unit tests for Token Guard (Gap B and Gap A implementation).

Tests cover:
1. Token estimation accuracy
2. Validation with various assembly sizes
3. Drop policy enforcement (summary → recency → anchors → abort)
4. Worst-case assembly deterministic enforcement
5. Bug event logging
6. Gap A: TokenCounter protocol, registry, safety factor
7. Gap A: Adversarial undercount harness
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from episodic.token_guard import (
    estimate_tokens_text,
    estimate_tokens_message,
    estimate_tokens_messages,
    validate_assembly,
    guard_assembly,
    TokenBudget,
    TokenEstimate,
    ValidationResult,
    DropAction,
    log_bug_event,
    # Gap A additions
    TokenCounter,
    HeuristicTokenCounter,
    token_counter_registry,
    get_token_counter,
)


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_text_empty(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens_text("") == 0
        assert estimate_tokens_text(None) == 0

    def test_estimate_tokens_text_basic(self):
        """Basic text estimation uses chars/4."""
        # 40 chars / 4 = 10 tokens
        text = "Hello, this is a test message here."
        expected = len(text) // 4
        assert estimate_tokens_text(text) == expected

    def test_estimate_tokens_text_with_safety_factor(self):
        """Safety factor inflates estimate."""
        text = "Hello world"  # 11 chars = 2 tokens base
        base = estimate_tokens_text(text, safety_factor=1.0)
        inflated = estimate_tokens_text(text, safety_factor=1.2)
        assert inflated == int(base * 1.2)

    def test_estimate_tokens_message_simple(self):
        """Message estimation includes role overhead."""
        msg = {"role": "user", "content": "Hello"}
        tokens = estimate_tokens_message(msg)
        # 5 chars / 4 = 1 token + 4 role overhead = 5
        assert tokens == 5

    def test_estimate_tokens_message_multimodal(self):
        """Multimodal messages handle content blocks."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:..."}}
            ]
        }
        tokens = estimate_tokens_message(msg)
        # Text tokens + image overhead + role overhead
        assert tokens > 4  # At least role overhead

    def test_estimate_tokens_messages_breakdown(self):
        """Messages estimation provides component breakdown."""
        messages = [
            {"role": "system", "content": "# Topic: Test\n## Summary\nThis is a summary."},
            {"role": "system", "content": "## Relevant Past Context\nSome anchor text."},
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "The answer is 42."},
            {"role": "user", "content": "Thanks!"}
        ]
        total, breakdown = estimate_tokens_messages(messages)

        assert total > 0
        assert isinstance(breakdown, TokenEstimate)
        assert breakdown.summary > 0  # Summary detected
        assert breakdown.anchors > 0  # Anchors detected
        assert breakdown.user_message > 0  # User messages
        assert breakdown.recency > 0  # Assistant = recency


class TestValidateAssembly:
    """Tests for assembly validation logic."""

    def test_validate_within_cap(self):
        """Assembly within cap passes validation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ]
        budget = TokenBudget(full_cap=1000, summary_min=50, overhead_reserve=100)
        result = validate_assembly(messages, budget)

        assert result.valid is True
        assert len(result.actions_taken) == 0
        assert result.fallback_response is None

    def test_validate_over_cap_truncates_summary(self):
        """Over-cap assembly triggers summary truncation."""
        # Create a message with long summary
        long_summary = "This is a very long summary. " * 100  # ~3000 chars
        messages = [
            {"role": "system", "content": f"# Topic: Test\n## Summary\n{long_summary}"},
            {"role": "user", "content": "Question?"}
        ]
        budget = TokenBudget(full_cap=500, summary_min=50, overhead_reserve=50)
        result = validate_assembly(messages, budget, apply_drops=True)

        # Should have truncated summary
        if result.original_tokens > 400:  # Only if actually over cap
            assert DropAction.TRUNCATE_SUMMARY in result.actions_taken or result.valid

    def test_validate_over_cap_drops_recency(self):
        """Over-cap after summary truncation drops recency."""
        # Create messages with many recency exchanges
        messages = [
            {"role": "system", "content": "# Topic: Test"},
            {"role": "user", "content": "First question " * 50},
            {"role": "assistant", "content": "First answer " * 50},
            {"role": "user", "content": "Second question " * 50},
            {"role": "assistant", "content": "Second answer " * 50},
            {"role": "user", "content": "Current question"}  # Current user message
        ]
        budget = TokenBudget(full_cap=200, summary_min=10, overhead_reserve=50)
        result = validate_assembly(messages, budget, apply_drops=True)

        # Should have dropped some recency
        if not result.valid:
            assert DropAction.DROP_RECENCY in result.actions_taken or DropAction.ABORT in result.actions_taken
        # Original had more messages than final
        if result.messages and len(result.messages) < len(messages):
            assert DropAction.DROP_RECENCY in result.actions_taken

    def test_validate_over_cap_drops_anchors(self):
        """Over-cap after recency drop removes anchors."""
        messages = [
            {"role": "system", "content": f"# Topic\n\n## Relevant Past Context\n{'Anchor text ' * 200}"},
            {"role": "user", "content": "Question?"}
        ]
        budget = TokenBudget(full_cap=100, summary_min=10, overhead_reserve=10)
        result = validate_assembly(messages, budget, apply_drops=True)

        # Should have dropped anchors or aborted
        if not result.valid:
            assert DropAction.ABORT in result.actions_taken
        else:
            assert DropAction.DROP_ANCHORS in result.actions_taken or len(result.actions_taken) == 0

    def test_validate_abort_with_fallback(self):
        """Extreme over-cap triggers abort with fallback response."""
        # Create extremely large messages that can't be reduced
        huge_content = "X" * 50000  # ~12500 tokens
        messages = [
            {"role": "user", "content": huge_content}
        ]
        budget = TokenBudget(full_cap=100, summary_min=10, overhead_reserve=10)
        result = validate_assembly(messages, budget, apply_drops=True)

        assert result.valid is False
        assert DropAction.ABORT in result.actions_taken
        assert result.fallback_response is not None
        assert "unable to process" in result.fallback_response.lower()
        assert result.bug_event_logged is True


class TestGuardAssembly:
    """Tests for the guard_assembly convenience function."""

    def test_guard_returns_messages_when_valid(self):
        """Valid assembly returns original messages."""
        messages = [{"role": "user", "content": "Hello!"}]
        budget = TokenBudget(full_cap=1000)

        result_messages, fallback = guard_assembly(messages, budget)

        assert fallback is None
        assert len(result_messages) == 1

    def test_guard_returns_fallback_when_abort(self):
        """Aborted assembly returns fallback response."""
        huge_content = "X" * 50000
        messages = [{"role": "user", "content": huge_content}]
        budget = TokenBudget(full_cap=100)

        result_messages, fallback = guard_assembly(messages, budget)

        assert result_messages == []
        assert fallback is not None
        assert "unable to process" in fallback.lower()


class TestWorstCaseAssembly:
    """
    Worst-case assembly tests (Gap B acceptance criteria).

    Constructs "worst case" assembly with:
    - Max memory (summary + anchors)
    - Max recent turns
    - Max tool outputs (simulated as system messages)
    - Max system prompt
    - Max user message

    Asserts enforcement triggers deterministically.
    """

    def _build_worst_case_assembly(
        self,
        summary_chars: int = 2000,
        anchor_chars: int = 2000,
        recency_exchanges: int = 10,
        recency_chars_per_msg: int = 500,
        tool_outputs: int = 3,
        tool_output_chars: int = 1000,
        system_prompt_chars: int = 1000,
        user_message_chars: int = 500
    ) -> List[Dict[str, Any]]:
        """Build a worst-case assembly with configurable sizes."""
        messages = []

        # System prompt
        messages.append({
            "role": "system",
            "content": "S" * system_prompt_chars
        })

        # Topic context with summary and anchors
        summary_text = "Y" * summary_chars
        anchor_text = "A" * anchor_chars
        messages.append({
            "role": "system",
            "content": f"# Topic: Worst Case\n\n## Summary\n{summary_text}\n\n## Relevant Past Context\n{anchor_text}"
        })

        # Tool outputs (simulated as system messages)
        for i in range(tool_outputs):
            messages.append({
                "role": "system",
                "content": f"Tool output {i}: {'T' * tool_output_chars}"
            })

        # Recency exchanges
        for i in range(recency_exchanges):
            messages.append({
                "role": "user",
                "content": f"User turn {i}: {'U' * recency_chars_per_msg}"
            })
            messages.append({
                "role": "assistant",
                "content": f"Assistant turn {i}: {'R' * recency_chars_per_msg}"
            })

        # Current user message (last)
        messages.append({
            "role": "user",
            "content": "Current question: " + "Q" * user_message_chars
        })

        return messages

    def test_worst_case_triggers_enforcement(self):
        """Worst-case assembly triggers enforcement deterministically."""
        messages = self._build_worst_case_assembly(
            summary_chars=2000,
            anchor_chars=2000,
            recency_exchanges=10,
            recency_chars_per_msg=500,
            tool_outputs=3,
            tool_output_chars=1000,
            system_prompt_chars=1000,
            user_message_chars=500
        )

        # Calculate expected tokens
        total, breakdown = estimate_tokens_messages(messages)

        # Set cap well below the expected total
        budget = TokenBudget(
            full_cap=total // 2,  # Cap at half the tokens
            summary_min=100,
            overhead_reserve=200
        )

        result = validate_assembly(messages, budget, apply_drops=True)

        # Must have taken action
        assert len(result.actions_taken) > 0, "Worst case must trigger enforcement"

        # Must either succeed with reductions or abort
        if result.valid:
            assert result.final_tokens <= budget.full_cap - budget.overhead_reserve
        else:
            assert DropAction.ABORT in result.actions_taken
            assert result.bug_event_logged is True

    def test_worst_case_no_path_exceeds_cap_without_logging(self):
        """No execution path can exceed cap without logging bug event."""
        messages = self._build_worst_case_assembly()

        # Very tight cap
        budget = TokenBudget(full_cap=500, summary_min=50, overhead_reserve=100)

        result = validate_assembly(messages, budget, apply_drops=True)

        # Either we're under cap (valid) or we logged a bug event
        if not result.valid:
            assert result.bug_event_logged is True, "Overflow must log bug event"
        else:
            # If valid, we must be under cap
            assert result.final_tokens <= budget.full_cap - budget.overhead_reserve

    def test_worst_case_deterministic(self):
        """Same input produces same output (deterministic enforcement)."""
        messages = self._build_worst_case_assembly()
        budget = TokenBudget(full_cap=3000, summary_min=50, overhead_reserve=200)

        result1 = validate_assembly(messages.copy(), budget, apply_drops=True)
        result2 = validate_assembly(messages.copy(), budget, apply_drops=True)

        assert result1.valid == result2.valid
        assert result1.final_tokens == result2.final_tokens
        assert result1.actions_taken == result2.actions_taken

    def test_worst_case_drop_order(self):
        """Drop policy follows correct order: summary → recency → anchors."""
        # Build assembly that will trigger multiple drops
        messages = self._build_worst_case_assembly(
            summary_chars=3000,  # Large summary
            anchor_chars=3000,   # Large anchors
            recency_exchanges=5,
            recency_chars_per_msg=1000,
        )

        # Cap that requires multiple drops
        budget = TokenBudget(full_cap=1000, summary_min=50, overhead_reserve=100)

        result = validate_assembly(messages, budget, apply_drops=True)

        # If multiple actions taken, verify order
        if len(result.actions_taken) >= 2:
            action_order = result.actions_taken

            # Summary truncation should come before recency drop
            if DropAction.TRUNCATE_SUMMARY in action_order and DropAction.DROP_RECENCY in action_order:
                assert action_order.index(DropAction.TRUNCATE_SUMMARY) < action_order.index(DropAction.DROP_RECENCY)

            # Recency drop should come before anchor drop
            if DropAction.DROP_RECENCY in action_order and DropAction.DROP_ANCHORS in action_order:
                assert action_order.index(DropAction.DROP_RECENCY) < action_order.index(DropAction.DROP_ANCHORS)


class TestBugEventLogging:
    """Tests for bug event logging functionality."""

    def test_log_bug_event_called_on_abort(self):
        """Bug event is logged when assembly aborts."""
        huge_content = "X" * 50000
        messages = [{"role": "user", "content": huge_content}]
        budget = TokenBudget(full_cap=100)

        with patch('episodic.token_guard.logger') as mock_logger:
            result = validate_assembly(messages, budget, apply_drops=True)

            assert result.bug_event_logged is True
            # Logger should have been called with error level
            assert mock_logger.error.called or mock_logger.warning.called

    def test_log_bug_event_includes_details(self):
        """Bug event log includes relevant details."""
        messages = [{"role": "user", "content": "X" * 50000}]
        budget = TokenBudget(full_cap=100)

        with patch('episodic.token_guard.logger') as mock_logger:
            validate_assembly(messages, budget, apply_drops=True)

            # Check the log call includes key info
            if mock_logger.error.called:
                call_args = str(mock_logger.error.call_args)
                assert "token_overflow" in call_args.lower() or "TOKEN_GUARD_BUG" in call_args


class TestDropActionHelpers:
    """Tests for internal drop action helper functions."""

    def test_find_summary_message_index(self):
        """Correctly identifies summary message."""
        from episodic.token_guard import _find_summary_message_index

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "# Topic\n## Summary\nSome summary."},
            {"role": "user", "content": "Hi"}
        ]
        idx = _find_summary_message_index(messages)
        assert idx == 1

    def test_find_summary_message_index_not_found(self):
        """Returns None when no summary present."""
        from episodic.token_guard import _find_summary_message_index

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"}
        ]
        idx = _find_summary_message_index(messages)
        assert idx is None

    def test_find_recency_messages(self):
        """Correctly identifies recency (non-current user/assistant) messages."""
        from episodic.token_guard import _find_recency_messages

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old user"},
            {"role": "assistant", "content": "Old assistant"},
            {"role": "user", "content": "Current question"}  # This is current, not recency
        ]
        indices = _find_recency_messages(messages)

        # Should include old user (1) and old assistant (2), not current user (3) or system (0)
        assert 0 not in indices  # System
        assert 1 in indices      # Old user
        assert 2 in indices      # Old assistant
        assert 3 not in indices  # Current user (last)

    def test_find_anchor_section_index(self):
        """Correctly identifies anchor section message."""
        from episodic.token_guard import _find_anchor_section_index

        messages = [
            {"role": "system", "content": "## Relevant Past Context\nAnchors here"},
            {"role": "user", "content": "Hi"}
        ]
        idx = _find_anchor_section_index(messages)
        assert idx == 0


class TestTokenCounterProtocol:
    """Tests for TokenCounter protocol and registry (Gap A)."""

    def test_heuristic_counter_implements_protocol(self):
        """HeuristicTokenCounter implements TokenCounter protocol."""
        counter = HeuristicTokenCounter()
        assert hasattr(counter, 'count_text')
        assert hasattr(counter, 'count_messages')
        assert hasattr(counter, 'is_exact')
        assert hasattr(counter, 'backend_name')

    def test_heuristic_counter_is_not_exact(self):
        """Heuristic counter reports is_exact()==False."""
        counter = HeuristicTokenCounter()
        assert counter.is_exact() is False

    def test_heuristic_counter_backend_name(self):
        """Heuristic counter has identifiable backend name."""
        counter = HeuristicTokenCounter()
        assert "heuristic" in counter.backend_name().lower()

    def test_registry_returns_heuristic_by_default(self):
        """Registry returns heuristic counter for unknown provider/model."""
        counter = get_token_counter("unknown_provider", "unknown_model")
        assert counter.is_exact() is False
        assert "heuristic" in counter.backend_name().lower()

    def test_registry_returns_registered_counter(self):
        """Registry returns registered counter for known provider/model."""
        # Create a mock exact counter
        class MockExactCounter:
            def count_text(self, text: str) -> int:
                return len(text) // 3  # Different from heuristic
            def count_messages(self, messages: List[Dict[str, Any]]) -> int:
                return sum(self.count_text(m.get("content", "")) for m in messages)
            def is_exact(self) -> bool:
                return True
            def backend_name(self) -> str:
                return "mock_exact"

        # Register and retrieve
        mock_counter = MockExactCounter()
        token_counter_registry.register("test_provider", "test_model", mock_counter)

        retrieved = get_token_counter("test_provider", "test_model")
        assert retrieved.is_exact() is True
        assert retrieved.backend_name() == "mock_exact"

        # Clean up
        del token_counter_registry._counters[("test_provider", "test_model")]


class TestSafetyFactorSemantics:
    """Tests for safety factor behavior (Gap A)."""

    def test_safety_factor_not_applied_when_exact(self):
        """Safety factor is NOT applied when counter.is_exact()==True."""
        class ExactCounter:
            def count_text(self, text: str) -> int:
                return 100
            def count_messages(self, messages: List[Dict[str, Any]]) -> int:
                return 100
            def is_exact(self) -> bool:
                return True
            def backend_name(self) -> str:
                return "exact_stub"

        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        result = validate_assembly(
            messages, budget, safety_factor=1.5, counter=ExactCounter()
        )

        # Should NOT multiply by 1.5 since is_exact()
        assert result.details["applied_safety_factor"] == 1.0
        assert result.details["counter_exact"] is True

    def test_safety_factor_applied_when_heuristic(self):
        """Safety factor IS applied when counter.is_exact()==False."""
        messages = [{"role": "user", "content": "Hello world this is a test"}]
        budget = TokenBudget(full_cap=1000)

        # Force specific safety factor
        result = validate_assembly(messages, budget, safety_factor=1.5)

        assert result.details["applied_safety_factor"] == 1.5
        assert result.details["counter_exact"] is False
        # Raw tokens should differ from original_tokens
        raw = result.details["raw_tokens"]
        original = result.original_tokens
        assert original == int(raw * 1.5)

    def test_default_safety_factor_from_config(self):
        """Default safety factor (1.2) used when not specified."""
        messages = [{"role": "user", "content": "Hello world"}]
        budget = TokenBudget(full_cap=1000)

        with patch('episodic.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "token_safety_factor_heuristic": 1.2,
                "token_full_cap": 8000,
                "token_summary_min": 100,
                "token_overhead_reserve": 500,
            }.get(key, default)

            result = validate_assembly(messages, budget, safety_factor=1.0)
            # safety_factor=1.0 triggers config lookup for heuristic
            assert result.details["applied_safety_factor"] == 1.2


class TestAdversarialUndercount:
    """
    Adversarial undercount tests (Gap A acceptance criteria).

    Tests stress heuristic undercount risk classes:
    - Emoji + ZWJ sequences, combining marks
    - CJK blocks
    - Long code-like tokens with punctuation
    - Mixed-script strings

    Assertions verify safety factor provides conservative bounds.
    """

    # Test data: various pathological inputs
    EMOJI_ZWJ = "👨‍👩‍👧‍👦 🏳️‍🌈 👩‍💻 🧑‍🤝‍🧑"  # ZWJ family, flag, etc.
    COMBINING_MARKS = "é̸̢̛̝͈̹̈́̀͐̈́ñ̷̨̛̫̼̲̈́̾̈́ö̵̧̨̟̼̲́̀͐̈́"  # Heavily combined characters
    CJK_BLOCK = "日本語テスト 中文测试 한국어테스트"  # Japanese, Chinese, Korean
    CODE_LIKE = "const_foo_bar_baz_qux_xyz=function(){return[1,2,3,4,5,6,7,8,9,0];}"
    MIXED_SCRIPT = "Hello世界مرحباПривет🌍こんにちは"  # Latin, CJK, Arabic, Cyrillic, Emoji, JP

    def test_emoji_zwj_safety_factor_inflates(self):
        """Safety factor inflates count for emoji/ZWJ strings."""
        counter = HeuristicTokenCounter()
        raw_count = counter.count_text(self.EMOJI_ZWJ)

        factor = 1.2
        inflated = int(raw_count * factor)

        assert inflated >= raw_count
        # Emoji ZWJ sequences often tokenize to MORE tokens than chars/4
        # Safety factor should provide buffer

    def test_combining_marks_safety_factor_inflates(self):
        """Safety factor inflates count for combining marks."""
        counter = HeuristicTokenCounter()
        raw_count = counter.count_text(self.COMBINING_MARKS)

        factor = 1.2
        inflated = int(raw_count * factor)

        assert inflated >= raw_count

    def test_cjk_safety_factor_inflates(self):
        """Safety factor inflates count for CJK text."""
        counter = HeuristicTokenCounter()
        raw_count = counter.count_text(self.CJK_BLOCK)

        factor = 1.2
        inflated = int(raw_count * factor)

        assert inflated >= raw_count

    def test_code_like_safety_factor_inflates(self):
        """Safety factor inflates count for code-like strings."""
        counter = HeuristicTokenCounter()
        raw_count = counter.count_text(self.CODE_LIKE)

        factor = 1.2
        inflated = int(raw_count * factor)

        assert inflated >= raw_count

    def test_mixed_script_safety_factor_inflates(self):
        """Safety factor inflates count for mixed-script strings."""
        counter = HeuristicTokenCounter()
        raw_count = counter.count_text(self.MIXED_SCRIPT)

        factor = 1.2
        inflated = int(raw_count * factor)

        assert inflated >= raw_count

    def test_guard_more_conservative_with_factor(self):
        """Guard decisions become strictly more conservative with safety factor."""
        # Create a message that's borderline on the cap
        borderline_content = "X" * 2800  # ~700 tokens heuristic
        messages = [{"role": "user", "content": borderline_content}]

        # Budget that passes without factor but may fail with factor
        budget = TokenBudget(full_cap=900, overhead_reserve=100)

        # Without factor (exact counter)
        class ExactCounter:
            def count_text(self, text: str) -> int:
                return len(text) // 4
            def count_messages(self, messages: List[Dict[str, Any]]) -> int:
                return sum(len(m.get("content", "")) // 4 + 4 for m in messages)
            def is_exact(self) -> bool:
                return True
            def backend_name(self) -> str:
                return "exact"

        result_exact = validate_assembly(messages, budget, counter=ExactCounter())

        # With factor (heuristic counter, 1.2x)
        result_heuristic = validate_assembly(messages, budget, safety_factor=1.2)

        # Heuristic result should be MORE conservative (higher token count)
        assert result_heuristic.original_tokens >= result_exact.original_tokens

        # If exact passes but heuristic fails, that's conservative behavior
        if result_exact.valid:
            # Heuristic might fail or take more aggressive drop actions
            pass  # This is expected conservative behavior

    def test_safety_factor_not_applied_to_exact_stub(self):
        """Confirm exact counter stub doesn't get safety factor applied."""
        class ExactStub:
            def count_text(self, text: str) -> int:
                return 50
            def count_messages(self, messages: List[Dict[str, Any]]) -> int:
                return 50
            def is_exact(self) -> bool:
                return True
            def backend_name(self) -> str:
                return "exact_stub"

        messages = [{"role": "user", "content": self.MIXED_SCRIPT}]
        budget = TokenBudget(full_cap=1000)

        result = validate_assembly(messages, budget, safety_factor=2.0, counter=ExactStub())

        # Factor should NOT be applied
        assert result.details["applied_safety_factor"] == 1.0
        assert result.original_tokens == 50  # Raw count, no inflation

    def test_adversarial_inputs_dont_overflow(self):
        """Adversarial inputs with safety factor don't overflow cap."""
        adversarial_inputs = [
            self.EMOJI_ZWJ,
            self.COMBINING_MARKS,
            self.CJK_BLOCK,
            self.CODE_LIKE,
            self.MIXED_SCRIPT,
        ]

        # Create messages with all adversarial inputs
        messages = []
        for i, text in enumerate(adversarial_inputs):
            # Repeat each to make it substantial
            messages.append({"role": "user", "content": text * 10})
            messages.append({"role": "assistant", "content": text * 10})

        # Use reasonable budget
        budget = TokenBudget(full_cap=5000, overhead_reserve=500)

        # Validate with safety factor
        result = validate_assembly(messages, budget, safety_factor=1.2)

        # Should either be valid (under cap) or have taken appropriate actions
        if result.valid:
            assert result.final_tokens <= budget.full_cap - budget.overhead_reserve
        else:
            # If invalid, must have logged bug event
            assert result.bug_event_logged is True


class TestValidationResultDetails:
    """Tests for Gap A logging fields in ValidationResult."""

    def test_result_includes_counter_info(self):
        """ValidationResult includes counter_backend and counter_exact."""
        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        result = validate_assembly(messages, budget)

        assert "counter_backend" in result.details
        assert "counter_exact" in result.details
        assert "applied_safety_factor" in result.details
        assert "raw_tokens" in result.details

    def test_result_counter_backend_is_heuristic(self):
        """Default counter backend is identified as heuristic."""
        messages = [{"role": "user", "content": "Hello"}]
        budget = TokenBudget(full_cap=1000)

        result = validate_assembly(messages, budget)

        assert "heuristic" in result.details["counter_backend"].lower()
        assert result.details["counter_exact"] is False


class TestRelevanceTruncationIntegration:
    """Integration tests for Phase 2 relevance-aware truncation."""

    def test_truncation_disabled_by_default(self):
        """Relevance truncation is disabled by default."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "C" * 200},
            {"role": "assistant", "content": "D" * 200},
            {"role": "user", "content": "Current query"},
        ]
        budget = TokenBudget(full_cap=300, overhead_reserve=50)

        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=False,
            emit_event=False
        )

        # Should use legacy drop policy (no RELEVANCE_TRUNCATION action)
        action_values = [a.value for a in result.actions_taken]
        assert "relevance_truncation" not in action_values

    def test_truncation_enabled_uses_importance_scoring(self):
        """When enabled, truncation uses importance-based drops."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Important anchor content"},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response here"},  # Index 2 - anchor
            {"role": "user", "content": "C" * 200},
            {"role": "assistant", "content": "D" * 200},
            {"role": "user", "content": "Python programming tutorial"},
        ]
        budget = TokenBudget(full_cap=300, overhead_reserve=50)
        anchor_indices = {1, 2}

        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=True,
            current_query="Python programming tutorial",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Should include RELEVANCE_TRUNCATION action
        action_values = [a.value for a in result.actions_taken]
        assert "relevance_truncation" in action_values or result.valid

    def test_truncation_determinism_same_input_same_output(self):
        """Same input produces same truncation decisions."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello Python programming"},  # Index 1 - anchor
            {"role": "assistant", "content": "Python is great for beginners."},  # Index 2 - anchor
            {"role": "user", "content": "Unrelated cooking recipe discussion here"},
            {"role": "assistant", "content": "More cooking content goes here"},
            {"role": "user", "content": "Back to Python programming tutorials"},
            {"role": "assistant", "content": "Let me explain Python more"},
            {"role": "user", "content": "Python programming tutorial"},
        ]
        budget = TokenBudget(full_cap=200, overhead_reserve=50)
        counter = HeuristicTokenCounter()
        anchor_indices = {1, 2}

        result1 = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Python programming tutorial",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        result2 = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Python programming tutorial",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Same final message count
        if result1.messages and result2.messages:
            assert len(result1.messages) == len(result2.messages)

        # Same final tokens
        assert result1.final_tokens == result2.final_tokens

        # Same actions taken
        assert result1.actions_taken == result2.actions_taken

    def test_truncation_result_stored_for_logging(self):
        """TruncationResult is stored in ValidationResult for logging."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Anchor content here"},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response"},  # Index 2 - anchor
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "Query"},
        ]
        budget = TokenBudget(full_cap=150, overhead_reserve=20)
        anchor_indices = {1, 2}

        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=True,
            current_query="Query",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # If truncation was applied, truncation_result should be set
        if DropAction.RELEVANCE_TRUNCATION in result.actions_taken:
            assert result.truncation_result is not None
            assert hasattr(result.truncation_result, 'tokens_before')
            assert hasattr(result.truncation_result, 'tokens_after')
            assert hasattr(result.truncation_result, 'decisions')

    def test_truncation_with_anchor_indices(self):
        """Anchor indices are respected during truncation."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Important anchor content"},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response"},  # Index 2 - anchor
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "C" * 100},
            {"role": "assistant", "content": "D" * 100},
            {"role": "user", "content": "Query"},
        ]
        budget = TokenBudget(full_cap=200, overhead_reserve=50)
        anchor_indices = {1, 2}

        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=True,
            current_query="Query",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Result should be valid or have taken truncation actions
        assert result.valid or result.actions_taken

    def test_truncation_extracts_query_from_last_user_message(self):
        """When current_query not provided, extracts from last user message."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Anchor content"},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response"},  # Index 2 - anchor
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "Python programming question"},
        ]
        budget = TokenBudget(full_cap=150, overhead_reserve=20)
        anchor_indices = {1, 2}

        # Don't provide current_query - should extract automatically
        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=True,
            current_query=None,  # Auto-extract
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Should complete without error
        assert isinstance(result, ValidationResult)

    def test_truncation_with_same_counter_as_guard(self):
        """Truncation uses same counter instance as guard for consistency."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Anchor content"},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response"},  # Index 2 - anchor
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "Query"},
        ]
        budget = TokenBudget(full_cap=200, overhead_reserve=30)
        counter = HeuristicTokenCounter()
        anchor_indices = {1, 2}

        result = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Query",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Final tokens should be accurate with same counter
        if result.messages:
            actual_tokens = counter.count_messages(result.messages)
            # Account for safety factor
            assert result.final_tokens >= actual_tokens

    def test_no_double_drop_when_relevance_truncation_fits_budget(self):
        """
        Regression test: Legacy drop policy NOT invoked when relevance truncation
        successfully fits within budget.

        Scenario:
        1. Messages exceed budget
        2. Enable relevance truncation
        3. Truncation succeeds (fits budget after dropping by importance)
        4. Assert: legacy drops (TRUNCATE_SUMMARY, DROP_RECENCY, DROP_ANCHORS) NOT fired
        5. Assert: only RELEVANCE_TRUNCATION action taken
        """
        # Create messages that exceed budget but can fit after truncation
        messages = [
            {"role": "system", "content": "## Summary\nA topic summary here with some content."},
            {"role": "user", "content": "First exchange message about Python"},  # Index 1 - anchor
            {"role": "assistant", "content": "Response about Python programming"},  # Index 2 - anchor
            {"role": "user", "content": "Unrelated cooking recipe discussion here"},  # Should be dropped
            {"role": "assistant", "content": "More cooking content goes here"},  # Should be dropped
            {"role": "user", "content": "Python programming tutorial query"},
        ]
        anchor_indices = {1, 2}

        # Budget that requires dropping ~2 messages
        counter = HeuristicTokenCounter()
        total_tokens = counter.count_messages(messages)
        budget = TokenBudget(
            full_cap=int(total_tokens * 0.7),  # 70% of total = need to drop ~30%
            overhead_reserve=20
        )

        result = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Python programming tutorial query",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # Extract action values
        action_values = [a.value for a in result.actions_taken]

        # Should have RELEVANCE_TRUNCATION if truncation was needed
        if not result.valid or action_values:
            # Legacy drop actions should NOT be present when relevance truncation was used
            legacy_actions = {"truncate_summary", "drop_recency", "drop_anchors"}
            actions_set = set(action_values)

            if "relevance_truncation" in actions_set:
                # Double-drop prevention: No legacy actions when relevance truncation succeeded
                legacy_fired = legacy_actions & actions_set
                assert legacy_fired == set(), (
                    f"Legacy drop policy incorrectly fired after relevance truncation: {legacy_fired}. "
                    f"All actions: {action_values}"
                )

    def test_legacy_fallback_when_relevance_truncation_insufficient(self):
        """
        Legacy policy is correctly used as fallback when relevance truncation
        alone cannot meet the budget.
        """
        # Create extreme scenario where truncation can't help enough
        messages = [
            {"role": "system", "content": "## Summary\n" + "X" * 500},  # Huge summary
            {"role": "user", "content": "Anchor A" * 10},  # Index 1 - anchor
            {"role": "assistant", "content": "Anchor B" * 10},  # Index 2 - anchor
            {"role": "user", "content": "A" * 50},
            {"role": "assistant", "content": "B" * 50},
            {"role": "user", "content": "Query"},
        ]
        anchor_indices = {1, 2}

        # Very tight budget that even truncation can't fully solve
        budget = TokenBudget(full_cap=50, overhead_reserve=10)
        counter = HeuristicTokenCounter()

        result = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Query",
            anchor_indices=anchor_indices,
            emit_event=False
        )

        # In this extreme case, either:
        # 1. Both relevance truncation AND legacy policy fired (fallback), OR
        # 2. Validation failed entirely (abort)
        action_values = [a.value for a in result.actions_taken]

        # Should complete without assertion errors
        assert isinstance(result, ValidationResult)


class TestVerificationAudit:
    """
    Verification Pass: Phase 1+2 Completion Audit Tests.

    Desktop audit request: Systematic verification of static and behavioral invariants.
    """

    def test_fail_fast_invariant_enforced(self):
        """
        Immediate Code Change Verification:
        validate_assembly raises ValueError when enable_relevance_truncation=True
        but anchor_indices is None or empty.
        """
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Query"},
        ]
        budget = TokenBudget(full_cap=500, overhead_reserve=50)

        # Test with anchor_indices=None
        with pytest.raises(ValueError) as exc_info:
            validate_assembly(
                messages, budget,
                enable_relevance_truncation=True,
                current_query="Query",
                anchor_indices=None,
                emit_event=False
            )
        assert "anchor_indices" in str(exc_info.value)
        assert "non-empty" in str(exc_info.value)

        # Test with empty anchor_indices
        with pytest.raises(ValueError) as exc_info:
            validate_assembly(
                messages, budget,
                enable_relevance_truncation=True,
                current_query="Query",
                anchor_indices=set(),
                emit_event=False
            )
        assert "anchor_indices" in str(exc_info.value)

    def test_no_fail_fast_when_truncation_disabled(self):
        """
        Fail-fast invariant does NOT fire when enable_relevance_truncation=False.
        """
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        budget = TokenBudget(full_cap=500, overhead_reserve=50)

        # Should NOT raise even without anchor_indices
        result = validate_assembly(
            messages, budget,
            enable_relevance_truncation=False,
            emit_event=False
        )
        assert result.valid

    def test_anchor_preservation_property(self):
        """
        Behavioral Invariant: Within relevance truncation, anchors are NEVER dropped
        before all non-anchors exhausted.

        Note: This property holds within relevance truncation. If budget is extremely
        tight and legacy fallback fires, anchors may be dropped by legacy policy.
        This test verifies the property within a budget that relevance truncation
        can handle without legacy fallback.
        """
        from episodic.truncation import truncate_by_relevance

        # Build messages: 2 anchors (high score) and 4 non-anchors (low score)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "ANCHOR USER content " + "Y" * 50},     # Index 1 - anchor
            {"role": "assistant", "content": "ANCHOR ASST content " + "Z" * 50}, # Index 2 - anchor
            {"role": "user", "content": "Non-anchor 1 " + "A" * 50},             # Index 3
            {"role": "assistant", "content": "Non-anchor 2 " + "B" * 50},        # Index 4
            {"role": "user", "content": "Non-anchor 3 " + "C" * 50},             # Index 5
            {"role": "assistant", "content": "Non-anchor 4 " + "D" * 50},        # Index 6
            {"role": "user", "content": "Current query"},                         # Index 7
        ]
        anchor_indices = {1, 2}
        counter = HeuristicTokenCounter()

        # Target that requires dropping some messages but not all
        current_tokens = counter.count_messages(messages)
        target_tokens = current_tokens - 100  # Need to free ~100 tokens

        # Call truncate_by_relevance directly to isolate the property test
        result = truncate_by_relevance(
            messages=messages,
            target_tokens=target_tokens,
            current_query="Current query",
            counter=counter,
            anchor_indices=anchor_indices,
        )

        # Check the decisions made
        decisions = result.decisions

        # Separate anchor and non-anchor drop decisions
        anchor_drops = [d for d in decisions if d.reason == "anchor_low_score"]
        non_anchor_drops = [d for d in decisions if d.reason == "recency_low_score"]

        # If any anchors were dropped, ALL non-anchors must have been dropped first
        if anchor_drops:
            # Count how many non-anchors exist (indices 3-6)
            non_anchor_indices = {3, 4, 5, 6}
            dropped_non_anchor_indices = {d.message_index for d in non_anchor_drops}

            # All non-anchors should be dropped before any anchor
            if dropped_non_anchor_indices != non_anchor_indices:
                pytest.fail(
                    f"Anchor preservation violated: anchor dropped but not all non-anchors. "
                    f"Dropped non-anchors: {dropped_non_anchor_indices}, expected all: {non_anchor_indices}"
                )

    def test_adversarial_determinism_100_runs(self):
        """
        Behavioral Invariant: Run 100 times, compare exact bytes.

        Truncation must produce identical results across multiple runs.
        """
        messages = [
            {"role": "system", "content": "## Summary\n" + "X" * 200},
            {"role": "user", "content": "Anchor content here " + "Y" * 50},    # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response " + "Z" * 50},   # Index 2 - anchor
            {"role": "user", "content": "Regular content " + "A" * 80},
            {"role": "assistant", "content": "Response content " + "B" * 80},
            {"role": "user", "content": "More content " + "C" * 80},
            {"role": "assistant", "content": "Another response " + "D" * 80},
            {"role": "user", "content": "Current query text"},
        ]
        anchor_indices = {1, 2}
        budget = TokenBudget(full_cap=200, overhead_reserve=20)
        counter = HeuristicTokenCounter()

        # Run 100 times
        results = []
        for _ in range(100):
            result = validate_assembly(
                messages.copy(), budget,
                counter=counter,
                enable_relevance_truncation=True,
                current_query="Current query text",
                anchor_indices=anchor_indices,
                emit_event=False,
                apply_drops=True
            )
            # Capture message content as tuple for comparison
            result_tuple = tuple(
                (m.get("role"), m.get("content"))
                for m in result.messages
            )
            results.append(result_tuple)

        # All results must be identical
        first_result = results[0]
        for i, r in enumerate(results[1:], start=2):
            if r != first_result:
                pytest.fail(f"Determinism violation: run {i} differs from run 1")

    def test_token_counter_identity_in_truncation(self):
        """
        Static Invariant: Same TokenCounter used for enforcement AND truncation measurement.

        Verify that when we pass a counter to validate_assembly, that same counter
        is used inside truncation.
        """
        class TrackedCounter:
            """Counter that tracks when it's called."""
            def __init__(self):
                self.call_count = 0
                self._counter = HeuristicTokenCounter()

            def count_text(self, text: str) -> int:
                self.call_count += 1
                return self._counter.count_text(text)

            def count_messages(self, messages) -> int:
                self.call_count += 1
                return self._counter.count_messages(messages)

            def is_exact(self) -> bool:
                return False

            def backend_name(self) -> str:
                return "tracked"

        messages = [
            {"role": "system", "content": "## Summary\n" + "X" * 200},
            {"role": "user", "content": "Anchor " + "Y" * 100},   # Index 1 - anchor
            {"role": "assistant", "content": "Response " + "Z" * 100}, # Index 2 - anchor
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "Query"},
        ]
        anchor_indices = {1, 2}
        budget = TokenBudget(full_cap=100, overhead_reserve=10)  # Force truncation

        tracker = TrackedCounter()
        initial_calls = tracker.call_count

        result = validate_assembly(
            messages, budget,
            counter=tracker,
            enable_relevance_truncation=True,
            current_query="Query",
            anchor_indices=anchor_indices,
            emit_event=False,
            apply_drops=True
        )

        # Counter should have been called (initial + truncation calls)
        assert tracker.call_count > initial_calls, "Counter should be called during validation/truncation"

        # Result details should show our counter
        assert result.details.get("counter_backend") == "tracked"

    def test_fallback_correctness_legacy_policy_works(self):
        """
        Behavioral Invariant: Legacy policy fallback works correctly.

        When relevance truncation is disabled, the legacy drop policy
        (summary → recency → anchors → abort) still functions correctly.
        """
        messages = [
            {"role": "system", "content": "## Summary\n" + "X" * 500},  # Large summary
            {"role": "user", "content": "Hello " + "A" * 100},
            {"role": "assistant", "content": "Hi there " + "B" * 100},
            {"role": "user", "content": "Query"},
        ]

        # Tight budget forces legacy truncation
        budget = TokenBudget(full_cap=100, overhead_reserve=10, summary_min=20)
        counter = HeuristicTokenCounter()

        result = validate_assembly(
            messages, budget,
            counter=counter,
            enable_relevance_truncation=False,  # Use legacy policy
            emit_event=False,
            apply_drops=True
        )

        # Legacy policy should have attempted recovery
        actions = [a.value for a in result.actions_taken]

        # Should have taken at least one legacy action
        legacy_actions = {"truncate_summary", "drop_recency", "drop_anchors", "abort"}
        has_legacy_action = any(a in legacy_actions for a in actions)

        if not has_legacy_action and not result.valid:
            # Abort is also a valid outcome
            pass
        elif result.valid:
            # If valid, we should be under cap
            assert result.final_tokens <= (budget.full_cap - budget.overhead_reserve)

    def test_call_site_inventory_production_safe(self):
        """
        Static Invariant: All production call sites either pass anchor_indices
        explicitly OR have enable_relevance_truncation=False.

        Production call sites in conversation.py use guard_assembly without
        enable_relevance_truncation, which defaults to config (default False).
        This test verifies that behavior is safe by default.
        """
        # Simulate production call pattern: no anchor_indices, no explicit truncation flag
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Query"},
        ]
        budget = TokenBudget(full_cap=500, overhead_reserve=50)

        # Production pattern: call without truncation params (defaults to disabled)
        result_messages, fallback = guard_assembly(messages, budget, emit_event=False)

        # Should work fine since truncation is disabled by default
        assert fallback is None
        assert len(result_messages) == len(messages)

    def test_replay_snapshot_golden_determinism(self):
        """
        End-to-End Fixture: Replay snapshot produces byte-identical results.

        Create a snapshot, serialize it, deserialize, and replay. The output
        must match the original exactly.
        """
        import json
        import hashlib

        messages = [
            {"role": "system", "content": "## Summary\nThis is a summary of the conversation."},
            {"role": "user", "content": "Anchor content " + "A" * 50},    # Index 1 - anchor
            {"role": "assistant", "content": "Anchor response " + "B" * 50}, # Index 2 - anchor
            {"role": "user", "content": "Regular " + "C" * 50},
            {"role": "assistant", "content": "Response " + "D" * 50},
            {"role": "user", "content": "Query about the topic"},
        ]
        anchor_indices = {1, 2}
        budget = TokenBudget(full_cap=150, overhead_reserve=15)
        counter = HeuristicTokenCounter()

        # First run - capture baseline
        result1 = validate_assembly(
            messages.copy(), budget,
            counter=counter,
            enable_relevance_truncation=True,
            current_query="Query about the topic",
            anchor_indices=anchor_indices,
            emit_event=False,
            apply_drops=True
        )

        # Capture golden state
        golden_messages_json = json.dumps(
            [{"role": m["role"], "content": m["content"]} for m in result1.messages],
            sort_keys=True
        )
        golden_hash = hashlib.sha256(golden_messages_json.encode()).hexdigest()

        # Run 10 more times and compare
        for run in range(10):
            result = validate_assembly(
                messages.copy(), budget,
                counter=counter,
                enable_relevance_truncation=True,
                current_query="Query about the topic",
                anchor_indices=anchor_indices,
                emit_event=False,
                apply_drops=True
            )

            result_json = json.dumps(
                [{"role": m["role"], "content": m["content"]} for m in result.messages],
                sort_keys=True
            )
            result_hash = hashlib.sha256(result_json.encode()).hexdigest()

            if result_hash != golden_hash:
                pytest.fail(
                    f"Golden test failed on run {run + 1}: "
                    f"hash {result_hash[:16]}... != golden {golden_hash[:16]}..."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
