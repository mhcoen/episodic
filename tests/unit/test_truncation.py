"""
Unit tests for Relevance-Aware Truncation (Phase 2).

Tests cover:
1. Score computation (weights, components)
2. Anchor priority (anchors retained before recency)
3. Early exchange protection
4. Reference detection (quote overlap, markers + shared tokens)
5. Determinism (same input = same output)
6. Replay compatibility (logging, decisions)
7. Drop order (ascending score, ties by older-first)
"""

import pytest
from typing import Dict, Any, List, Set

from episodic.truncation import (
    WEIGHT_ANCHOR,
    WEIGHT_EARLY,
    WEIGHT_LEX_SIM,
    WEIGHT_REFERENCED,
    MessageScore,
    TruncationDecision,
    TruncationResult,
    detect_reference,
    compute_lexical_similarity,
    score_message,
    score_messages,
    drop_by_importance,
    truncate_by_relevance,
    _find_longest_common_substring,
    _extract_key_words,
)
from episodic.token_guard import HeuristicTokenCounter


@pytest.fixture
def counter() -> HeuristicTokenCounter:
    """Create a token counter."""
    return HeuristicTokenCounter()


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Create sample conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, I want to learn about Python programming."},
        {"role": "assistant", "content": "Python is a great programming language for beginners."},
        {"role": "user", "content": "What about data science?"},
        {"role": "assistant", "content": "Python is widely used in data science with libraries like pandas and numpy."},
        {"role": "user", "content": "Can you explain machine learning?"},
        {"role": "assistant", "content": "Machine learning uses algorithms to learn from data and make predictions."},
        {"role": "user", "content": "Tell me more about neural networks."},  # Current query
    ]


class TestScoreComputation:
    """Tests for importance score computation."""

    def test_score_weights_defined(self):
        """Score weights are defined correctly."""
        assert WEIGHT_ANCHOR == 100
        assert WEIGHT_EARLY == 3
        assert WEIGHT_LEX_SIM == 2
        assert WEIGHT_REFERENCED == 5

    def test_anchor_message_high_score(self):
        """Anchor messages get high score (100 points)."""
        msg = {"role": "user", "content": "Some anchored content"}
        anchor_indices = {0}

        score = score_message(
            message=msg,
            index=0,
            current_query="test query",
            anchor_indices=anchor_indices,
            total_recency_count=10,
            referenced_indices=set(),
        )

        assert score.is_anchor
        assert score.score >= WEIGHT_ANCHOR

    def test_early_message_bonus(self):
        """Early messages (first 2 exchanges) get bonus."""
        msg = {"role": "user", "content": "First message content"}

        score = score_message(
            message=msg,
            index=0,  # First message
            current_query="unrelated query",
            anchor_indices=set(),
            total_recency_count=10,
            referenced_indices=set(),
        )

        assert score.is_early
        assert score.score >= WEIGHT_EARLY

    def test_late_message_no_early_bonus(self):
        """Late messages don't get early bonus."""
        msg = {"role": "user", "content": "Late message content"}

        score = score_message(
            message=msg,
            index=10,  # Late message
            current_query="unrelated query",
            anchor_indices=set(),
            total_recency_count=20,
            referenced_indices=set(),
        )

        assert not score.is_early

    def test_lexical_similarity_affects_score(self):
        """Lexical similarity contributes to score."""
        msg = {"role": "user", "content": "Python programming is great"}

        score = score_message(
            message=msg,
            index=5,
            current_query="Python programming tutorial",
            anchor_indices=set(),
            total_recency_count=10,
            referenced_indices=set(),
        )

        assert score.lex_similarity > 0
        assert score.score > 0

    def test_referenced_message_bonus(self):
        """Referenced messages get bonus."""
        msg = {"role": "user", "content": "Some content"}

        score = score_message(
            message=msg,
            index=0,
            current_query="test",
            anchor_indices=set(),
            total_recency_count=10,
            referenced_indices={0},  # This message is referenced
        )

        assert score.is_referenced
        assert score.score >= WEIGHT_REFERENCED

    def test_score_components_additive(self):
        """Score is sum of weighted components."""
        msg = {"role": "user", "content": "Python data science tutorial"}

        score = score_message(
            message=msg,
            index=0,  # Early
            current_query="Python data science",  # High similarity
            anchor_indices={0},  # Is anchor
            total_recency_count=10,
            referenced_indices={0},  # Is referenced
        )

        # Should have all bonuses
        assert score.is_anchor
        assert score.is_early
        assert score.is_referenced
        assert score.lex_similarity > 0

        # Score should be sum of components
        expected_min = WEIGHT_ANCHOR + WEIGHT_EARLY + WEIGHT_REFERENCED
        assert score.score >= expected_min


class TestReferenceDetection:
    """Tests for reference detection."""

    def test_quote_overlap_detected(self):
        """Detects quote overlap >= 40 chars."""
        current = "You said 'Python is a great programming language for beginners' earlier."
        message = "Python is a great programming language for beginners."

        assert detect_reference(current, message, min_quote_overlap=40)

    def test_short_overlap_not_detected(self):
        """Short overlaps not detected as references."""
        current = "You mentioned Python."
        message = "Python is great."

        assert not detect_reference(current, message, min_quote_overlap=40)

    def test_explicit_marker_with_shared_words(self):
        """Detects explicit markers with shared tokens."""
        # Need at least 6 shared key words (non-stop words)
        current = "You said something about Python programming, data science, machine learning, and neural networks."
        message = "Python programming for data science includes machine learning and neural networks."

        assert detect_reference(current, message, min_shared_tokens=6)

    def test_marker_without_enough_shared_words(self):
        """Markers without enough shared words don't trigger."""
        current = "You said something about cooking."
        message = "Python programming is great."

        assert not detect_reference(current, message, min_shared_tokens=6)

    def test_empty_strings_not_reference(self):
        """Empty strings are not references."""
        assert not detect_reference("", "content")
        assert not detect_reference("query", "")
        assert not detect_reference("", "")


class TestLexicalSimilarity:
    """Tests for lexical similarity computation."""

    def test_identical_strings_high_similarity(self):
        """Identical strings have high similarity."""
        sim = compute_lexical_similarity("Python programming", "Python programming")
        assert sim == 1.0

    def test_different_strings_low_similarity(self):
        """Completely different strings have low similarity."""
        sim = compute_lexical_similarity("Python programming", "cooking recipes")
        assert sim < 0.2

    def test_partial_overlap_medium_similarity(self):
        """Partial overlap gives medium similarity."""
        sim = compute_lexical_similarity(
            "Python programming tutorial",
            "Python data science tutorial"
        )
        assert 0.2 < sim < 0.8

    def test_empty_strings_zero_similarity(self):
        """Empty strings have zero similarity."""
        assert compute_lexical_similarity("", "content") == 0.0
        assert compute_lexical_similarity("query", "") == 0.0


class TestAnchorPriority:
    """Tests for anchor priority in truncation."""

    def test_anchors_dropped_after_recency(self, counter):
        """Anchors are dropped only after recency is exhausted."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Anchor content here"},  # Anchor
            {"role": "assistant", "content": "Anchor response"},  # Anchor
            {"role": "user", "content": "Recent message 1"},
            {"role": "assistant", "content": "Recent response 1"},
            {"role": "user", "content": "Recent message 2"},
            {"role": "assistant", "content": "Recent response 2"},
            {"role": "user", "content": "Current query"},
        ]

        anchor_indices = {1, 2}

        result = drop_by_importance(
            messages=messages,
            target_reduction=50,  # Small reduction
            current_query="Current query",
            counter=counter,
            anchor_indices=anchor_indices,
        )

        # Check that recency was dropped before anchors
        dropped_indices = {d.message_index for d in result.decisions}

        # If any anchors were dropped, all recency should have been dropped first
        anchor_dropped = dropped_indices & anchor_indices
        recency_indices = {3, 4, 5, 6}  # Non-anchor recency

        if anchor_dropped:
            # All recency should be dropped
            assert recency_indices <= dropped_indices

    def test_anchors_retained_when_recency_sufficient(self, counter):
        """Anchors are retained if recency reduction is sufficient."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Important anchor content"},  # Anchor
            {"role": "user", "content": "A" * 200},  # Large recency message
            {"role": "assistant", "content": "B" * 200},  # Large recency message
            {"role": "user", "content": "Current query"},
        ]

        anchor_indices = {1}

        result = drop_by_importance(
            messages=messages,
            target_reduction=50,  # Small reduction
            current_query="Current query",
            counter=counter,
            anchor_indices=anchor_indices,
        )

        # Anchor should not be dropped
        dropped_indices = {d.message_index for d in result.decisions}
        assert 1 not in dropped_indices


class TestEarlyProtection:
    """Tests for early exchange protection."""

    def test_early_messages_higher_score(self, counter):
        """Early messages have higher score than late messages."""
        messages = [
            {"role": "user", "content": "First message content"},  # Early
            {"role": "assistant", "content": "First response"},  # Early
            {"role": "user", "content": "Second message"},  # Early
            {"role": "assistant", "content": "Second response"},  # Early
            {"role": "user", "content": "Later message same content"},  # Late
            {"role": "assistant", "content": "Later response"},  # Late
            {"role": "user", "content": "Current query"},
        ]

        scores = score_messages(messages, "Current query", set())

        # Early messages (indices 0-3) should have higher scores
        early_scores = [s for s in scores if s.index < 4]
        late_scores = [s for s in scores if s.index >= 4 and s.index < 6]

        if early_scores and late_scores:
            avg_early = sum(s.score for s in early_scores) / len(early_scores)
            avg_late = sum(s.score for s in late_scores) / len(late_scores)
            assert avg_early > avg_late


class TestDropOrder:
    """Tests for drop order (ascending score, older-first ties)."""

    def test_lowest_score_dropped_first(self, counter):
        """Lowest score messages are dropped first."""
        # Create messages with varying relevance
        # Note: First 4 messages (indices 0-3 after system) are "early" and get +3 bonus
        # So we need late messages with low similarity to be lowest score
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Python programming tutorial guide"},  # Early, high sim
            {"role": "assistant", "content": "Python is great for beginners"},  # Early, medium sim
            {"role": "user", "content": "More about Python code"},  # Early
            {"role": "assistant", "content": "Python data structures"},  # Early
            {"role": "user", "content": "Unrelated cooking recipe discussion"},  # Late, no sim - LOWEST
            {"role": "assistant", "content": "More cooking stuff here"},  # Late, no sim - LOWEST
            {"role": "user", "content": "Python programming tutorial"},  # Current query
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=30,
            current_query="Python programming tutorial",
            counter=counter,
            anchor_indices=set(),
        )

        # First dropped should be lowest score (late messages with no relevance)
        if result.decisions:
            first_dropped = result.decisions[0]
            # Should be from the late, unrelated messages (indices 5, 6)
            assert first_dropped.message_index in {5, 6}

    def test_ties_broken_by_older_first(self, counter):
        """When scores are equal, older messages dropped first."""
        # Create messages with identical content (same score)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Hello world"},  # Same as index 1
            {"role": "assistant", "content": "Hi there"},  # Same as index 2
            {"role": "user", "content": "Current query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=100,
            current_query="Different query",
            counter=counter,
            anchor_indices=set(),
        )

        # With equal scores, older (lower index) should be dropped first
        if len(result.decisions) >= 2:
            indices = [d.message_index for d in result.decisions]
            # Index 1 should be dropped before index 3 (both have same content)
            if 1 in indices and 3 in indices:
                assert indices.index(1) < indices.index(3)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, counter):
        """Same input produces same output."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Query"},
        ]

        result1 = drop_by_importance(
            messages=messages,
            target_reduction=50,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        result2 = drop_by_importance(
            messages=messages,
            target_reduction=50,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        # Same messages should remain
        assert len(result1.messages) == len(result2.messages)

        # Same decisions should be made
        assert len(result1.decisions) == len(result2.decisions)

        for d1, d2 in zip(result1.decisions, result2.decisions):
            assert d1.message_index == d2.message_index
            assert d1.score == d2.score

    def test_score_computation_deterministic(self):
        """Score computation is deterministic."""
        msg = {"role": "user", "content": "Test message content"}

        score1 = score_message(
            message=msg,
            index=5,
            current_query="Test query",
            anchor_indices=set(),
            total_recency_count=10,
            referenced_indices=set(),
        )

        score2 = score_message(
            message=msg,
            index=5,
            current_query="Test query",
            anchor_indices=set(),
            total_recency_count=10,
            referenced_indices=set(),
        )

        assert score1.score == score2.score
        assert score1.lex_similarity == score2.lex_similarity


class TestTruncationLogging:
    """Tests for truncation decision logging."""

    def test_decisions_logged(self, counter):
        """Truncation decisions are logged."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "Query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=20,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        # Decisions should be logged
        for decision in result.decisions:
            assert decision.message_index >= 0
            assert decision.tokens_freed >= 0
            assert decision.reason != ""

    def test_scores_logged(self, counter):
        """Message scores are logged."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=10,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        # Scores should be logged
        assert len(result.scores) > 0
        for score in result.scores:
            assert hasattr(score, 'score')
            assert hasattr(score, 'is_anchor')
            assert hasattr(score, 'is_early')

    def test_result_serializable(self, counter):
        """Result can be serialized to dict."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=10,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        d = result.to_dict()
        assert "tokens_before" in d
        assert "tokens_after" in d
        assert "decisions" in d
        assert "scores" in d


class TestTruncateByRelevance:
    """Tests for truncate_by_relevance function."""

    def test_no_truncation_when_under_limit(self, counter):
        """No truncation when already under limit."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Short message"},
            {"role": "user", "content": "Query"},
        ]

        result = truncate_by_relevance(
            messages=messages,
            target_tokens=10000,  # Very high limit
            current_query="Query",
            counter=counter,
        )

        assert len(result.messages) == len(messages)
        assert len(result.decisions) == 0

    def test_truncation_when_over_limit(self, counter):
        """Truncation occurs when over limit."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "A" * 400},  # ~100 tokens
            {"role": "assistant", "content": "B" * 400},  # ~100 tokens
            {"role": "user", "content": "C" * 400},  # ~100 tokens
            {"role": "assistant", "content": "D" * 400},  # ~100 tokens
            {"role": "user", "content": "Query"},
        ]

        result = truncate_by_relevance(
            messages=messages,
            target_tokens=200,  # Low limit
            current_query="Query",
            counter=counter,
        )

        assert len(result.messages) < len(messages)
        assert len(result.decisions) > 0


class TestKeyWordExtraction:
    """Tests for key word extraction helper."""

    def test_filters_stop_words(self):
        """Stop words are filtered out."""
        words = _extract_key_words("the quick brown fox jumps over the lazy dog")
        assert "the" not in words
        assert "over" not in words
        assert "quick" in words
        assert "brown" in words
        assert "fox" in words

    def test_short_words_filtered(self):
        """Words shorter than 3 chars are filtered."""
        words = _extract_key_words("I am a cat")
        assert "am" not in words
        assert "cat" in words


class TestLongestCommonSubstring:
    """Tests for longest common substring helper."""

    def test_finds_common_substring(self):
        """Finds longest common substring."""
        length = _find_longest_common_substring(
            "Python is a great programming language",
            "I think Python is a great choice"
        )
        # "Python is a great" is common
        assert length >= 15

    def test_no_common_substring(self):
        """Returns 0 when no common substring."""
        length = _find_longest_common_substring("abc", "xyz")
        assert length == 0

    def test_empty_strings(self):
        """Handles empty strings."""
        assert _find_longest_common_substring("", "test") == 0
        assert _find_longest_common_substring("test", "") == 0


class TestReplayCompatibility:
    """Tests for replay compatibility."""

    def test_tokens_before_after_tracked(self, counter):
        """Tokens before and after are tracked."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "Query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=20,
            current_query="Query",
            counter=counter,
            anchor_indices=set(),
        )

        assert result.tokens_before > 0
        assert result.tokens_after >= 0
        assert result.tokens_before >= result.tokens_after

    def test_decision_reasons_logged(self, counter):
        """Decision reasons are logged for debugging."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Query"},
        ]

        result = drop_by_importance(
            messages=messages,
            target_reduction=10,
            current_query="Query",
            counter=counter,
            anchor_indices={1},  # One anchor
        )

        for decision in result.decisions:
            assert decision.reason in ["recency_low_score", "anchor_low_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
