"""
Tests for topic alias extraction and matching.
"""

import pytest

from episodic.recall.topic_aliases import (
    _normalize_term,
    _extract_terms,
    compute_alias_score,
    STOPWORDS,
    MIN_TERM_LENGTH,
)


class TestNormalizeTerm:
    """Tests for _normalize_term function."""

    def test_normalizes_to_lowercase(self):
        """Test that terms are lowercased."""
        assert _normalize_term("Python") == "python"
        assert _normalize_term("RETRY") == "retry"

    def test_removes_non_alphanumeric(self):
        """Test that non-alphanumeric chars are removed."""
        assert _normalize_term("python-retry") == "pythonretry"
        assert _normalize_term("api_key") == "apikey"

    def test_filters_short_terms(self):
        """Test that short terms are filtered out."""
        assert _normalize_term("a") is None
        assert _normalize_term("ab") is None
        assert _normalize_term("abc") == "abc"

    def test_filters_stopwords(self):
        """Test that stopwords are filtered out."""
        assert _normalize_term("the") is None
        assert _normalize_term("and") is None
        assert _normalize_term("with") is None

    def test_conversation_stopwords(self):
        """Test that conversation-specific stopwords are filtered."""
        assert _normalize_term("discussed") is None
        assert _normalize_term("mentioned") is None
        assert _normalize_term("thing") is None


class TestExtractTerms:
    """Tests for _extract_terms function."""

    def test_extracts_from_text(self):
        """Test basic term extraction."""
        terms = _extract_terms("Python retry patterns")
        assert "python" in terms
        assert "retry" in terms
        assert "patterns" in terms

    def test_splits_on_punctuation(self):
        """Test splitting on various punctuation."""
        terms = _extract_terms("python-retry/patterns,tenacity")
        assert "python" in terms
        assert "retry" in terms
        assert "patterns" in terms
        assert "tenacity" in terms

    def test_filters_stopwords(self):
        """Test that stopwords are removed."""
        terms = _extract_terms("the Python and retry patterns")
        assert "the" not in terms
        assert "and" not in terms
        assert "python" in terms

    def test_empty_text(self):
        """Test that empty text returns empty set."""
        assert _extract_terms("") == set()
        assert _extract_terms(None) == set()


class TestComputeAliasScore:
    """Tests for compute_alias_score function."""

    def test_counts_matching_terms(self):
        """Test that alias hits are counted correctly."""
        aliases = {"python", "retry", "tenacity", "patterns"}

        # Query with 2 matches
        score = compute_alias_score("Back to that Python retry thing", aliases)
        assert score == 2  # python, retry

    def test_no_matches(self):
        """Test score is 0 when no matches."""
        aliases = {"python", "retry"}
        score = compute_alias_score("What about the database?", aliases)
        assert score == 0

    def test_empty_aliases(self):
        """Test score is 0 when aliases are empty."""
        score = compute_alias_score("Python retry patterns", set())
        assert score == 0

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        aliases = {"python", "retry"}
        score = compute_alias_score("PYTHON and RETRY", aliases)
        assert score == 2

    def test_multiple_distinct_terms(self):
        """Test counting distinct term matches."""
        aliases = {"python", "retry", "tenacity", "api", "endpoint"}

        # Query with 3 distinct matches
        score = compute_alias_score(
            "Back to Python retry with tenacity",
            aliases
        )
        assert score == 3  # python, retry, tenacity


class TestStopwords:
    """Tests for stopword list."""

    def test_common_english_stopwords(self):
        """Test that common English stopwords are included."""
        common = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to"]
        for word in common:
            assert word in STOPWORDS, f"Missing common stopword: {word}"

    def test_conversation_stopwords(self):
        """Test that conversation-specific stopwords are included."""
        conversation = ["discussed", "talked", "mentioned", "thing", "stuff"]
        for word in conversation:
            assert word in STOPWORDS, f"Missing conversation stopword: {word}"


class TestMinTermLength:
    """Tests for minimum term length constant."""

    def test_min_length_is_reasonable(self):
        """Test that MIN_TERM_LENGTH is a reasonable value."""
        assert MIN_TERM_LENGTH >= 2
        assert MIN_TERM_LENGTH <= 5
