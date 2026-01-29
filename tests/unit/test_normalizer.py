"""Unit tests for MQL normalizer."""

import pytest
from episodic.query import normalize, PUNCT_MAP


class TestNormalize:
    """Tests for the normalize function."""

    def test_smart_quotes_to_ascii(self):
        """Smart quotes should be normalized to ASCII quotes."""
        s_norm, audit = normalize('\u201cHello\u201d')  # "Hello"
        assert s_norm == '"Hello"'
        # Check that both left and right quotes were replaced
        assert any('\u201c' in c for c in audit.changes)
        assert any('\u201d' in c for c in audit.changes)

    def test_left_right_double_quotes(self):
        """Both left and right double quotes should normalize."""
        s_norm, audit = normalize('"test"')
        assert s_norm == '"test"'

    def test_single_smart_quotes(self):
        """Smart single quotes should normalize to ASCII."""
        s_norm, audit = normalize("'don't'")  # Using smart quotes
        assert s_norm == "'don't'"

    def test_em_dash_to_hyphen(self):
        """Em dash should normalize to hyphen."""
        s_norm, audit = normalize("foo—bar")
        assert s_norm == "foo-bar"
        assert any("'—'" in c for c in audit.changes)

    def test_en_dash_to_hyphen(self):
        """En dash should normalize to hyphen."""
        s_norm, audit = normalize("foo–bar")
        assert s_norm == "foo-bar"

    def test_nbsp_to_space(self):
        """Non-breaking space should normalize to regular space."""
        s_norm, audit = normalize("hello\u00A0world")
        assert s_norm == "hello world"

    def test_em_space_to_space(self):
        """Em space should normalize to regular space."""
        s_norm, audit = normalize("hello\u2003world")
        assert s_norm == "hello world"

    def test_thin_space_to_space(self):
        """Thin space should normalize to regular space."""
        s_norm, audit = normalize("hello\u2009world")
        assert s_norm == "hello world"

    def test_whitespace_collapse(self):
        """Multiple spaces should collapse to single space."""
        s_norm, audit = normalize("hello    world")
        assert s_norm == "hello world"
        assert "collapsed whitespace" in audit.changes

    def test_newline_collapse(self):
        """Newlines should collapse to single space."""
        s_norm, audit = normalize("hello\n\nworld")
        assert s_norm == "hello world"

    def test_trim_leading_whitespace(self):
        """Leading whitespace should be trimmed."""
        s_norm, audit = normalize("  hello")
        assert s_norm == "hello"
        assert "trimmed whitespace" in audit.changes

    def test_trim_trailing_whitespace(self):
        """Trailing whitespace should be trimmed."""
        s_norm, audit = normalize("hello  ")
        assert s_norm == "hello"

    def test_no_lowercase(self):
        """Normalization should NOT lowercase (preserves quoted content)."""
        s_norm, audit = normalize("Hello WORLD")
        assert s_norm == "Hello WORLD"

    def test_audit_record_includes_raw(self):
        """Audit should include raw input."""
        s_norm, audit = normalize('"test"')
        assert audit.raw == '"test"'
        assert audit.normalized == '"test"'

    def test_no_changes_for_clean_input(self):
        """Clean input should have no changes recorded."""
        s_norm, audit = normalize("hello world")
        assert s_norm == "hello world"
        assert len(audit.changes) == 0

    def test_multiple_normalizations(self):
        """Multiple normalizations should all be recorded."""
        s_norm, audit = normalize('  "hello"  ')
        assert s_norm == '"hello"'
        # Should have smart quote and whitespace changes
        assert len(audit.changes) >= 2

    def test_empty_input(self):
        """Empty input should work."""
        s_norm, audit = normalize("")
        assert s_norm == ""
        assert len(audit.changes) == 0

    def test_whitespace_only(self):
        """Whitespace-only input should normalize to empty."""
        s_norm, audit = normalize("   ")
        assert s_norm == ""


class TestPunctMap:
    """Tests for PUNCT_MAP coverage."""

    def test_all_punct_map_chars_handled(self):
        """All characters in PUNCT_MAP should be transformed."""
        for old, new in PUNCT_MAP.items():
            s_norm, audit = normalize(f"a{old}b")
            assert new in s_norm
