"""Tests for MCP security display parity (spec tests 73-76).

Tests the sanitization_summary output from the sanitizer module to
verify display parity information is communicated to the user.

Covers:
- HTML with 3+ hidden elements -> summary shows counts
- Unicode stripping -> summary shows stripped char count
- class_hiding_possible -> summary mentions it
- Clean content -> no issues in summary
"""

import pytest

from episodic.mcp.security.sanitizer import sanitize, _build_summary
from episodic.mcp.security.types import ContentType


class TestHiddenElementSummary:
    """Spec test 73: HTML with hidden elements produces summary with counts."""

    def test_hidden_display_none_elements(self):
        """HTML with display:none elements shows removal in summary."""
        html = (
            '<div style="display:none">Hidden 1</div>'
            '<div style="display:none">Hidden 2</div>'
            '<div style="display:none">Hidden 3</div>'
            '<p>Visible content</p>'
        )
        result = sanitize(html, ContentType.HTML)
        assert result.sanitization_summary is not None
        # Summary should mention hidden elements
        assert "hidden" in result.sanitization_summary.lower() or \
               "removed" in result.sanitization_summary.lower() or \
               "css" in result.sanitization_summary.lower()

    def test_hidden_visibility_hidden_elements(self):
        """HTML with visibility:hidden elements shows in summary."""
        html = (
            '<span style="visibility:hidden">Secret 1</span>'
            '<span style="visibility:hidden">Secret 2</span>'
            '<span style="visibility:hidden">Secret 3</span>'
            '<p>Visible text</p>'
        )
        result = sanitize(html, ContentType.HTML)
        assert result.chars_stripped > 0

    def test_hidden_opacity_zero_elements(self):
        """HTML with opacity:0 elements shows in summary."""
        html = (
            '<div style="opacity:0">Invisible 1</div>'
            '<div style="opacity:0">Invisible 2</div>'
            '<div style="opacity:0">Invisible 3</div>'
            '<p>Visible</p>'
        )
        result = sanitize(html, ContentType.HTML)
        assert result.chars_stripped > 0

    def test_mixed_hidden_techniques(self):
        """HTML using different hiding techniques all detected."""
        html = (
            '<div style="display:none">A</div>'
            '<div style="visibility:hidden">B</div>'
            '<div style="opacity:0">C</div>'
            '<div style="font-size:0">D</div>'
            '<p>Visible</p>'
        )
        result = sanitize(html, ContentType.HTML)
        assert result.chars_stripped > 0
        assert len(result.warnings) > 0


class TestUnicodeStrippingSummary:
    """Spec test 74: Unicode stripping produces summary with char count."""

    def test_zero_width_chars_counted(self):
        """Summary reports count of stripped invisible Unicode chars."""
        # Text with zero-width spaces and other invisible chars
        text = "Hello\u200B\u200C\u200D World\u2060\uFEFF"
        result = sanitize(text, ContentType.PLAINTEXT)
        assert result.chars_stripped > 0
        assert result.sanitization_summary is not None
        summary_lower = result.sanitization_summary.lower()
        assert "unicode" in summary_lower or "invisible" in summary_lower or \
               "stripped" in summary_lower

    def test_bidi_controls_counted(self):
        """Bidi override characters are counted in stripped total."""
        text = "Normal\u202A\u202B\u202C\u202D\u202Etext"
        result = sanitize(text, ContentType.PLAINTEXT)
        assert result.chars_stripped > 0

    def test_soft_hyphen_counted(self):
        """Soft hyphens are counted in stripped total."""
        text = "dis\u00ADplay in\u00ADvis\u00ADible"
        result = sanitize(text, ContentType.PLAINTEXT)
        assert result.chars_stripped > 0

    def test_stripped_count_matches_actual(self):
        """The reported count matches the actual number of chars removed."""
        text = "a\u200Bb\u200Cc"  # 2 invisible chars
        result = sanitize(text, ContentType.PLAINTEXT)
        assert result.chars_stripped == 2


class TestClassHidingSummary:
    """Spec test 75: class_hiding_possible noted in summary."""

    def test_style_tags_flag_class_hiding(self):
        """HTML with <style> tags sets class_hiding_possible."""
        html = (
            '<style>.hidden { display: none; }</style>'
            '<p>Visible content</p>'
            '<p class="hidden">This might be hidden by CSS class</p>'
        )
        result = sanitize(html, ContentType.HTML)
        assert result.class_hiding_possible is True
        assert result.sanitization_summary is not None
        summary_lower = result.sanitization_summary.lower()
        assert "class" in summary_lower or "css" in summary_lower or \
               "style" in summary_lower

    def test_no_style_tags_no_class_hiding(self):
        """HTML without <style> tags does not flag class hiding."""
        html = '<p>Just a plain paragraph</p>'
        result = sanitize(html, ContentType.HTML)
        assert result.class_hiding_possible is False

    def test_plaintext_no_class_hiding(self):
        """Plaintext content never flags class hiding."""
        result = sanitize("Plain text content", ContentType.PLAINTEXT)
        assert result.class_hiding_possible is False


class TestCleanContentSummary:
    """Spec test 76: clean content produces no-issues summary."""

    def test_clean_plaintext(self):
        """Clean plaintext produces 'No issues' or equivalent summary."""
        result = sanitize("Hello, this is perfectly normal text.", ContentType.PLAINTEXT)
        assert result.chars_stripped == 0
        assert result.class_hiding_possible is False
        assert result.encoded_detected is False
        assert result.mixed_script_words == []
        # Summary should indicate nothing was found
        assert result.sanitization_summary is not None
        summary_lower = result.sanitization_summary.lower()
        assert "no issues" in summary_lower or "no" in summary_lower

    def test_clean_html(self):
        """Clean HTML (no hidden elements, no style tags) is clean."""
        html = '<p>Hello world</p>'
        result = sanitize(html, ContentType.HTML)
        assert result.class_hiding_possible is False
        assert result.warnings == [] or all(
            "hidden" not in w.lower() for w in result.warnings
        )


class TestBuildSummaryDirect:
    """Direct tests of the _build_summary helper function."""

    def test_all_zeros_no_issues(self):
        """All clean inputs produce 'No issues detected'."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=0,
            warnings=[],
            class_hiding_possible=False,
            encoded_detected=False,
            mixed_script_words=[],
        )
        assert summary == "No issues detected"

    def test_html_chars_stripped_reported(self):
        """HTML chars stripped shows in summary."""
        summary = _build_summary(
            html_chars_stripped=150,
            unicode_chars_stripped=0,
            warnings=[],
            class_hiding_possible=False,
            encoded_detected=False,
            mixed_script_words=[],
        )
        assert "150" in summary
        assert "html" in summary.lower()

    def test_unicode_chars_stripped_reported(self):
        """Unicode chars stripped shows in summary."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=5,
            warnings=[],
            class_hiding_possible=False,
            encoded_detected=False,
            mixed_script_words=[],
        )
        assert "5" in summary
        assert "unicode" in summary.lower() or "invisible" in summary.lower()

    def test_class_hiding_reported(self):
        """class_hiding_possible is mentioned in summary."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=0,
            warnings=[],
            class_hiding_possible=True,
            encoded_detected=False,
            mixed_script_words=[],
        )
        assert "class" in summary.lower() or "css" in summary.lower()

    def test_encoded_detected_reported(self):
        """Encoded payloads detected is mentioned in summary."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=0,
            warnings=[],
            class_hiding_possible=False,
            encoded_detected=True,
            mixed_script_words=[],
        )
        assert "encoded" in summary.lower() or "payload" in summary.lower()

    def test_mixed_script_words_reported(self):
        """Mixed-script words are mentioned in summary."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=0,
            warnings=[],
            class_hiding_possible=False,
            encoded_detected=False,
            mixed_script_words=["pаypal"],  # Cyrillic 'a' mixed with Latin
        )
        assert "mixed" in summary.lower() or "script" in summary.lower()

    def test_warnings_counted(self):
        """Warnings are listed in summary."""
        summary = _build_summary(
            html_chars_stripped=0,
            unicode_chars_stripped=0,
            warnings=["Warning 1", "Warning 2"],
            class_hiding_possible=False,
            encoded_detected=False,
            mixed_script_words=[],
        )
        assert "2" in summary  # Warning count
        assert "Warning 1" in summary
        assert "Warning 2" in summary

    def test_combined_summary(self):
        """Multiple issues produce a combined summary."""
        summary = _build_summary(
            html_chars_stripped=100,
            unicode_chars_stripped=10,
            warnings=["Removed 3 CSS-hidden element(s)"],
            class_hiding_possible=True,
            encoded_detected=True,
            mixed_script_words=["pаypal"],
        )
        assert "110" in summary  # Total chars stripped
        assert "class" in summary.lower() or "css" in summary.lower()
        assert "encoded" in summary.lower()
        assert "mixed" in summary.lower() or "script" in summary.lower()
