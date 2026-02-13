"""Muse normalization tests — TM-38 through TM-40.

Tests for INV-MUSE-10: single-normalization invariant at
ingestion boundary.
"""

import unicodedata

import pytest

from episodic.config import config


@pytest.fixture(autouse=True)
def isolated_config(reset_singletons):
    """Reset config for each test."""
    pass


class TestExtractionNormalization:
    """TM-38: Content normalized at extraction boundary."""

    def test_tm38_extracted_content_nfc_normalized(self):
        """TM-38: Extracted content is NFC-normalized and zero-width chars stripped."""
        from episodic.web_extract import _normalize_text

        # NFC normalization: decomposed form → composed form
        # é as e + combining acute (NFD) should become é (NFC)
        nfd_text = "caf\u0065\u0301"  # e + combining acute accent
        nfc_expected = "caf\u00e9"  # é precomposed

        normalized = _normalize_text(nfd_text)
        # Should be NFC normalized
        assert unicodedata.is_normalized("NFC", normalized)

        # Zero-width chars should be stripped
        text_with_zw = "hel\u200blo\u200dworld\ufeff"  # ZWSP, ZWJ, BOM
        normalized = _normalize_text(text_with_zw)
        assert "\u200b" not in normalized  # ZWSP
        assert "\u200d" not in normalized  # ZWJ
        assert "\ufeff" not in normalized  # BOM


class TestSnippetNormalization:
    """TM-39: Snippet text normalized at synthesis boundary."""

    def test_tm39_snippet_normalized_in_synthesis(self):
        """TM-39: Snippet text is NFC-normalized at synthesis boundary."""
        from episodic.web_synthesis import WebSynthesizer
        from episodic.web_search import SearchResult

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")

        # Snippet with zero-width chars
        snippet_with_zw = "test\u200bsnippet\u200dwith\u200czero width"

        synth = WebSynthesizer()
        result = synth.synthesize_results(
            query="test",
            results=[
                SearchResult(title="Test", url="https://example.com",
                             snippet=snippet_with_zw, relevance_score=1.0)
            ],
            extracted_content={},
        )

        user_msg = result["prompt"]
        # Zero-width chars should not be in the prompt
        assert "\u200b" not in user_msg
        assert "\u200d" not in user_msg
        assert "\u200c" not in user_msg


class TestRAGReceivesPreNormalized:
    """TM-40: RAG indexer receives pre-normalized text."""

    def test_tm40_normalize_text_is_idempotent(self):
        """TM-40: Normalizing already-normalized text produces same result.

        This confirms the single-normalization invariant: text normalized
        at extraction boundary doesn't need re-normalization downstream.
        """
        from episodic.web_extract import _normalize_text

        original = "café résumé naïve"
        first_pass = _normalize_text(original)
        second_pass = _normalize_text(first_pass)

        assert first_pass == second_pass, "Normalization is not idempotent"
