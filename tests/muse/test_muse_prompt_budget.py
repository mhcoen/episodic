"""Muse prompt budget tests — TM-29 through TM-33, TM-41.

Tests for INV-MUSE-8: prompt budget enforcement, truncation order,
tag integrity, and canary placement assertions.
"""

import pytest
from unittest.mock import patch

from episodic.config import config
from episodic.web_search import SearchResult


@pytest.fixture(autouse=True)
def isolated_config(reset_singletons):
    """Reset config for each test."""
    pass


def _make_results(n, content_size=3000):
    """Create n SearchResults with large extracted content."""
    results = []
    extracted = {}
    for i in range(n):
        url = f"https://source{i}.com"
        results.append(SearchResult(
            title=f"Source {i}",
            url=url,
            snippet=f"Snippet for source {i}",
            relevance_score=1.0 - (i * 0.1),
        ))
        extracted[url] = "X" * content_size
    return results, extracted


class TestSystemMessagePreservation:
    """TM-29, TM-30: System message fully present after truncation."""

    def test_tm29_system_message_present_when_content_exceeds_budget(self):
        """TM-29: System message fully present when extracted content exceeds budget."""
        from episodic.web_synthesis import WebSynthesizer

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")
        config.set("muse_max_chars_per_source", 2000)
        config.set("muse_max_chars_total", 5000)  # Low total to trigger truncation

        synth = WebSynthesizer()
        results, extracted = _make_results(10, content_size=3000)

        result = synth.synthesize_results(
            query="test query",
            results=results,
            extracted_content=extracted,
        )

        system_msg = result["system_message"]
        # System message should contain all critical components
        assert "CRITICAL" in system_msg  # Anti-injection fence
        assert "synthesize" in system_msg.lower()  # Behavioral instructions

    def test_tm30_extracted_content_truncated_before_system(self):
        """TM-30: Extracted content truncated (sources dropped) before system message."""
        from episodic.web_synthesis import WebSynthesizer

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")
        config.set("muse_max_chars_per_source", 2000)
        config.set("muse_max_chars_total", 4000)

        synth = WebSynthesizer()
        results, extracted = _make_results(5, content_size=3000)

        result = synth.synthesize_results(
            query="test query",
            results=results,
            extracted_content=extracted,
        )

        user_msg = result["prompt"]
        system_msg = result["system_message"]

        # System message is fully present
        assert "CRITICAL" in system_msg

        # Not all sources should be in extracted content (budget exceeded)
        # Count how many <untrusted_content source="web:"> blocks
        web_blocks = user_msg.count('<untrusted_content source="web:')
        assert web_blocks < 5  # Some sources dropped


class TestTagIntegrity:
    """TM-31: No half-open <untrusted_content> tags after truncation."""

    def test_tm31_no_half_open_tags(self):
        """TM-31: All untrusted_content tags are properly closed."""
        from episodic.web_synthesis import WebSynthesizer

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")
        config.set("muse_max_chars_per_source", 2000)
        config.set("muse_max_chars_total", 3000)

        synth = WebSynthesizer()
        results, extracted = _make_results(8, content_size=2000)

        result = synth.synthesize_results(
            query="test query",
            results=results,
            extracted_content=extracted,
        )

        user_msg = result["prompt"]

        # Count opens and closes
        opens = user_msg.count("<untrusted_content")
        closes = user_msg.count("</untrusted_content>")
        assert opens == closes, f"Tag mismatch: {opens} opens, {closes} closes"


class TestPerSourceCap:
    """TM-32: Per-source cap enforced."""

    def test_tm32_per_source_cap_enforced(self):
        """TM-32: Each source's content is capped at muse_max_chars_per_source."""
        from episodic.web_synthesis import WebSynthesizer

        cap = 500
        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")
        config.set("muse_max_chars_per_source", cap)
        config.set("muse_max_chars_total", 50000)  # High total

        synth = WebSynthesizer()
        results, extracted = _make_results(2, content_size=5000)

        result = synth.synthesize_results(
            query="test query",
            results=results,
            extracted_content=extracted,
        )

        user_msg = result["prompt"]

        # Each source block content should be at most `cap` chars
        # The "XXXXX..." pattern should be capped
        # Find content between From Source tags
        import re
        blocks = re.findall(
            r'From Source \[\d+\]:\n(.*?)\n</untrusted_content>',
            user_msg, re.DOTALL,
        )
        for block in blocks:
            assert len(block) <= cap, f"Source content {len(block)} exceeds cap {cap}"


class TestTotalCap:
    """TM-33: Total cap enforced."""

    def test_tm33_total_cap_enforced(self):
        """TM-33: Total extracted content capped at muse_max_chars_total."""
        from episodic.web_synthesis import WebSynthesizer

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")
        config.set("muse_max_chars_per_source", 2000)
        config.set("muse_max_chars_total", 3000)

        synth = WebSynthesizer()
        results, extracted = _make_results(5, content_size=2000)

        result = synth.synthesize_results(
            query="test query",
            results=results,
            extracted_content=extracted,
        )

        user_msg = result["prompt"]

        # Extract total chars in extracted content blocks
        import re
        blocks = re.findall(
            r'From Source \[\d+\]:\n(.*?)\n</untrusted_content>',
            user_msg, re.DOTALL,
        )
        total_chars = sum(len(b) for b in blocks)
        assert total_chars <= 3000 + 100  # Small margin for tag overhead


class TestCanaryNotInUserMessage:
    """TM-41: Canary token does not appear in user-message region (Erratum 4)."""

    def test_tm41_canary_not_in_user_message(self):
        """TM-41: Session canary does not appear in user-message region."""
        from episodic.web_synthesis import WebSynthesizer

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")

        canary = "CANARY-test12345678"

        synth = WebSynthesizer()
        result = synth.synthesize_results(
            query="test query",
            results=[
                SearchResult(title="T", url="https://example.com",
                             snippet="s", relevance_score=1.0)
            ],
            extracted_content={},
            session_canary=canary,
        )

        user_msg = result["prompt"]
        system_msg = result["system_message"]

        # Canary should be in system message
        assert canary in system_msg

        # Canary must NOT be in user message
        assert canary not in user_msg
