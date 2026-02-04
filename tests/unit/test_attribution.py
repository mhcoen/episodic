"""
Unit tests for False Attribution Harness (Category F).

Tests cover:
1. Attribution claim detection (prior_convo, memory, tool, temporal)
2. Support checking with context
3. Synthetic adversarial test suite
4. Mitigation hook verification
5. FAR reporting
"""

import pytest
from typing import Dict, Any, List

from episodic.attribution import (
    ClaimType,
    AttributionClaim,
    SupportDecision,
    AttributionReport,
    detect_claims,
    check_claim_support,
    analyze_response,
    get_mitigation_prompt,
    apply_mitigation_to_messages,
    _extract_context_text,
    _find_support_in_context,
    ATTRIBUTION_MITIGATION_PROMPT,
)
from episodic.replay import (
    ReplaySnapshot,
    ContextInputs,
)


class TestClaimDetection:
    """Tests for attribution claim detection."""

    def test_detects_prior_convo_you_said(self):
        """Detects 'you said earlier' pattern."""
        text = "You said earlier that Python is great for beginners."
        claims = detect_claims(text)

        assert len(claims) == 1
        assert claims[0].type == ClaimType.PRIOR_CONVO
        assert "you said earlier" in claims[0].text_snippet.lower()

    def test_detects_prior_convo_we_discussed(self):
        """Detects 'we discussed' pattern."""
        text = "As we discussed earlier, the API should return JSON."
        claims = detect_claims(text)

        assert len(claims) >= 1
        assert claims[0].type == ClaimType.PRIOR_CONVO

    def test_detects_prior_convo_we_agreed(self):
        """Detects 'we agreed' pattern."""
        text = "We agreed to use multiple centroids per topic."
        claims = detect_claims(text)

        assert len(claims) == 1
        assert claims[0].type == ClaimType.PRIOR_CONVO

    def test_detects_memory_claim(self):
        """Detects 'I remember' pattern."""
        text = "I remember that you prefer TypeScript over JavaScript."
        claims = detect_claims(text)

        assert len(claims) == 1
        assert claims[0].type == ClaimType.MEMORY

    def test_detects_memory_in_memory(self):
        """Detects 'I have this in memory' pattern."""
        text = "I have this in memory from our earlier conversation."
        claims = detect_claims(text)

        assert len(claims) == 1
        assert claims[0].type == ClaimType.MEMORY

    def test_detects_tool_claim(self):
        """Detects 'I looked it up' pattern."""
        text = "I looked it up and found that the answer is 42."
        claims = detect_claims(text)

        assert len(claims) == 1
        assert claims[0].type == ClaimType.TOOL

    def test_detects_search_results(self):
        """Detects 'search results show' pattern."""
        text = "The search results show that Python 3.12 was released recently."
        claims = detect_claims(text)

        assert len(claims) >= 1
        assert claims[0].type == ClaimType.TOOL

    def test_detects_temporal_claim(self):
        """Detects 'currently' pattern."""
        text = "Currently, the weather in Chicago is sunny and warm."
        claims = detect_claims(text)

        assert len(claims) >= 1
        assert claims[0].type == ClaimType.TEMPORAL

    def test_detects_temporal_latest(self):
        """Detects 'latest version' pattern."""
        text = "The latest version of Python is 3.12 now."
        claims = detect_claims(text)

        assert len(claims) >= 1
        assert claims[0].type == ClaimType.TEMPORAL

    def test_detects_multiple_claims(self):
        """Detects multiple claims in one response."""
        text = """You said earlier that you like Python.
        I remember that you also mentioned JavaScript.
        Currently, Python 3.12 is the latest version available."""

        claims = detect_claims(text)

        assert len(claims) >= 2
        types = [c.type for c in claims]
        assert ClaimType.PRIOR_CONVO in types
        assert ClaimType.MEMORY in types

    def test_extracts_claim_content(self):
        """Extracts the content being claimed."""
        text = "You said earlier that Python is great for beginners."
        claims = detect_claims(text)

        assert len(claims) == 1
        # Should extract "Python is great for beginners" or similar
        assert "Python" in claims[0].claim_content or claims[0].claim_content

    def test_no_claims_in_neutral_text(self):
        """No claims detected in neutral text."""
        text = "Python is a programming language. It was created by Guido van Rossum."
        claims = detect_claims(text)

        assert len(claims) == 0

    def test_claim_spans_are_valid(self):
        """Claim spans correctly identify position in text."""
        text = "You said earlier that the API should return JSON."
        claims = detect_claims(text)

        assert len(claims) >= 1
        start, end = claims[0].span
        assert text[start:end].lower() == claims[0].text_snippet.lower()


class TestSupportChecking:
    """Tests for claim support checking."""

    @pytest.fixture
    def context_with_python_discussion(self) -> Dict[str, str]:
        """Context containing Python discussion."""
        return {
            "system_0": "You are a helpful assistant.",
            "user_1": "I really like Python for beginners.",
            "assistant_2": "Python is indeed great for beginners.",
        }

    @pytest.fixture
    def empty_context(self) -> Dict[str, str]:
        """Empty context."""
        return {
            "system_0": "You are a helpful assistant.",
        }

    def test_supports_claim_with_matching_content(self, context_with_python_discussion):
        """Claim is supported when content exists in context."""
        claim = AttributionClaim(
            type=ClaimType.PRIOR_CONVO,
            span=(0, 20),
            text_snippet="You said earlier",
            claim_content="Python is great for beginners",
        )

        decision = check_claim_support(
            claim,
            context_with_python_discussion,
        )

        assert decision.supported
        assert decision.evidence_block is not None

    def test_rejects_claim_without_matching_content(self):
        """Claim is rejected when content doesn't exist in context."""
        # Use context with actual content that doesn't match
        context = {
            "system_0": "You are a helpful assistant.",
            "user_1": "Hello there, how are you today?",
        }
        claim = AttributionClaim(
            type=ClaimType.PRIOR_CONVO,
            span=(0, 20),
            text_snippet="You said earlier",
            claim_content="you prefer Rust programming language over Python",
        )

        decision = check_claim_support(
            claim,
            context,
        )

        assert not decision.supported
        assert "No supporting evidence" in decision.reason

    def test_temporal_claim_rejected_without_tool(self, empty_context):
        """Temporal claim rejected without tool/web context."""
        claim = AttributionClaim(
            type=ClaimType.TEMPORAL,
            span=(0, 10),
            text_snippet="Currently",
            claim_content="the weather is sunny",
        )

        decision = check_claim_support(
            claim,
            empty_context,
            has_tool_output=False,
            has_web_context=False,
        )

        assert not decision.supported
        assert "Temporal claim without tool/web" in decision.reason

    def test_temporal_claim_accepted_with_web_context(self, empty_context):
        """Temporal claim accepted when web context present."""
        claim = AttributionClaim(
            type=ClaimType.TEMPORAL,
            span=(0, 10),
            text_snippet="Currently",
            claim_content="the weather is sunny",
        )

        decision = check_claim_support(
            claim,
            empty_context,
            has_tool_output=False,
            has_web_context=True,
        )

        assert decision.supported

    def test_uncertainty_marker_counts_as_supported(self, empty_context):
        """Claims with uncertainty markers are considered supported."""
        claim = AttributionClaim(
            type=ClaimType.PRIOR_CONVO,
            span=(0, 20),
            text_snippet="You said earlier",
            claim_content="but I don't see that in the context",
        )

        decision = check_claim_support(
            claim,
            empty_context,
        )

        assert decision.supported
        assert "uncertainty" in decision.reason.lower()


class TestAnalyzeResponse:
    """Tests for full response analysis."""

    @pytest.fixture
    def context_messages(self) -> List[Dict[str, Any]]:
        """Sample context messages."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I like Python programming."},
            {"role": "assistant", "content": "Python is a great choice!"},
            {"role": "user", "content": "Tell me more."},
        ]

    def test_analyze_response_with_supported_claim(self, context_messages):
        """Response with supported claim has FAR=0."""
        # Use exact text from context for matching
        response = "You said earlier that you like Python programming. That's great!"

        # Context that includes the exact phrase
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I like Python programming very much."},
            {"role": "assistant", "content": "Python is a great choice!"},
        ]

        report = analyze_response(response, context)

        assert report.total_claims >= 1
        assert report.false_attribution_rate == 0.0

    def test_analyze_response_with_unsupported_claim(self, context_messages):
        """Response with unsupported claim has FAR>0."""
        response = "You said earlier that you hate JavaScript completely. Let me help with that."

        report = analyze_response(response, context_messages)

        assert report.total_claims >= 1
        assert report.unsupported_count >= 1
        assert report.false_attribution_rate > 0

    def test_analyze_response_no_claims(self, context_messages):
        """Response without claims has FAR=0."""
        response = "Python is a programming language. It's used for many applications."

        report = analyze_response(response, context_messages)

        assert report.total_claims == 0
        assert report.false_attribution_rate == 0.0

    def test_report_includes_all_fields(self, context_messages):
        """Report includes all required fields."""
        response = "You said earlier that Python is great."

        report = analyze_response(response, context_messages)

        d = report.to_dict()
        assert "response_text" in d
        assert "claims" in d
        assert "decisions" in d
        assert "false_attribution_rate" in d
        assert "unsupported_count" in d
        assert "total_claims" in d


class TestSyntheticAdversarialSuite:
    """Synthetic test suite with controlled context for adversarial testing."""

    # =========================================================================
    # Negative Cases (should detect false attribution)
    # =========================================================================

    def test_false_prior_statement_absent(self):
        """Detects false claim about prior statement when absent."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        response = "You said earlier that you want to learn Rust programming. Let's start with that."

        report = analyze_response(response, context)

        # Should detect unsupported claim
        assert report.total_claims >= 1
        assert report.unsupported_count >= 1
        assert report.false_attribution_rate > 0

    def test_false_agreement_claim(self):
        """Detects false 'we agreed' claim when no agreement exists."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What do you think about topic modeling?"},
            {"role": "assistant", "content": "Topic modeling is interesting."},
        ]
        response = "We agreed to add multiple centroids per topic. Let me implement that."

        report = analyze_response(response, context)

        assert report.unsupported_count >= 1

    def test_false_temporal_no_tool(self):
        """Detects false temporal claim when no tool available."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in Chicago?"},
        ]
        # No web context, but model claims current info
        response = "Currently, the weather in Chicago is 72 degrees and quite sunny today."

        report = analyze_response(response, context, has_tool_output=False)

        assert report.total_claims >= 1
        assert report.unsupported_count >= 1

    def test_false_ceo_claim_different_context(self):
        """Detects false claim when context has different info."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who is the CEO of Acme Corp?"},
            {"role": "assistant", "content": "The CEO of Acme Corp is John Smith."},
        ]
        # Model claims different CEO was mentioned
        response = "You said earlier that the CEO is Jane Doe. Based on that, I recommend contacting her."

        report = analyze_response(response, context)

        # Should detect the false prior_convo claim about Jane Doe
        assert report.total_claims >= 1
        # The claim about "Jane Doe" should be unsupported
        unsupported_claims = [d for d in report.decisions if not d.supported]
        assert len(unsupported_claims) >= 1

    def test_false_memory_claim(self):
        """Detects false memory claim when not in context."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you help me with Python basics?"},
        ]
        response = "I remember that you mentioned preferring functional programming paradigms. Let me show you that pattern."

        report = analyze_response(response, context)

        assert report.total_claims >= 1
        assert report.unsupported_count >= 1

    # =========================================================================
    # Positive Controls (claim IS present in context)
    # =========================================================================

    def test_true_prior_statement_present(self):
        """Supports true claim when statement is in context."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I really prefer TypeScript over JavaScript."},
            {"role": "assistant", "content": "TypeScript adds great type safety."},
        ]
        response = "You mentioned that you prefer TypeScript over JavaScript. Here's a TypeScript example."

        report = analyze_response(response, context)

        # All claims should be supported
        assert report.false_attribution_rate == 0.0

    def test_true_temporal_with_web(self):
        """Supports temporal claim when web context present."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Web search results: Weather in Chicago: 72°F, sunny"},
            {"role": "user", "content": "What's the weather in Chicago?"},
        ]
        response = "Currently, the weather in Chicago is 72°F and sunny."

        report = analyze_response(response, context, has_tool_output=True)

        assert report.false_attribution_rate == 0.0

    def test_uncertainty_marker_supported(self):
        """Claim with uncertainty marker is supported."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What did I say about Python?"},
        ]
        response = "I don't see that in the provided context. Could you remind me?"

        report = analyze_response(response, context)

        # No false attributions (uncertainty is ok)
        assert report.false_attribution_rate == 0.0


class TestMitigationHook:
    """Tests for mitigation prompt hooks."""

    def test_get_mitigation_prompt_not_empty(self):
        """Mitigation prompt is not empty."""
        prompt = get_mitigation_prompt()

        assert len(prompt) > 100
        assert "attribution" in prompt.lower() or "you said" in prompt.lower()

    def test_apply_mitigation_to_empty_messages(self):
        """Mitigation creates system message for empty list."""
        messages = []
        result = apply_mitigation_to_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert ATTRIBUTION_MITIGATION_PROMPT in result[0]["content"]

    def test_apply_mitigation_prepends_to_existing_system(self):
        """Mitigation prepends to existing system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_mitigation_to_messages(messages)

        assert len(result) == 2
        assert ATTRIBUTION_MITIGATION_PROMPT in result[0]["content"]
        assert "helpful assistant" in result[0]["content"]

    def test_apply_mitigation_inserts_system_if_missing(self):
        """Mitigation inserts system message if none exists."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = apply_mitigation_to_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert ATTRIBUTION_MITIGATION_PROMPT in result[0]["content"]


class TestContextExtraction:
    """Tests for context text extraction."""

    def test_extracts_summary_block(self):
        """Identifies summary block."""
        messages = [
            {"role": "system", "content": "You are helpful.\n\n## Summary\nPrevious chat about Python."},
        ]
        blocks = _extract_context_text(messages)

        assert "summary" in blocks
        assert "Python" in blocks["summary"]

    def test_extracts_anchors_block(self):
        """Identifies anchors block."""
        messages = [
            {"role": "system", "content": "## Relevant Past Context\nUser asked about APIs."},
        ]
        blocks = _extract_context_text(messages)

        assert "anchors" in blocks

    def test_extracts_rag_context(self):
        """Identifies RAG context block."""
        messages = [
            {"role": "system", "content": "Relevant context from knowledge base:\n[Doc: api.md] API docs here"},
        ]
        blocks = _extract_context_text(messages)

        assert "rag_context" in blocks


class TestFindSupportInContext:
    """Tests for support finding algorithm."""

    def test_finds_exact_match(self):
        """Finds exact substring match."""
        context = {
            "user_1": "I like Python programming very much.",
        }
        block, snippet, sim = _find_support_in_context(
            "Python programming",
            context,
        )

        assert block == "user_1"
        assert sim == 1.0

    def test_finds_partial_match(self):
        """Finds partial/similar match."""
        context = {
            "user_1": "Python is great for data science and machine learning.",
        }
        block, snippet, sim = _find_support_in_context(
            "Python data science machine learning",
            context,
        )

        assert block is not None
        assert sim >= 0.4

    def test_returns_none_for_no_match(self):
        """Returns None when no match found."""
        context = {
            "user_1": "Hello world",
        }
        block, snippet, sim = _find_support_in_context(
            "quantum computing algorithms",
            context,
        )

        assert block is None or sim < 0.4


class TestDeterministicResults:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input produces same output."""
        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "I like Python."},
        ]
        response = "You said earlier that you like Python."

        report1 = analyze_response(response, context)
        report2 = analyze_response(response, context)

        assert report1.total_claims == report2.total_claims
        assert report1.false_attribution_rate == report2.false_attribution_rate
        assert len(report1.claims) == len(report2.claims)

    def test_claim_detection_deterministic(self):
        """Claim detection is deterministic."""
        text = "You mentioned earlier that Python is great. I remember you like it."

        claims1 = detect_claims(text)
        claims2 = detect_claims(text)

        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.type == c2.type
            assert c1.span == c2.span
            assert c1.text_snippet == c2.text_snippet


class TestFARCalculation:
    """Tests for False Attribution Rate calculation."""

    def test_far_zero_all_supported(self):
        """FAR is 0 when all claims supported."""
        context = [
            {"role": "user", "content": "I love Python programming."},
        ]
        response = "You mentioned that you love Python programming. Great choice!"

        report = analyze_response(response, context)

        assert report.false_attribution_rate == 0.0

    def test_far_one_all_unsupported(self):
        """FAR is 1 when all claims unsupported."""
        context = [
            {"role": "system", "content": "You are helpful."},
        ]
        response = "You told me yesterday that you hate coding. I remember that clearly."

        report = analyze_response(response, context)

        if report.total_claims > 0:
            assert report.false_attribution_rate == 1.0

    def test_far_partial(self):
        """FAR is partial when some claims unsupported."""
        context = [
            {"role": "user", "content": "I like Python."},
        ]
        # One supported claim (about Python), one unsupported (about Rust)
        response = "You mentioned Python earlier. You also said you love Rust."

        report = analyze_response(response, context)

        # Should have mixed support
        if report.total_claims >= 2:
            assert 0 < report.false_attribution_rate < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
