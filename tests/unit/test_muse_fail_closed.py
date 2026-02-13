"""Tests for Muse fail-closed behavior when web search has no results."""

from unittest.mock import patch

from episodic.conversation_pipeline import TurnContext
from episodic.conversation_pipeline_llm import phase_llm_query


class _DummyManager:
    pass


def test_muse_no_web_context_fails_closed_without_llm():
    ctx = TurnContext(
        user_input="latest ai news",
        model="gpt-4o-mini",
        web_context=None,
        context_debug={
            "web_search_error": {
                "summary": "Web search returned no results from configured providers.",
                "details": ["DuckDuckGo: error (Cannot run the event loop while another loop is running)"],
            }
        },
    )

    with patch("episodic.config.config.get", side_effect=lambda k, d=None: True if k == "muse_mode" else d), \
         patch("episodic.conversation_pipeline_llm._phase_llm_regular") as regular_mock, \
         patch("episodic.conversation_pipeline_llm._phase_llm_muse") as muse_mock:
        phase_llm_query(_DummyManager(), ctx)

    assert ctx.early_return is True
    assert ctx.early_return_value == (None, None)
    assert not regular_mock.called
    assert not muse_mock.called


def test_muse_with_web_context_uses_muse_path():
    ctx = TurnContext(
        user_input="latest ai news",
        model="gpt-4o-mini",
        web_context={"results": [{"title": "x", "url": "u", "content": "c"}], "extracted_content": {}},
    )

    with patch("episodic.config.config.get", side_effect=lambda k, d=None: True if k == "muse_mode" else d), \
         patch("episodic.conversation_pipeline_llm._phase_llm_muse") as muse_mock, \
         patch("episodic.conversation_pipeline_llm._phase_llm_regular") as regular_mock:
        phase_llm_query(_DummyManager(), ctx)

    assert muse_mock.called
    assert not regular_mock.called
