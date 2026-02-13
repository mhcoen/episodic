"""Tests for Muse web-context search execution path."""

from unittest.mock import patch

from episodic.context_builder import ContextBuilder


class _StubSearchManager:
    def __init__(self):
        self.sync_called = False
        self.async_called = False

    def search(self, query: str):
        self.sync_called = True
        raise RuntimeError("sync search should not be called in muse path")

    async def search_async(self, query: str):
        self.async_called = True
        return []

    def get_last_search_diagnostics(self):
        return {
            "providers_attempted": [
                {"provider": "DuckDuckGo", "status": "error", "reason": "loop conflict"}
            ]
        }


def test_muse_context_builder_uses_async_search():
    builder = ContextBuilder()
    stub = _StubSearchManager()

    with patch("episodic.context_builder.config.get", side_effect=lambda k, d=None: {
        "web_search_enabled": True,
        "debug": False,
    }.get(k, d)), patch("episodic.web_search.get_web_search_manager", return_value=stub):
        result = builder._add_web_context("weekend in madison", "gpt-4o-mini")

    assert result is None
    assert stub.async_called is True
    assert stub.sync_called is False
    assert builder.web_error_info is not None
    assert builder.web_error_info["reason"] == "no_results"
