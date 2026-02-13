from unittest.mock import patch

from episodic.context_builder import ContextBuilder
from episodic.web_search_providers.base import SearchResult


class _StubSearchManager:
    def __init__(self, results):
        self._results = results

    async def search_async(self, query: str):
        return self._results

    def get_last_search_diagnostics(self):
        return {}


def _run_with_result_url(url: str):
    cb = ContextBuilder()
    results = [SearchResult(title="t", url=url, snippet="s")]
    stub = _StubSearchManager(results)

    captured_urls = []

    def fake_fetch(u):
        captured_urls.append(u)
        return "x" * 80

    with patch("episodic.web_search.get_web_search_manager", return_value=stub), \
         patch("episodic.web_extract.fetch_page_content_sync", side_effect=fake_fetch), \
         patch("episodic.context_builder.config.get", side_effect=lambda k, d=None: {
             "web_search_enabled": True,
             "web_search_extract_content": True,
             "web_extract_max_pages": 1,
             "web_extract_timeout": 5,
             "debug": False,
         }.get(k, d)):
        ctx = cb._add_web_context("q", "m")

    assert ctx is not None
    assert captured_urls
    return captured_urls[0]


def test_fix_search_url_unwraps_https_duckduckgo_redirect():
    src = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.eventbrite.com%2Fd%2Fca--san-francisco%2Fevents--this-weekend%2F"
    fetched = _run_with_result_url(src)
    assert fetched.startswith("https://www.eventbrite.com/d/ca--san-francisco/events--this-weekend/")


def test_fix_search_url_unwraps_protocol_relative_duckduckgo_redirect():
    src = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fsf.funcheap.com%2Fweekend%2F"
    fetched = _run_with_result_url(src)
    assert fetched.startswith("https://sf.funcheap.com/weekend/")
