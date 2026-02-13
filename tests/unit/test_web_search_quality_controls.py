"""Tests for Muse web-search quality controls and cache behavior."""

from episodic.web_search import SearchResult, WebSearchManager


class _DummyProvider:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def is_available(self):
        return True

    async def search(self, query, num_results=5):
        self.calls += 1
        return self._results[:num_results]


def _patch_config_get(monkeypatch, **overrides):
    from episodic.config import config

    def fake_get(key, default=None):
        if key in overrides:
            return overrides[key]
        return default

    monkeypatch.setattr(config, "get", fake_get)


def test_time_sensitive_detection():
    assert WebSearchManager._is_time_sensitive_query(
        "What is there to do this weekend in San Francisco?"
    )
    assert not WebSearchManager._is_time_sensitive_query(
        "Explain the history of transistors"
    )


def test_quality_controls_filter_low_signal_domains_for_event_queries(monkeypatch):
    _patch_config_get(
        monkeypatch,
        web_search_providers=["duckduckgo"],
        web_search_excluded_domains=[],
        web_search_time_sensitive_excluded_domains=["tiktok.com", "pinterest.com"],
    )

    manager = WebSearchManager()
    raw = [
        SearchResult(title="TikTok", url="https://www.tiktok.com/discover/x", snippet="x"),
        SearchResult(title="Pinterest", url="https://www.pinterest.com/x", snippet="x"),
        SearchResult(
            title="Eventbrite this weekend",
            url="https://www.eventbrite.com/d/ca--san-francisco/events--this-weekend/",
            snippet="events this weekend in San Francisco",
        ),
    ]

    filtered = manager._apply_result_quality_controls(
        "What is there to do this weekend in San Francisco?", raw
    )
    urls = [r.url for r in filtered]
    assert "https://www.tiktok.com/discover/x" not in urls
    assert "https://www.pinterest.com/x" not in urls
    assert urls[0].startswith("https://www.eventbrite.com/")


def test_search_bypasses_cache_for_time_sensitive_queries(monkeypatch):
    _patch_config_get(
        monkeypatch,
        web_search_providers=["duckduckgo"],
        web_search_max_results=5,
        web_search_rate_limit=60,
        web_search_fallback_enabled=True,
        web_search_fallback_cache_minutes=5,
        web_search_cache_duration=3600,
        web_search_bypass_cache_for_time_sensitive=True,
        web_search_excluded_domains=[],
        web_search_time_sensitive_excluded_domains=[],
        debug=False,
    )

    manager = WebSearchManager()
    query = "What is there to do this weekend in San Francisco?"
    cached = [SearchResult(title="Cached", url="https://cached.example", snippet="cached")]
    fresh = [SearchResult(title="Fresh", url="https://fresh.example/events", snippet="today events")]
    manager.cache.set(query, cached)
    provider = _DummyProvider(fresh)
    manager.providers = [provider]

    out = manager.search(query, num_results=5, use_cache=True)
    assert provider.calls == 1
    assert out[0].url == "https://fresh.example/events"


def test_search_uses_cache_for_non_time_sensitive_queries(monkeypatch):
    _patch_config_get(
        monkeypatch,
        web_search_providers=["duckduckgo"],
        web_search_max_results=5,
        web_search_rate_limit=60,
        web_search_fallback_enabled=True,
        web_search_fallback_cache_minutes=5,
        web_search_cache_duration=3600,
        web_search_bypass_cache_for_time_sensitive=True,
        web_search_excluded_domains=[],
        web_search_time_sensitive_excluded_domains=[],
        debug=False,
    )

    manager = WebSearchManager()
    query = "history of the transistor"
    cached = [SearchResult(title="Cached", url="https://cached.example", snippet="cached")]
    manager.cache.set(query, cached)
    provider = _DummyProvider([SearchResult(title="Fresh", url="https://fresh.example", snippet="fresh")])
    manager.providers = [provider]

    out = manager.search(query, num_results=5, use_cache=True)
    assert provider.calls == 0
    assert out[0].url == "https://cached.example"
