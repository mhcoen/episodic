"""
Tests for News Provider.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json

from episodic.utility.providers.news import NewsProvider, CATEGORY_ALIASES
from episodic.utility.providers.base import ProviderResult


class TestCategoryAliases:
    """Test category alias mapping."""

    def test_tech_alias(self):
        assert CATEGORY_ALIASES["tech"] == "technology"

    def test_sci_alias(self):
        assert CATEGORY_ALIASES["sci"] == "science"

    def test_biz_alias(self):
        assert CATEGORY_ALIASES["biz"] == "business"

    def test_ent_alias(self):
        assert CATEGORY_ALIASES["ent"] == "entertainment"


class TestNewsProvider:
    """Test NewsProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a configured provider."""
        p = NewsProvider()
        p.configure({"api_key": "test_api_key"})
        return p

    def test_configure(self, provider):
        """Test configuration."""
        assert provider._api_key == "test_api_key"
        assert provider._country == "us"
        assert provider._default_count == 5

    def test_status(self, provider):
        """Test status method."""
        status = provider.status()
        assert status["name"] == "news"
        assert status["configured"] is True
        assert status["country"] == "us"

    def test_no_api_key(self):
        """Test error when no API key configured."""
        provider = NewsProvider()
        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "NEWSAPI_KEY" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_fetch_headlines(self, mock_urlopen, provider):
        """Test fetching headlines."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "NPR"},
                    "title": "Test Headline 1",
                    "description": "Test description 1",
                    "url": "https://example.com/1",
                },
                {
                    "source": {"name": "BBC"},
                    "title": "Test Headline 2",
                    "description": "Test description 2",
                    "url": "https://example.com/2",
                },
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general", "count": 5})

        assert result.status == "ok"
        assert len(result.payload["headlines"]) == 2
        assert result.payload["headlines"][0]["source"] == "NPR"
        assert "📰" in result.display_text
        assert "NPR" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_fetch_with_category_alias(self, mock_urlopen, provider):
        """Test fetching with category alias."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "TechCrunch"},
                    "title": "Tech News",
                    "description": "Tech description",
                    "url": "https://example.com/tech",
                },
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Use alias "tech" instead of "technology"
        result = provider.get("news_headlines", {"category": "tech"})

        assert result.status == "ok"
        # Verify the request was made (alias is resolved internally)
        assert mock_urlopen.called

    @patch("urllib.request.urlopen")
    def test_cache_hit(self, mock_urlopen, provider):
        """Test that cached results are returned."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Test"},
                    "title": "Cached Headline",
                    "description": "Cached desc",
                    "url": "https://example.com",
                },
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # First call - should fetch
        result1 = provider.get("news_headlines", {"category": "general"})
        assert mock_urlopen.call_count == 1

        # Second call - should use cache
        result2 = provider.get("news_headlines", {"category": "general"})
        assert mock_urlopen.call_count == 1  # Not called again
        assert result2.status == "ok"

    @patch("urllib.request.urlopen")
    def test_count_limiting(self, mock_urlopen, provider):
        """Test that count parameter limits results."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [
                {"source": {"name": f"Source{i}"}, "title": f"Headline {i}", "description": "", "url": ""}
                for i in range(10)
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general", "count": 3})

        assert result.status == "ok"
        assert len(result.payload["headlines"]) == 3

    @patch("urllib.request.urlopen")
    def test_filters_removed_articles(self, mock_urlopen, provider):
        """Test that [Removed] articles are filtered out."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [
                {"source": {"name": "Good"}, "title": "Good Article", "description": "", "url": ""},
                {"source": {"name": "Bad"}, "title": "[Removed]", "description": "", "url": ""},
                {"source": {"name": "Also Good"}, "title": "Another Good Article", "description": "", "url": ""},
            ],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {})

        assert result.status == "ok"
        assert len(result.payload["headlines"]) == 2
        assert all(h["title"] != "[Removed]" for h in result.payload["headlines"])


class TestNewsProviderErrors:
    """Test error handling."""

    @pytest.fixture
    def provider(self):
        p = NewsProvider()
        p.configure({"api_key": "test_key"})
        return p

    @patch("urllib.request.urlopen")
    def test_invalid_api_key(self, mock_urlopen, provider):
        """Test handling of invalid API key."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 401, "Unauthorized", {}, None)

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "Invalid API key" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_rate_limited(self, mock_urlopen, provider):
        """Test handling of rate limit."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 429, "Too Many Requests", {}, None)

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "Rate limit" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen, provider):
        """Test handling of network error."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "Network error" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_api_error_response(self, mock_urlopen, provider):
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "error",
            "message": "API key exhausted",
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "exhausted" in result.speech_text

    @patch("urllib.request.urlopen")
    def test_no_headlines_found(self, mock_urlopen, provider):
        """Test handling of empty results."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "ok",
            "articles": [],
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "obscure"})
        assert result.status == "error"
        assert "No headlines" in result.speech_text


class TestNewsSpeechAndDisplay:
    """Test speech and display text generation."""

    @pytest.fixture
    def provider(self):
        p = NewsProvider()
        p.configure({"api_key": "test_key"})
        return p

    def test_speech_text_format(self, provider):
        """Test speech text formatting."""
        headlines = [
            {"title": "First Story", "source": "NPR", "description": "First desc"},
            {"title": "Second Story", "source": "BBC", "description": "Second desc"},
        ]
        speech = provider._build_speech(headlines)

        assert "Here are today's headlines" in speech
        assert "First, from NPR: First Story" in speech
        assert "Second, from BBC: Second Story" in speech

    def test_display_text_format(self, provider):
        """Test display text formatting."""
        headlines = [
            {"title": "First Story", "source": "NPR", "description": "First desc"},
            {"title": "Second Story", "source": "BBC", "description": "Second desc"},
        ]
        display = provider._build_display(headlines)

        assert "📰 Top Headlines" in display
        assert "1. First Story — NPR" in display
        assert "2. Second Story — BBC" in display
        assert "First desc" in display

    def test_long_description_truncation(self, provider):
        """Test that long descriptions are truncated."""
        long_desc = "x" * 200
        headlines = [
            {"title": "Story", "source": "Test", "description": long_desc},
        ]
        display = provider._build_display(headlines)

        # Should be truncated with ...
        assert "..." in display
        assert len(display.split("\n")[2]) < 150  # Line should be reasonable length
