"""
Tests for News Provider (NPR RSS implementation).
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

from episodic.utility.providers.news import NewsProvider, CATEGORY_ALIASES, NPR_FEEDS
from episodic.utility.providers.base import ProviderResult


class TestCategoryAliases:
    """Test category alias mapping."""

    def test_tech_alias(self):
        assert CATEGORY_ALIASES["tech"] == "technology"

    def test_sci_alias(self):
        assert CATEGORY_ALIASES["sci"] == "science"

    def test_biz_alias(self):
        assert CATEGORY_ALIASES["biz"] == "business"

    def test_no_ent_alias(self):
        """RSS implementation doesn't have entertainment alias."""
        assert "ent" not in CATEGORY_ALIASES


class TestNewsProvider:
    """Test NewsProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a configured provider."""
        p = NewsProvider()
        p.configure({})  # No API key needed for RSS
        return p

    def test_configure(self, provider):
        """Test configuration."""
        # RSS doesn't use API key
        assert provider._default_count == 5
        assert provider._voice_count == 3

    def test_status(self, provider):
        """Test status method."""
        status = provider.status()
        assert status["name"] == "news"
        assert status["configured"] is True
        # RSS implementation doesn't have "country" in status
        assert "categories" in status
        assert "general" in status["categories"]

    def test_unknown_category(self, provider):
        """Test error for unknown category."""
        result = provider.get("news_headlines", {"category": "nonexistent"})
        assert result.status == "error"
        assert "Unknown news category" in result.payload.get("error", "")

    @patch("urllib.request.urlopen")
    def test_fetch_headlines_from_rss(self, mock_urlopen, provider):
        """Test fetching headlines from RSS feed."""
        # Create sample RSS XML
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Headline 1</title>
                    <description>Test description 1</description>
                    <link>https://example.com/1</link>
                    <pubDate>Wed, 01 Jan 2025 12:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Test Headline 2</title>
                    <description>Test description 2</description>
                    <link>https://example.com/2</link>
                    <pubDate>Wed, 01 Jan 2025 11:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.read.return_value = rss_xml.encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general", "count": 5})

        assert result.status == "ok"
        assert len(result.payload["headlines"]) == 2
        assert result.payload["headlines"][0]["title"] == "Test Headline 1"

    @patch("urllib.request.urlopen")
    def test_fetch_with_category_alias(self, mock_urlopen, provider):
        """Test fetching with category alias."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Tech News</title>
                    <description>Tech description</description>
                    <link>https://example.com/tech</link>
                </item>
            </channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.read.return_value = rss_xml.encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Use alias "tech" instead of "technology"
        result = provider.get("news_headlines", {"category": "tech"})

        assert result.status == "ok"
        assert mock_urlopen.called

    @patch("urllib.request.urlopen")
    def test_cache_hit(self, mock_urlopen, provider):
        """Test that cached results are returned."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Cached Headline</title>
                    <description>Cached desc</description>
                    <link>https://example.com</link>
                </item>
            </channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.read.return_value = rss_xml.encode()
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
        # Create RSS with 10 items
        items = "\n".join([
            f"""<item>
                <title>Headline {i}</title>
                <description>Description {i}</description>
                <link>https://example.com/{i}</link>
            </item>""" for i in range(10)
        ])
        rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>{items}</channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.read.return_value = rss_xml.encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general", "count": 3})

        assert result.status == "ok"
        assert len(result.payload["headlines"]) == 3


class TestNewsProviderErrors:
    """Test error handling."""

    @pytest.fixture
    def provider(self):
        p = NewsProvider()
        p.configure({})
        return p

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen, provider):
        """Test handling of network error."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "Network error" in result.payload.get("error", "")

    @patch("urllib.request.urlopen")
    def test_http_error(self, mock_urlopen, provider):
        """Test handling of HTTP error."""
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(None, 500, "Server Error", {}, None)

        result = provider.get("news_headlines", {})
        assert result.status == "error"
        assert "HTTP error" in result.payload.get("error", "")

    @patch("urllib.request.urlopen")
    def test_no_headlines_found(self, mock_urlopen, provider):
        """Test handling of empty results."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel></channel>
        </rss>"""

        mock_response = MagicMock()
        mock_response.read.return_value = rss_xml.encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general"})
        assert result.status == "error"
        assert "No headlines" in result.payload.get("error", "")

    @patch("urllib.request.urlopen")
    def test_malformed_rss(self, mock_urlopen, provider):
        """Test handling of malformed RSS."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<not>valid<xml"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = provider.get("news_headlines", {"category": "general"})
        assert result.status == "error"


class TestNewsSpeechAndDisplay:
    """Test speech and display text generation."""

    @pytest.fixture
    def provider(self):
        p = NewsProvider()
        p.configure({})
        return p

    def test_speech_text_format(self, provider):
        """Test speech text formatting (no source prefix in NPR RSS version)."""
        headlines = [
            {"title": "First Story", "description": "First desc"},
            {"title": "Second Story", "description": "Second desc"},
        ]
        speech = provider._build_speech(headlines)

        assert "Here are today's headlines" in speech
        # NPR RSS version uses ordinal + title format without "from NPR:"
        assert "First:" in speech
        assert "First Story" in speech
        assert "Second:" in speech
        assert "Second Story" in speech

    def test_display_text_format(self, provider):
        """Test display text formatting (no em-dash, no description line)."""
        headlines = [
            {"title": "First Story", "author": "John Doe", "description": "First desc"},
            {"title": "Second Story", "author": "", "description": "Second desc"},
        ]
        display = provider._build_display(headlines)

        assert "Headlines" in display
        assert "1. First Story" in display
        assert "2. Second Story" in display
        # Authors not shown in display
        assert "(John Doe)" not in display

    def test_display_text_with_category(self, provider):
        """Test display text includes category in header."""
        headlines = [
            {"title": "Tech Story", "author": "", "description": "Tech desc"},
        ]
        display = provider._build_display(headlines, "technology")

        assert "Technology Headlines" in display
