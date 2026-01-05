#!/usr/bin/env python3
"""
Integration tests for web search functionality.
"""

import unittest
import json
import time
import types
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from episodic.config import config
from episodic.web_search import (
    SearchResult, SearchCache, RateLimiter, 
    DuckDuckGoProvider, WebSearchManager, get_web_search_manager
)


class TestSearchCache(unittest.TestCase):
    """Test search result caching."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = SearchCache()
    
    def test_cache_store_and_retrieve(self):
        """Test storing and retrieving from cache."""
        query = "test query"
        results = [
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Test snippet",
                timestamp=datetime.now()
            )
        ]
        
        # Store in cache
        self.cache.set(query, results)
        
        # Retrieve from cache
        cached = self.cache.get(query)
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0].title, "Test Result")
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = SearchCache()
        
        query = "test"
        results = [SearchResult("Test", "url", "Snippet", datetime.now())]
        
        # Store and retrieve immediately
        cache.set(query, results)
        self.assertIsNotNone(cache.get(query, max_age_seconds=1))
        
        # Wait for expiration
        time.sleep(1.5)
        self.assertIsNone(cache.get(query, max_age_seconds=1))
    
    def test_cache_clearing(self):
        """Test clearing cache."""
        # Add multiple entries
        self.cache.set("query1", [SearchResult("R1", "U1", "S1", datetime.now())])
        self.cache.set("query2", [SearchResult("R2", "U2", "S2", datetime.now())])
        
        # Clear cache
        self.cache.clear()
        
        # Verify all cleared
        self.assertIsNone(self.cache.get("query1"))
        self.assertIsNone(self.cache.get("query2"))
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        # Perform some operations
        self.cache.set("q1", [SearchResult("R1", "U1", "S1", datetime.now())])
        self.cache.get("q1")
        self.cache.get("q2")
        
        stats = self.cache.stats()
        
        self.assertEqual(stats['entries'], 1)
        self.assertIn("q1", stats['queries'])


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting."""
    
    def test_rate_limiting(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(max_per_hour=2)
        
        # First two should be allowed
        self.assertTrue(limiter.can_search())
        limiter.record_search()
        self.assertTrue(limiter.can_search())
        limiter.record_search()
        
        # Third should be blocked
        self.assertFalse(limiter.can_search())
    
    def test_rate_limit_reset(self):
        """Test rate limit reset over time."""
        # Use very short window for testing
        limiter = RateLimiter(max_per_hour=1)
        
        # Use up the limit
        self.assertTrue(limiter.can_search())
        limiter.record_search()
        self.assertFalse(limiter.can_search())
        
        limiter.searches = [datetime.now() - timedelta(hours=2)]
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        self.assertTrue(limiter.can_search())
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        limiter = RateLimiter(max_per_hour=1)
        limiter.record_search()
        self.assertEqual(limiter.remaining(), 0)


class TestDuckDuckGoProvider(unittest.IsolatedAsyncioTestCase):
    """Test DuckDuckGo search provider."""
    
    async def test_search_parsing(self):
        """Test parsing of search results."""
        fake_ddgs = types.SimpleNamespace()
        fake_ddgs.DDGS = MagicMock()
        fake_ddgs.DDGS.return_value.text.return_value = [
            {
                "title": "Test Result",
                "href": "https://example.com",
                "body": "Test snippet content"
            }
        ]
        
        # Create provider and search
        provider = DuckDuckGoProvider()
        with patch.dict("sys.modules", {"ddgs": fake_ddgs}):
            results = await provider.search("test query", num_results=1)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Test Result")
        self.assertIn("snippet", results[0].snippet)
        self.assertEqual(results[0].url, "https://example.com")
    
    async def test_error_handling(self):
        """Test error handling in search."""
        fake_ddgs = types.SimpleNamespace()
        fake_ddgs.DDGS = MagicMock()
        fake_ddgs.DDGS.return_value.text.side_effect = Exception("boom")
        
        # Should return empty results on error
        provider = DuckDuckGoProvider()
        with patch.dict("sys.modules", {"ddgs": fake_ddgs}):
            results = await provider.search("test")
        self.assertEqual(len(results), 0)


class TestWebSearchManager(unittest.TestCase):
    """Test web search manager integration."""
    
    def setUp(self):
        """Set up test environment."""
        config.set('web_search_enabled', True)
        config.set('web_search_providers', ['duckduckgo'])
        self.manager = WebSearchManager()
    
    @patch.object(DuckDuckGoProvider, 'search')
    @patch.object(DuckDuckGoProvider, 'is_available', return_value=True)
    def test_search_with_caching(self, _mock_available, mock_search):
        """Test search with caching."""
        # Mock provider search
        async def mock_search_impl(query, num_results):
            return [
                SearchResult("Result 1", "url1", "Snippet 1", datetime.now()),
                SearchResult("Result 2", "url2", "Snippet 2", datetime.now())
            ]
        mock_search.side_effect = mock_search_impl
        
        # First search - should hit provider
        results1 = self.manager.search("test query")
        self.assertEqual(len(results1), 2)
        self.assertEqual(mock_search.call_count, 1)
        
        # Second search - should hit cache
        results2 = self.manager.search("test query")
        self.assertEqual(len(results2), 2)
        self.assertEqual(mock_search.call_count, 1)  # No additional call
    
    @patch.object(DuckDuckGoProvider, 'search')
    @patch.object(DuckDuckGoProvider, 'is_available', return_value=True)
    def test_rate_limiting(self, _mock_available, mock_search):
        """Test rate limiting."""
        # Configure strict rate limit
        config.set('web_search_rate_limit', 2)  # 2 per hour
        manager = WebSearchManager()
        
        # Mock provider
        async def mock_search_impl(query, num_results):
            return [SearchResult("Result", "url", "Snippet", datetime.now())]
        mock_search.side_effect = mock_search_impl
        
        # First two searches should succeed
        results1 = manager.search("query1")
        results2 = manager.search("query2")
        self.assertIsNotNone(results1)
        self.assertIsNotNone(results2)
        
        # Third should be rate limited
        results3 = manager.search("query3")
        self.assertEqual(len(results3), 0)
    
    def test_statistics(self):
        """Test search statistics."""
        stats = self.manager.get_stats()
        
        self.assertIn('providers', stats)
        self.assertIn('rate_limit_remaining', stats)
        self.assertIn('cache', stats)


class TestWebSearchCommands(unittest.TestCase):
    """Test web search CLI commands."""
    
    def setUp(self):
        """Set up test environment."""
        config.set('web_search_enabled', True)
        
        # Mock the search manager
        self.mock_manager = MagicMock()
        self.patcher = patch('episodic.commands.web_search.get_web_search_manager')
        self.mock_get_manager = self.patcher.start()
        self.mock_get_manager.return_value = self.mock_manager
    
    def tearDown(self):
        """Clean up."""
        self.patcher.stop()
    
    def test_web_command(self):
        """Test /web command."""
        from episodic.commands.web_search import websearch
        
        # Mock search results
        self.mock_manager.search.return_value = [
            SearchResult("Test Title", "https://test.com", "Test snippet", datetime.now())
        ]
        
        # Run search
        websearch("test query")
        
        # Verify search was called
        self.mock_manager.search.assert_called_once_with("test query", num_results=5)
    
    def test_web_toggle(self):
        """Test /web on/off."""
        from episodic.commands.web_search import websearch_toggle
        
        # Test enabling
        websearch_toggle(True)
        self.assertTrue(config.get('web_search_enabled'))
        
        # Test disabling  
        websearch_toggle(False)
        self.assertFalse(config.get('web_search_enabled'))
    
    def test_web_stats(self):
        """Test /web stats."""
        from episodic.commands.web_search import websearch_stats
        
        # Mock stats
        self.mock_manager.get_stats.return_value = {
            'providers': ['DuckDuckGo'],
            'current_provider': None,
            'cache': {'entries': 0, 'queries': []},
            'rate_limit_remaining': 10,
            'rate_limit_max': 10
        }
        
        # Run stats command
        websearch_stats()
        
        # Verify stats were retrieved
        self.mock_manager.get_stats.assert_called_once()


class TestRAGWebSearchIntegration(unittest.TestCase):
    """Test integration between RAG and web search."""
    
    def setUp(self):
        """Set up test environment."""
        config.set('rag_enabled', True)
        config.set('web_search_enabled', True)
        config.set('web_search_auto_enhance', True)
    
    @patch('episodic.rag.get_rag_system')
    @patch('episodic.web_search.WebSearchManager')
    def test_auto_web_enhancement(self, mock_get_web, mock_get_rag):
        """Test automatic web search when RAG has no results."""
        from episodic.rag import EpisodicRAG
        
        # Mock RAG with no results
        mock_rag = MagicMock(spec=EpisodicRAG)
        mock_rag.search.return_value = {
            'query': 'test query',
            'results': [],
            'total': 0
        }
        mock_get_rag.return_value = mock_rag
        
        # Mock web search with results
        mock_web = MagicMock()
        mock_web.search.return_value = [
            SearchResult("Web Result", "https://web.com", "Web snippet", datetime.now())
        ]
        mock_get_web.return_value = mock_web
        
        # Mock the _should_search_web method to return True
        with patch.object(EpisodicRAG, '_should_search_web', return_value=True):
            with patch('typer.echo'):
                # Enhance message
                rag = mock_get_rag()
                rag.enhance_with_context = EpisodicRAG.enhance_with_context.__get__(rag)
                enhanced = rag.enhance_with_context("test query")
                
                # Should have called web search
                mock_get_web.assert_called()
                mock_web.search.assert_called()


if __name__ == '__main__':
    unittest.main()
