#!/usr/bin/env python3
"""
Integration tests for web search functionality.
"""

import unittest
import json
import time
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
                snippet="Test snippet",
                url="https://example.com",
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
        # Create cache with short TTL
        cache = SearchCache(default_ttl=1)  # 1 second
        
        query = "test"
        results = [SearchResult("Test", "Snippet", "url", datetime.now())]
        
        # Store and retrieve immediately
        cache.set(query, results)
        self.assertIsNotNone(cache.get(query))
        
        # Wait for expiration
        time.sleep(1.5)
        self.assertIsNone(cache.get(query))
    
    def test_cache_clearing(self):
        """Test clearing cache."""
        # Add multiple entries
        self.cache.set("query1", [SearchResult("R1", "S1", "U1", datetime.now())])
        self.cache.set("query2", [SearchResult("R2", "S2", "U2", datetime.now())])
        
        # Clear cache
        self.cache.clear()
        
        # Verify all cleared
        self.assertIsNone(self.cache.get("query1"))
        self.assertIsNone(self.cache.get("query2"))
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        # Perform some operations
        self.cache.set("q1", [SearchResult("R1", "S1", "U1", datetime.now())])
        self.cache.get("q1")  # Hit
        self.cache.get("q2")  # Miss
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5, places=2)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting."""
    
    def test_rate_limiting(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(max_per_minute=2)
        
        # First two should be allowed
        self.assertTrue(limiter.check_rate_limit())
        self.assertTrue(limiter.check_rate_limit())
        
        # Third should be blocked
        self.assertFalse(limiter.check_rate_limit())
    
    def test_rate_limit_reset(self):
        """Test rate limit reset over time."""
        # Use very short window for testing
        limiter = RateLimiter(max_per_minute=1)
        limiter.window_minutes = 1/60  # 1 second window
        
        # Use up the limit
        self.assertTrue(limiter.check_rate_limit())
        self.assertFalse(limiter.check_rate_limit())
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        self.assertTrue(limiter.check_rate_limit())
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        limiter = RateLimiter(max_per_minute=1)
        
        # Use up limit
        limiter.record_request()
        
        # Get wait time
        wait_time = limiter.get_wait_time()
        self.assertGreater(wait_time, 0)
        self.assertLessEqual(wait_time, 60)


class TestDuckDuckGoProvider(unittest.TestCase):
    """Test DuckDuckGo search provider."""
    
    @patch('aiohttp.ClientSession')
    async def test_search_parsing(self, mock_session_class):
        """Test parsing of search results."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='''
            <html>
            <div class="result">
                <h2 class="result__title">
                    <a href="https://example.com">Test Result</a>
                </h2>
                <a class="result__snippet">Test snippet content</a>
            </div>
            </html>
        ''')
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Create provider and search
        provider = DuckDuckGoProvider()
        results = await provider.search("test query", num_results=1)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Test Result")
        self.assertIn("snippet", results[0].snippet)
        self.assertEqual(results[0].url, "https://example.com")
    
    @patch('aiohttp.ClientSession')
    async def test_error_handling(self, mock_session_class):
        """Test error handling in search."""
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Should return empty results on error
        provider = DuckDuckGoProvider()
        results = await provider.search("test")
        self.assertEqual(len(results), 0)


class TestWebSearchManager(unittest.TestCase):
    """Test web search manager integration."""
    
    def setUp(self):
        """Set up test environment."""
        config.set('web_search_enabled', True)
        config.set('web_search_provider', 'duckduckgo')
        self.manager = WebSearchManager()
    
    @patch.object(DuckDuckGoProvider, 'search')
    def test_search_with_caching(self, mock_search):
        """Test search with caching."""
        # Mock provider search
        async def mock_search_impl(query, num_results):
            return [
                SearchResult("Result 1", "Snippet 1", "url1", datetime.now()),
                SearchResult("Result 2", "Snippet 2", "url2", datetime.now())
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
    def test_rate_limiting(self, mock_search):
        """Test rate limiting."""
        # Configure strict rate limit
        config.set('web_search_rate_limit', 2)  # 2 per hour
        manager = WebSearchManager()
        
        # Mock provider
        async def mock_search_impl(query, num_results):
            return [SearchResult("Result", "Snippet", "url", datetime.now())]
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
        
        self.assertIn('provider', stats)
        self.assertIn('cache_stats', stats)
        self.assertIn('total_searches', stats)
        self.assertEqual(stats['provider'], 'duckduckgo')


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
            SearchResult("Test Title", "Test snippet", "https://test.com", datetime.now())
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
            'provider': 'duckduckgo',
            'total_searches': 10,
            'cache_stats': {'hits': 5, 'misses': 5}
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
    @patch('episodic.web_search.get_web_search_manager')
    def test_auto_web_enhancement(self, mock_get_web, mock_get_rag):
        """Test automatic web search when RAG has no results."""
        from episodic.rag import EpisodicRAG
        
        # Mock RAG with no results
        mock_rag = MagicMock(spec=EpisodicRAG)
        mock_rag.search.return_value = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        mock_get_rag.return_value = mock_rag
        
        # Mock web search with results
        mock_web = MagicMock()
        mock_web.search.return_value = [
            SearchResult("Web Result", "Web snippet", "https://web.com", datetime.now())
        ]
        mock_get_web.return_value = mock_web
        
        # Mock the _should_search_web method to return True
        with patch.object(EpisodicRAG, '_should_search_web', return_value=True):
            # Enhance message
            rag = mock_get_rag()
            rag.enhance_with_context = EpisodicRAG.enhance_with_context.__get__(rag)
            enhanced, sources = rag.enhance_with_context("test query")
            
            # Should have called web search
            mock_get_web.assert_called()
            mock_web.search.assert_called()


if __name__ == '__main__':
    unittest.main()