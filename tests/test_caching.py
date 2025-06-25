#!/usr/bin/env python3
"""
Tests for prompt caching functionality.

Validates that prompt caching works correctly for different providers.
"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock, Mock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic import llm
from episodic.config import config


class TestPromptCachingImplementation(unittest.TestCase):
    """Test the prompt caching implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original cache setting
        self.original_cache_setting = config.get("use_context_cache")
        
        # Enable caching for tests
        config.set("use_context_cache", True)
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original cache setting
        if self.original_cache_setting is not None:
            config.set("use_context_cache", self.original_cache_setting)
        else:
            config.delete("use_context_cache")
    
    def test_cache_initialization_enabled(self):
        """Test cache initialization when enabled."""
        config.set("use_context_cache", True)
        
        with patch('episodic.llm.litellm') as mock_litellm:
            result = llm.initialize_cache()
            
            # Should return True for success
            self.assertTrue(result)
            
            # Should disable response caching
            self.assertIsNone(mock_litellm.cache)
    
    def test_cache_initialization_disabled(self):
        """Test cache initialization when disabled."""
        config.set("use_context_cache", False)
        
        with patch('episodic.llm.litellm') as mock_litellm:
            result = llm.initialize_cache()
            
            # Should return False
            self.assertFalse(result)
            
            # Should disable all caching
            self.assertIsNone(mock_litellm.cache)
    
    def test_enable_cache_function(self):
        """Test the enable_cache function."""
        with patch('episodic.llm.initialize_cache', return_value=True) as mock_init:
            result = llm.enable_cache()
            
            # Should update config
            self.assertTrue(config.get("use_context_cache"))
            
            # Should call initialize_cache
            mock_init.assert_called_once()
            
            # Should return success
            self.assertTrue(result)
    
    def test_disable_cache_function(self):
        """Test the disable_cache function."""
        # Mock the actual litellm module that disable_cache imports
        with patch('litellm.cache', new=Mock()) as mock_cache:
            llm.disable_cache()
            
            # Should update config
            self.assertFalse(config.get("use_context_cache"))
            
        # After patching, litellm.cache should have been set to None
        # We need to check this by patching at the import level
        import litellm
        with patch.object(litellm, 'cache', Mock()):
            llm.disable_cache()
            # The disable_cache function sets it to None
            self.assertIsNone(litellm.cache)
    
    @patch('episodic.llm.supports_prompt_caching')
    def test_anthropic_cache_control_application(self, mock_supports_caching):
        """Test that cache_control is applied for Anthropic models."""
        mock_supports_caching.return_value = True
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        # Test with Anthropic model
        result = llm._apply_prompt_caching(messages, "anthropic/claude-3-sonnet")
        
        # System message should have cache_control
        self.assertEqual(result[0]["role"], "system")
        self.assertIsInstance(result[0]["content"], list)
        self.assertEqual(result[0]["content"][0]["type"], "text")
        self.assertEqual(result[0]["content"][0]["text"], "You are a helpful assistant.")
        self.assertEqual(result[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        
        # User message should be unchanged
        self.assertEqual(result[1], {"role": "user", "content": "Hello"})
    
    @patch('episodic.llm.supports_prompt_caching')
    def test_openai_no_cache_control(self, mock_supports_caching):
        """Test that cache_control is NOT applied for OpenAI models."""
        mock_supports_caching.return_value = True
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        # Test with OpenAI model
        result = llm._apply_prompt_caching(messages, "openai/gpt-4")
        
        # Messages should be unchanged (OpenAI uses automatic caching)
        self.assertEqual(result, messages)
    
    @patch('episodic.llm.supports_prompt_caching')
    def test_unsupported_model_no_caching(self, mock_supports_caching):
        """Test that unsupported models don't get caching applied."""
        mock_supports_caching.return_value = False
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = llm._apply_prompt_caching(messages, "unsupported/model")
        
        # Messages should be unchanged
        self.assertEqual(result, messages)
    
    def test_cache_control_exception_handling(self):
        """Test that exceptions in cache application are handled gracefully."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        with patch('episodic.llm.supports_prompt_caching', side_effect=Exception("Test error")):
            result = llm._apply_prompt_caching(messages, "test-model")
            
            # Should return original messages on error
            self.assertEqual(result, messages)


class TestCacheMetricsAndCostCalculation(unittest.TestCase):
    """Test cache metrics tracking and cost calculation."""
    
    def setUp(self):
        """Set up mock response for testing."""
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message.content = "Test response"
        self.mock_response.usage.prompt_tokens = 1000
        self.mock_response.usage.completion_tokens = 100
        self.mock_response.usage.total_tokens = 1100
    
    @patch('episodic.llm.cost_per_token')
    def test_cost_calculation_without_caching(self, mock_cost):
        """Test cost calculation when no caching is used."""
        # No cached tokens
        self.mock_response.usage.prompt_tokens_details = Mock()
        self.mock_response.usage.prompt_tokens_details.cached_tokens = 0
        
        mock_cost.return_value = [0.01]
        
        with patch('episodic.llm.get_current_provider', return_value='openai'):
            with patch('episodic.llm.get_model_string', return_value='openai/gpt-4'):
                with patch('episodic.llm.litellm.completion', return_value=self.mock_response):
                    response, cost_info = llm._execute_llm_query(
                        [{"role": "user", "content": "test"}], 
                        "gpt-4", 0.7, 1000
                    )
        
        # Should not have cache-related fields
        self.assertNotIn("cached_tokens", cost_info)
        self.assertNotIn("cache_savings_usd", cost_info)
        self.assertEqual(cost_info["input_tokens"], 1000)
        self.assertEqual(cost_info["output_tokens"], 100)
    
    @patch('episodic.llm.cost_per_token')
    def test_cost_calculation_with_caching(self, mock_cost):
        """Test cost calculation when caching is used."""
        # Simulate cached tokens
        self.mock_response.usage.prompt_tokens_details = Mock()
        self.mock_response.usage.prompt_tokens_details.cached_tokens = 800
        
        # Mock cost_per_token to return consistent values
        # First call: total_cost calculation (full prompt + completion)
        # Second call: actual_cost calculation (non-cached prompt + completion)
        # Third call: cached_cost calculation (cached tokens only)
        mock_cost.side_effect = [
            [0.01],  # total_cost: 1000 prompt + 100 completion = 0.01
            [0.003], # actual_cost: 200 non-cached + 100 completion = 0.003
            [0.008]  # cached_cost: 800 cached tokens = 0.008 (before discount)
        ]
        
        with patch('episodic.llm.get_current_provider', return_value='openai'):
            with patch('episodic.llm.get_model_string', return_value='openai/gpt-4'):
                with patch('episodic.llm.litellm.completion', return_value=self.mock_response):
                    response, cost_info = llm._execute_llm_query(
                        [{"role": "user", "content": "test"}], 
                        "gpt-4", 0.7, 1000
                    )
        
        # Should have cache-related fields
        self.assertEqual(cost_info["cached_tokens"], 800)
        self.assertEqual(cost_info["non_cached_tokens"], 200)  # 1000 - 800
        self.assertIn("cache_savings_usd", cost_info)
        
        # Cost should be lower due to caching
        # total_cost = 0.01, total_cost_with_cache = 0.003 + (0.008 * 0.5) = 0.007
        # savings = 0.01 - 0.007 = 0.003
        self.assertGreater(cost_info["cache_savings_usd"], 0)
        self.assertAlmostEqual(cost_info["cache_savings_usd"], 0.003, places=6)
    
    def test_cache_metrics_without_prompt_tokens_details(self):
        """Test handling when prompt_tokens_details is not available."""
        # No prompt_tokens_details attribute
        self.mock_response.usage.prompt_tokens_details = None
        
        with patch('episodic.llm.cost_per_token', return_value=[0.01]):
            with patch('episodic.llm.get_current_provider', return_value='openai'):
                with patch('episodic.llm.get_model_string', return_value='openai/gpt-4'):
                    with patch('episodic.llm.litellm.completion', return_value=self.mock_response):
                        response, cost_info = llm._execute_llm_query(
                            [{"role": "user", "content": "test"}], 
                            "gpt-4", 0.7, 1000
                        )
        
        # Should handle gracefully without cache fields
        self.assertNotIn("cached_tokens", cost_info)
        self.assertEqual(cost_info["input_tokens"], 1000)


class TestCacheIntegrationWithCLI(unittest.TestCase):
    """Test cache integration with CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_cache_setting = config.get("use_context_cache")
        
        # Import CLI module for testing
        from episodic import cli
        self.cli = cli
    
    def tearDown(self):
        """Clean up test environment."""
        if self.original_cache_setting is not None:
            config.set("use_context_cache", self.original_cache_setting)
        else:
            config.delete("use_context_cache")
    
    @patch('episodic.commands.settings.enable_cache')
    def test_cli_cache_enable_command(self, mock_enable):
        """Test CLI command to enable cache."""
        from episodic.commands.settings import set as cli_set
        cli_set("cache", "on")
        mock_enable.assert_called_once()
    
    @patch('episodic.commands.settings.disable_cache')
    def test_cli_cache_disable_command(self, mock_disable):
        """Test CLI command to disable cache."""
        from episodic.commands.settings import set as cli_set
        cli_set("cache", "off")
        mock_disable.assert_called_once()
    
    def test_cache_setting_persistence(self):
        """Test that cache settings persist in configuration."""
        # Enable cache
        config.set("use_context_cache", True)
        self.assertTrue(config.get("use_context_cache"))
        
        # Disable cache
        config.set("use_context_cache", False)
        self.assertFalse(config.get("use_context_cache"))


if __name__ == '__main__':
    unittest.main()