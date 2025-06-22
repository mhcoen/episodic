#!/usr/bin/env python3
"""
Unit tests for LLM integration functionality.

Tests the LLM module, prompt caching, and provider integration.
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


class TestLLMHelpers(unittest.TestCase):
    """Test LLM helper functions."""
    
    def test_get_model_string_local(self):
        """Test model string formatting for local providers."""
        with patch('episodic.llm.get_current_provider', return_value='local'):
            with patch('episodic.llm.get_provider_models') as mock_get_models:
                mock_get_models.return_value = [
                    {"name": "llama2", "backend": "llama.cpp"}
                ]
                
                result = llm.get_model_string("llama2")
                self.assertEqual(result, "llama.cpp/llama2")
    
    def test_get_model_string_ollama(self):
        """Test model string formatting for Ollama."""
        with patch('episodic.llm.get_current_provider', return_value='ollama'):
            result = llm.get_model_string("llama2")
            self.assertEqual(result, "ollama/llama2")
    
    def test_get_model_string_cloud_provider(self):
        """Test model string formatting for cloud providers."""
        with patch('episodic.llm.get_current_provider', return_value='openai'):
            # Test without provider prefix
            result = llm.get_model_string("gpt-4")
            self.assertEqual(result, "openai/gpt-4")
            
            # Test with provider prefix already included
            result = llm.get_model_string("openai/gpt-4")
            self.assertEqual(result, "openai/gpt-4")
    
    def test_get_model_string_lmstudio(self):
        """Test model string formatting for LMStudio."""
        with patch('episodic.llm.get_current_provider', return_value='lmstudio'):
            result = llm.get_model_string("local-model")
            self.assertEqual(result, "local-model")


class TestPromptCaching(unittest.TestCase):
    """Test prompt caching functionality."""
    
    def test_apply_prompt_caching_unsupported_model(self):
        """Test prompt caching with unsupported model."""
        with patch('episodic.llm.supports_prompt_caching', return_value=False):
            messages = [{"role": "system", "content": "Test prompt"}]
            result = llm._apply_prompt_caching(messages, "unsupported-model")
            self.assertEqual(result, messages)  # Should return unchanged
    
    def test_apply_prompt_caching_openai_model(self):
        """Test prompt caching with OpenAI model (should not apply cache_control)."""
        with patch('episodic.llm.supports_prompt_caching', return_value=True):
            messages = [{"role": "system", "content": "Test prompt"}]
            result = llm._apply_prompt_caching(messages, "openai/gpt-4")
            self.assertEqual(result, messages)  # Should return unchanged for OpenAI
    
    def test_apply_prompt_caching_anthropic_model(self):
        """Test prompt caching with Anthropic model."""
        with patch('episodic.llm.supports_prompt_caching', return_value=True):
            messages = [
                {"role": "system", "content": "Test system prompt"},
                {"role": "user", "content": "Test user message"}
            ]
            result = llm._apply_prompt_caching(messages, "anthropic/claude-3-sonnet")
            
            # Check that system message has cache_control
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["role"], "system")
            self.assertIsInstance(result[0]["content"], list)
            self.assertEqual(result[0]["content"][0]["type"], "text")
            self.assertEqual(result[0]["content"][0]["text"], "Test system prompt")
            self.assertEqual(result[0]["content"][0]["cache_control"], {"type": "ephemeral"})
            
            # Check that user message is unchanged
            self.assertEqual(result[1], {"role": "user", "content": "Test user message"})
    
    def test_apply_prompt_caching_exception_handling(self):
        """Test prompt caching exception handling."""
        with patch('episodic.llm.supports_prompt_caching', side_effect=Exception("Test error")):
            messages = [{"role": "system", "content": "Test prompt"}]
            result = llm._apply_prompt_caching(messages, "test-model")
            self.assertEqual(result, messages)  # Should return original on error


class TestCacheManagement(unittest.TestCase):
    """Test cache management functions."""
    
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.config')
    def test_initialize_cache_enabled(self, mock_config, mock_litellm):
        """Test cache initialization when enabled."""
        mock_config.get.return_value = True
        
        result = llm.initialize_cache()
        
        self.assertTrue(result)
        # Verify response caching is disabled
        self.assertIsNone(mock_litellm.cache)
    
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.config')
    def test_initialize_cache_disabled(self, mock_config, mock_litellm):
        """Test cache initialization when disabled."""
        mock_config.get.return_value = False
        
        result = llm.initialize_cache()
        
        self.assertFalse(result)
        self.assertIsNone(mock_litellm.cache)
    
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.config')
    def test_disable_cache(self, mock_config, mock_litellm):
        """Test cache disabling."""
        llm.disable_cache()
        
        self.assertIsNone(mock_litellm.cache)
        mock_config.set.assert_called_with("use_context_cache", False)
    
    @patch('episodic.llm.initialize_cache')
    @patch('episodic.llm.config')
    def test_enable_cache(self, mock_config, mock_initialize):
        """Test cache enabling."""
        mock_initialize.return_value = True
        
        result = llm.enable_cache()
        
        self.assertTrue(result)
        mock_config.set.assert_called_with("use_context_cache", True)
        mock_initialize.assert_called_once()


class TestLLMQueries(unittest.TestCase):
    """Test LLM query functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock litellm to avoid actual API calls
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message.content = "Test response"
        self.mock_response.usage.prompt_tokens = 100
        self.mock_response.usage.completion_tokens = 50
        self.mock_response.usage.total_tokens = 150
        
        # Mock prompt_tokens_details for caching tests
        self.mock_response.usage.prompt_tokens_details = Mock()
        self.mock_response.usage.prompt_tokens_details.cached_tokens = 0
    
    @patch('episodic.llm.cost_per_token')
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.get_current_provider')
    @patch('episodic.llm.get_model_string')
    def test_execute_llm_query(self, mock_get_model, mock_get_provider, mock_litellm, mock_cost):
        """Test basic LLM query execution."""
        mock_get_provider.return_value = "openai"
        mock_get_model.return_value = "openai/gpt-4"
        mock_litellm.completion.return_value = self.mock_response
        mock_cost.return_value = [0.005]
        
        messages = [{"role": "user", "content": "Test message"}]
        response, cost_info = llm._execute_llm_query(messages, "gpt-4", 0.7, 1000)
        
        self.assertEqual(response, "Test response")
        self.assertEqual(cost_info["input_tokens"], 100)
        self.assertEqual(cost_info["output_tokens"], 50)
        self.assertEqual(cost_info["total_tokens"], 150)
        self.assertEqual(cost_info["cost_usd"], 0.005)
    
    @patch('episodic.llm.cost_per_token')
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.get_current_provider')
    @patch('episodic.llm.get_model_string')
    def test_execute_llm_query_with_caching(self, mock_get_model, mock_get_provider, mock_litellm, mock_cost):
        """Test LLM query execution with prompt caching."""
        mock_get_provider.return_value = "openai"
        mock_get_model.return_value = "openai/gpt-4"
        mock_litellm.completion.return_value = self.mock_response
        mock_cost.return_value = [0.005]
        
        # Set up cached tokens
        self.mock_response.usage.prompt_tokens_details.cached_tokens = 80
        
        messages = [{"role": "user", "content": "Test message"}]
        response, cost_info = llm._execute_llm_query(messages, "gpt-4", 0.7, 1000)
        
        self.assertEqual(response, "Test response")
        self.assertEqual(cost_info["cached_tokens"], 80)
        self.assertEqual(cost_info["non_cached_tokens"], 20)  # 100 - 80
        self.assertIn("cache_savings_usd", cost_info)
    
    @patch('episodic.llm._execute_llm_query')
    def test_query_llm(self, mock_execute):
        """Test query_llm function."""
        mock_execute.return_value = ("Test response", {"cost": 0.01})
        
        response, cost = llm.query_llm("Test prompt", "gpt-4", "System message")
        
        self.assertEqual(response, "Test response")
        self.assertEqual(cost["cost"], 0.01)
        
        # Verify correct message structure was passed
        call_args = mock_execute.call_args[0]
        messages = call_args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System message")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Test prompt")


class TestProviderSpecificHandling(unittest.TestCase):
    """Test provider-specific LLM handling."""
    
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.get_current_provider')
    @patch('episodic.llm.get_model_string')
    @patch('episodic.llm.get_provider_config')
    def test_lmstudio_provider(self, mock_get_config, mock_get_model, mock_get_provider, mock_litellm):
        """Test LMStudio provider handling."""
        mock_get_provider.return_value = "lmstudio"
        mock_get_model.return_value = "local-model"
        mock_get_config.return_value = {"api_base": "http://localhost:1234/v1"}
        
        # Set up mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = None
        mock_litellm.completion.return_value = mock_response
        
        with patch('episodic.llm.cost_per_token', return_value=[0.005]):
            messages = [{"role": "user", "content": "Test"}]
            response, cost = llm._execute_llm_query(messages, "local-model", 0.7, 1000)
        
        # Verify LMStudio-specific parameters were used
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        self.assertEqual(call_kwargs["api_base"], "http://localhost:1234/v1")
    
    @patch('episodic.llm.litellm')
    @patch('episodic.llm.get_current_provider')
    @patch('episodic.llm.get_model_string')
    @patch('episodic.llm.get_provider_config')
    def test_ollama_provider(self, mock_get_config, mock_get_model, mock_get_provider, mock_litellm):
        """Test Ollama provider handling."""
        mock_get_provider.return_value = "ollama"
        mock_get_model.return_value = "ollama/llama2"
        mock_get_config.return_value = {"api_base": "http://localhost:11434"}
        
        # Set up mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = None
        mock_litellm.completion.return_value = mock_response
        
        with patch('episodic.llm.cost_per_token', return_value=[0.005]):
            messages = [{"role": "user", "content": "Test"}]
            response, cost = llm._execute_llm_query(messages, "llama2", 0.7, 1000)
        
        # Verify Ollama-specific parameters were used
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args[1]
        self.assertEqual(call_kwargs["api_base"], "http://localhost:11434")
        self.assertEqual(call_kwargs["stream"], False)


if __name__ == '__main__':
    unittest.main()