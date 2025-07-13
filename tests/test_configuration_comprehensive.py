"""
Comprehensive tests for configuration management.

Tests the complete configuration system including:
- Default values and initialization
- Model configurations for different contexts
- Parameter validation
- Persistence and migration
- CLI integration
- Environment variable handling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import json
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from episodic.config import Config
from episodic.config_defaults import (
    CORE_DEFAULTS, TOPIC_DEFAULTS, LLM_DEFAULTS,
    DISPLAY_DEFAULTS, COST_DEFAULTS, RAG_DEFAULTS,
    WEB_SEARCH_DEFAULTS, COMPRESSION_DEFAULTS,
    STREAM_DEFAULTS
)
from tests.fixtures.test_utils import isolated_config, temp_database


class TestConfigurationDefaults(unittest.TestCase):
    """Test configuration default values."""
    
    def test_core_defaults(self):
        """Test core configuration defaults."""
        self.assertEqual(CORE_DEFAULTS['active_prompt'], 'default')
        self.assertFalse(CORE_DEFAULTS['debug'])
        self.assertEqual(CORE_DEFAULTS['short_id_length'], 2)
        self.assertEqual(CORE_DEFAULTS['history_file'], '~/.episodic_history')
        self.assertEqual(CORE_DEFAULTS['visualization_port'], 5000)
    
    def test_topic_defaults(self):
        """Test topic detection defaults."""
        self.assertTrue(TOPIC_DEFAULTS['automatic_topic_detection'])
        self.assertEqual(TOPIC_DEFAULTS['min_messages_before_topic_change'], 8)
        self.assertEqual(TOPIC_DEFAULTS['topic_detection_model'], 'ollama/llama3')
        self.assertTrue(TOPIC_DEFAULTS['analyze_topic_boundaries'])
        self.assertEqual(TOPIC_DEFAULTS['topic_boundary_analysis_messages'], 10)
    
    def test_llm_defaults(self):
        """Test LLM configuration defaults."""
        self.assertEqual(LLM_DEFAULTS['main_model'], 'claude-3-5-sonnet-20241022')
        self.assertTrue(LLM_DEFAULTS['use_context_cache'])
        self.assertEqual(LLM_DEFAULTS['context_depth'], 50)
        self.assertEqual(LLM_DEFAULTS['main_params']['temperature'], 0.7)
        self.assertEqual(LLM_DEFAULTS['topic_params']['temperature'], 0)
    
    def test_display_defaults(self):
        """Test display configuration defaults."""
        self.assertFalse(DISPLAY_DEFAULTS['show_topics'])
        self.assertFalse(DISPLAY_DEFAULTS['show_cost'])
        self.assertTrue(DISPLAY_DEFAULTS['show_drift'])
        self.assertFalse(DISPLAY_DEFAULTS['show_model'])
        self.assertFalse(DISPLAY_DEFAULTS['show_timing'])
    
    def test_compression_defaults(self):
        """Test compression configuration defaults."""
        self.assertEqual(COMPRESSION_DEFAULTS['compression_model'], 'gpt-3.5-turbo')
        self.assertTrue(COMPRESSION_DEFAULTS['auto_compress_topics'])
        self.assertEqual(COMPRESSION_DEFAULTS['min_nodes_for_compression'], 5)
        self.assertEqual(COMPRESSION_DEFAULTS['compression_params']['temperature'], 0.3)
        self.assertEqual(COMPRESSION_DEFAULTS['compression_params']['max_tokens'], 500)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.json")
        self.config = Config(self.config_file)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_model_configuration(self):
        """Test model configuration for different contexts."""
        # Set different models for different contexts
        self.config.set('main_model', 'gpt-4')
        self.config.set('topic_detection_model', 'ollama/llama3')
        self.config.set('compression_model', 'gpt-3.5-turbo')
        self.config.set('synthesis_model', 'claude-3-sonnet')
        
        # Verify each context has correct model
        self.assertEqual(self.config.get('main_model'), 'gpt-4')
        self.assertEqual(self.config.get('topic_detection_model'), 'ollama/llama3')
        self.assertEqual(self.config.get('compression_model'), 'gpt-3.5-turbo')
        self.assertEqual(self.config.get('synthesis_model'), 'claude-3-sonnet')
    
    def test_model_parameters(self):
        """Test model parameter configuration."""
        # Set parameters for main model
        main_params = {
            'temperature': 0.8,
            'max_tokens': 2000,
            'top_p': 0.95,
            'presence_penalty': 0.1
        }
        self.config.set('main_params', main_params)
        
        # Set parameters for topic detection
        topic_params = {
            'temperature': 0,
            'max_tokens': 100
        }
        self.config.set('topic_params', topic_params)
        
        # Verify parameters
        retrieved_main = self.config.get('main_params')
        self.assertEqual(retrieved_main['temperature'], 0.8)
        self.assertEqual(retrieved_main['max_tokens'], 2000)
        
        retrieved_topic = self.config.get('topic_params')
        self.assertEqual(retrieved_topic['temperature'], 0)
    
    def test_configuration_validation(self):
        """Test configuration value validation."""
        # Test valid values
        self.config.set('context_depth', 100)
        self.assertEqual(self.config.get('context_depth'), 100)
        
        # Test type validation
        with self.assertRaises(TypeError):
            self.config.set('context_depth', "not a number")
        
        # Test range validation
        with self.assertRaises(ValueError):
            self.config.set('stream_rate', -5)  # Should be positive
    
    def test_nested_configuration_updates(self):
        """Test updating nested configuration values."""
        # Set initial nested config
        initial_params = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 0.9
        }
        self.config.set('main_params', initial_params)
        
        # Update single nested value
        params = self.config.get('main_params')
        params['temperature'] = 0.5
        self.config.set('main_params', params)
        
        # Verify update
        updated = self.config.get('main_params')
        self.assertEqual(updated['temperature'], 0.5)
        self.assertEqual(updated['max_tokens'], 1000)  # Unchanged
    
    def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        # Set environment variable
        os.environ['EPISODIC_DEBUG'] = 'true'
        os.environ['EPISODIC_MAIN_MODEL'] = 'gpt-4-turbo'
        
        try:
            # Create new config instance to pick up env vars
            config = Config(os.path.join(self.test_dir, "env_config.json"))
            
            # Environment variables should override defaults
            self.assertTrue(config.get('debug'))
            self.assertEqual(config.get('main_model'), 'gpt-4-turbo')
        finally:
            # Clean up
            del os.environ['EPISODIC_DEBUG']
            del os.environ['EPISODIC_MAIN_MODEL']
    
    def test_configuration_migration(self):
        """Test configuration migration from old formats."""
        # Write old format config
        old_config = {
            'model': 'gpt-3.5-turbo',  # Old single model config
            'temperature': 0.7,
            'show_topics': True
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(old_config, f)
        
        # Load with new config system
        config = Config(self.config_file)
        
        # Should migrate old values
        self.assertEqual(config.get('main_model'), 'gpt-3.5-turbo')
        self.assertEqual(config.get('main_params')['temperature'], 0.7)
        self.assertTrue(config.get('show_topics'))
    
    def test_api_key_configuration(self):
        """Test API key configuration handling."""
        # Test setting API keys (without exposing actual keys)
        test_keys = {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'GOOGLE_API_KEY': 'test-google-key'
        }
        
        for key, value in test_keys.items():
            os.environ[key] = value
        
        try:
            # Keys should be available in environment
            for key, value in test_keys.items():
                self.assertEqual(os.environ.get(key), value)
        finally:
            # Clean up
            for key in test_keys:
                if key in os.environ:
                    del os.environ[key]
    
    def test_feature_flags(self):
        """Test feature flag configuration."""
        # RAG feature flags
        self.config.set('rag_enabled', True)
        self.config.set('rag_auto_index', True)
        self.config.set('rag_chunk_size', 1000)
        
        self.assertTrue(self.config.get('rag_enabled'))
        self.assertTrue(self.config.get('rag_auto_index'))
        self.assertEqual(self.config.get('rag_chunk_size'), 1000)
        
        # Web search feature flags
        self.config.set('web_search_enabled', True)
        self.config.set('web_search_provider', 'duckduckgo')
        self.config.set('muse_mode', False)
        
        self.assertTrue(self.config.get('web_search_enabled'))
        self.assertEqual(self.config.get('web_search_provider'), 'duckduckgo')
        self.assertFalse(self.config.get('muse_mode'))
    
    def test_streaming_configuration(self):
        """Test streaming-related configuration."""
        # Set streaming config
        self.config.set('stream_responses', True)
        self.config.set('stream_rate', 20)
        self.config.set('stream_constant_rate', True)
        self.config.set('markdown_rendering', True)
        
        # Verify
        self.assertTrue(self.config.get('stream_responses'))
        self.assertEqual(self.config.get('stream_rate'), 20)
        self.assertTrue(self.config.get('stream_constant_rate'))
        self.assertTrue(self.config.get('markdown_rendering'))
    
    def test_reset_configuration(self):
        """Test resetting configuration to defaults."""
        # Change multiple values
        self.config.set('debug', True)
        self.config.set('main_model', 'gpt-4')
        self.config.set('stream_rate', 30)
        
        # Reset to defaults
        self.config.reset_to_defaults()
        
        # Should be back to defaults
        self.assertFalse(self.config.get('debug'))
        self.assertEqual(self.config.get('main_model'), CORE_DEFAULTS['main_model'])
        self.assertEqual(self.config.get('stream_rate'), STREAM_DEFAULTS['stream_rate'])


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration with other systems."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_context = isolated_config()
        self.config = self.config_context.__enter__()
        self.db_context = temp_database()
        self.db_path = self.db_context.__enter__()
        
    def tearDown(self):
        """Clean up test environment."""
        self.db_context.__exit__(None, None, None)
        self.config_context.__exit__(None, None, None)
    
    def test_database_configuration_sync(self):
        """Test configuration sync with database."""
        from episodic.db import set_config, get_config
        
        # Set config via database
        set_config('test_key', 'test_value')
        
        # Should be retrievable
        value = get_config('test_key')
        self.assertEqual(value, 'test_value')
        
        # Set complex value
        complex_value = {
            'nested': {
                'key': 'value',
                'list': [1, 2, 3]
            }
        }
        set_config('complex_key', complex_value)
        
        retrieved = get_config('complex_key')
        self.assertEqual(retrieved['nested']['key'], 'value')
        self.assertEqual(retrieved['nested']['list'], [1, 2, 3])
    
    @patch('episodic.llm.query_llm')
    def test_model_parameter_application(self, mock_llm):
        """Test that model parameters are correctly applied to LLM calls."""
        from episodic.conversation import ConversationManager
        
        # Set specific parameters
        self.config.set('main_params', {
            'temperature': 0.5,
            'max_tokens': 1500,
            'top_p': 0.8
        })
        
        # Capture parameters passed to LLM
        captured_params = None
        def capture_params(*args, **kwargs):
            nonlocal captured_params
            captured_params = kwargs
            return {'choices': [{'message': {'content': 'Test response'}}]}
        
        mock_llm.side_effect = capture_params
        
        # Send message
        manager = ConversationManager()
        manager.send_message("Test message")
        
        # Verify parameters were applied
        self.assertIsNotNone(captured_params)
        self.assertEqual(captured_params.get('temperature'), 0.5)
        self.assertEqual(captured_params.get('max_tokens'), 1500)
        self.assertEqual(captured_params.get('top_p'), 0.8)
    
    def test_configuration_export_import(self):
        """Test configuration export and import functionality."""
        # Set various configurations
        test_config = {
            'debug': True,
            'main_model': 'gpt-4',
            'topic_detection_model': 'claude-3',
            'main_params': {
                'temperature': 0.6,
                'max_tokens': 2000
            },
            'show_topics': True,
            'stream_rate': 25
        }
        
        for key, value in test_config.items():
            self.config.set(key, value)
        
        # Export configuration
        export_file = os.path.join(self.db_context.name, 'export.json')
        self.config.export_config(export_file)
        
        # Reset config
        self.config.reset_to_defaults()
        
        # Import configuration
        self.config.import_config(export_file)
        
        # Verify all values restored
        for key, value in test_config.items():
            self.assertEqual(self.config.get(key), value)


if __name__ == '__main__':
    unittest.main()