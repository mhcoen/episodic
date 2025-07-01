"""
Unit tests for unified command interfaces.

Tests the new unified command system including:
- Topics command with subactions
- Compression command with subactions  
- Settings command with subactions
- Command registry functionality
"""

import unittest
from unittest.mock import Mock, patch, call
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from episodic.commands.unified_topics import topics_command
from episodic.commands.unified_compression import compression_command
from episodic.commands.unified_settings import settings_command
from episodic.commands.registry import command_registry, CommandInfo
from tests.fixtures.test_utils import capture_cli_output, isolated_config


class TestUnifiedTopicsCommand(unittest.TestCase):
    """Test unified topics command functionality."""
    
    @patch('episodic.commands.unified_topics.list_topics_impl')
    def test_topics_list_action(self, mock_list):
        """Test /topics list action."""
        topics_command("list")
        mock_list.assert_called_once()
        
    @patch('episodic.commands.unified_topics.rename_topics_impl')
    def test_topics_rename_action(self, mock_rename):
        """Test /topics rename action."""
        topics_command("rename")
        mock_rename.assert_called_once()
        
    @patch('episodic.commands.unified_topics.compress_topic_impl')
    def test_topics_compress_action(self, mock_compress):
        """Test /topics compress action."""
        topics_command("compress")
        mock_compress.assert_called_once()
        
    @patch('episodic.commands.unified_topics.index_topics_impl')
    def test_topics_index_action(self, mock_index):
        """Test /topics index action."""
        topics_command("index", window_size=5, apply=True, verbose=False)
        mock_index.assert_called_once_with(window_size=5, apply=True, verbose=False)
        
    @patch('episodic.commands.unified_topics.topic_scores_impl')
    def test_topics_scores_action(self, mock_scores):
        """Test /topics scores action."""
        topics_command("scores", node_id="n1")
        mock_scores.assert_called_once_with(node_id="n1")
        
    @patch('episodic.commands.unified_topics.get_recent_topics')
    def test_topics_stats_action(self, mock_get_topics):
        """Test /topics stats action."""
        mock_get_topics.return_value = [
            {'name': 'Topic 1', 'end_node_id': 'n1'},
            {'name': 'Topic 2', 'end_node_id': None},
            {'name': 'Topic 1', 'end_node_id': 'n3'},
        ]
        
        with capture_cli_output() as (stdout, stderr):
            topics_command("stats", verbose=True)
            output = stdout.getvalue()
            
        self.assertIn("Topic Statistics", output)
        self.assertIn("Total topics: 3", output)
        self.assertIn("Ongoing: 1", output)
        self.assertIn("Topic 1: 2 occurrences", output)


class TestUnifiedCompressionCommand(unittest.TestCase):
    """Test unified compression command functionality."""
    
    @patch('episodic.commands.unified_compression.stats_impl')
    def test_compression_stats_action(self, mock_stats):
        """Test /compression stats action."""
        compression_command("stats")
        mock_stats.assert_called_once()
        
    @patch('episodic.commands.unified_compression.queue_impl')
    def test_compression_queue_action(self, mock_queue):
        """Test /compression queue action."""
        compression_command("queue")
        mock_queue.assert_called_once()
        
    @patch('episodic.commands.unified_compression.compress_impl')
    def test_compression_compress_with_topic(self, mock_compress):
        """Test /compression compress with topic name."""
        compression_command("compress", topic_name="test-topic")
        mock_compress.assert_called_once_with("test-topic")
        
    @patch('episodic.commands.unified_compression.compress_topic_impl')
    def test_compression_compress_current(self, mock_compress):
        """Test /compression compress without topic (current)."""
        compression_command("compress")
        mock_compress.assert_called_once()
        
    @patch('episodic.commands.unified_compression.api_stats_impl')
    def test_compression_api_stats_action(self, mock_api_stats):
        """Test /compression api-stats action."""
        compression_command("api-stats")
        mock_api_stats.assert_called_once()
        
    @patch('episodic.commands.unified_compression.reset_api_impl')
    def test_compression_reset_api_action(self, mock_reset):
        """Test /compression reset-api action."""
        compression_command("reset-api")
        mock_reset.assert_called_once()


class TestUnifiedSettingsCommand(unittest.TestCase):
    """Test unified settings command functionality."""
    
    @patch('episodic.commands.unified_settings.set_impl')
    def test_settings_show_action(self, mock_set):
        """Test /settings show action (default)."""
        settings_command("show")
        mock_set.assert_called_once_with()
        
    @patch('episodic.commands.unified_settings.set_impl')
    def test_settings_set_action(self, mock_set):
        """Test /settings set action."""
        settings_command("set", param="debug", value="true")
        mock_set.assert_called_once_with("debug", "true")
        
    @patch('episodic.commands.unified_settings.verify_impl')
    def test_settings_verify_action(self, mock_verify):
        """Test /settings verify action."""
        settings_command("verify")
        mock_verify.assert_called_once()
        
    @patch('episodic.commands.unified_settings.cost_impl')
    def test_settings_cost_action(self, mock_cost):
        """Test /settings cost action."""
        settings_command("cost")
        mock_cost.assert_called_once()
        
    @patch('episodic.commands.unified_settings.model_params_impl')
    def test_settings_params_action(self, mock_params):
        """Test /settings params action."""
        settings_command("params", param_set="main")
        mock_params.assert_called_once_with("main")
        
    @patch('episodic.commands.unified_settings.config_docs_impl')
    def test_settings_docs_action(self, mock_docs):
        """Test /settings docs action."""
        settings_command("docs")
        mock_docs.assert_called_once()


class TestCommandRegistry(unittest.TestCase):
    """Test command registry functionality."""
    
    def test_command_registration(self):
        """Test registering a command."""
        # The global registry is already populated, so we test the functionality
        topics_cmd = command_registry.get_command("topics")
        self.assertIsNotNone(topics_cmd)
        self.assertEqual(topics_cmd.category, "Topics")
        self.assertIn("list/rename/compress", topics_cmd.description)
        
    def test_deprecated_command_lookup(self):
        """Test looking up deprecated commands."""
        rename_cmd = command_registry.get_command("rename-topics")
        self.assertIsNotNone(rename_cmd)
        self.assertTrue(rename_cmd.deprecated)
        self.assertEqual(rename_cmd.replacement, "topics rename")
        
    def test_command_aliases(self):
        """Test command aliases."""
        mp_cmd = command_registry.get_command("mp")
        model_params_cmd = command_registry.get_command("model-params")
        self.assertEqual(mp_cmd, model_params_cmd)
        
    def test_commands_by_category(self):
        """Test getting commands organized by category."""
        categories = command_registry.get_commands_by_category()
        
        self.assertIn("Topics", categories)
        self.assertIn("Compression", categories)
        self.assertIn("Configuration", categories)
        
        # Check that topics category has the unified command
        topic_commands = categories["Topics"]
        topic_names = [cmd.name for cmd in topic_commands]
        self.assertIn("topics", topic_names)
        
    def test_unknown_command(self):
        """Test looking up non-existent command."""
        cmd = command_registry.get_command("nonexistent")
        self.assertIsNone(cmd)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of deprecated commands."""
    
    @patch('episodic.commands.topics.rename_ongoing_topics')
    def test_deprecated_rename_topics(self, mock_rename):
        """Test that old /rename-topics still works."""
        # Get the deprecated command
        cmd = command_registry.get_command("rename-topics")
        self.assertIsNotNone(cmd)
        self.assertTrue(cmd.deprecated)
        
        # The handler should still work
        cmd.handler()
        mock_rename.assert_called_once()
        
    def test_all_deprecated_have_replacements(self):
        """Test that all deprecated commands have replacements."""
        categories = command_registry.get_commands_by_category()
        
        for category_cmds in categories.values():
            for cmd in category_cmds:
                if cmd.deprecated:
                    self.assertIsNotNone(cmd.replacement,
                        f"Deprecated command '{cmd.name}' has no replacement")


if __name__ == '__main__':
    unittest.main()