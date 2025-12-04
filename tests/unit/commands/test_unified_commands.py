"""
Unit tests for unified command interfaces.

Tests the new unified command system including:
- Topics command with subactions
- Compression command with subactions
- Command registry functionality
"""

import pytest
from unittest.mock import Mock, patch

from episodic.commands.unified_topics import topics_command
from episodic.commands.unified_compression import compression_command
from episodic.commands.registry import command_registry, CommandInfo, register_all_commands
from tests.fixtures.test_utils import capture_cli_output


@pytest.fixture(scope="module", autouse=True)
def setup_registry():
    """Initialize the command registry before tests run."""
    register_all_commands()


class TestUnifiedTopicsCommand:
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

    @patch('episodic.db.get_topic_detection_scores')
    def test_topics_scores_action(self, mock_scores):
        """Test /topics scores action."""
        mock_scores.return_value = []  # Empty scores
        topics_command("scores", node_id="n1")
        mock_scores.assert_called_once()

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

        assert "Topic Statistics" in output
        assert "Total topics: 3" in output
        assert "Ongoing: 1" in output
        assert "Topic 1: 2 occurrences" in output


class TestUnifiedCompressionCommand:
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


class TestCommandRegistry:
    """Test command registry functionality."""

    def test_command_registration(self):
        """Test registering a command."""
        # The global registry is already populated, so we test the functionality
        topics_cmd = command_registry.get_command("topics")
        assert topics_cmd is not None
        assert topics_cmd.category == "Topics"
        assert "topic" in topics_cmd.description.lower()

    def test_deprecated_command_lookup(self):
        """Test looking up deprecated commands."""
        # model-params is deprecated in favor of mset
        mp_cmd = command_registry.get_command("model-params")
        assert mp_cmd is not None
        assert mp_cmd.deprecated is True
        assert mp_cmd.replacement == "mset"

    def test_command_aliases(self):
        """Test command aliases."""
        mp_cmd = command_registry.get_command("mp")
        model_params_cmd = command_registry.get_command("model-params")
        assert mp_cmd == model_params_cmd

    def test_commands_by_category(self):
        """Test getting commands organized by category."""
        categories = command_registry.get_commands_by_category()

        assert "Topics" in categories
        assert "Compression" in categories
        assert "Configuration" in categories

        # Check that topics category has the unified command
        topic_commands = categories["Topics"]
        topic_names = [cmd.name for cmd in topic_commands]
        assert "topics" in topic_names

    def test_unknown_command(self):
        """Test looking up non-existent command."""
        cmd = command_registry.get_command("nonexistent")
        assert cmd is None


class TestBackwardCompatibility:
    """Test backward compatibility of deprecated commands."""

    def test_deprecated_model_params_has_replacement(self):
        """Test that deprecated model-params has proper replacement."""
        cmd = command_registry.get_command("model-params")
        assert cmd is not None
        assert cmd.deprecated is True
        assert cmd.replacement == "mset"

    def test_all_deprecated_have_replacements(self):
        """Test that all deprecated commands have replacements."""
        categories = command_registry.get_commands_by_category()

        for category_cmds in categories.values():
            for cmd in category_cmds:
                if cmd.deprecated:
                    assert cmd.replacement is not None, \
                        f"Deprecated command '{cmd.name}' has no replacement"
