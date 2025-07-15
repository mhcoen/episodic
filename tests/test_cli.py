#!/usr/bin/env python3
"""
Unit tests for CLI functionality.

Tests the command-line interface, command parsing, and user interactions.
"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add the project root to the path so we can import episodic modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic import cli
from episodic.commands import set as set_command
from episodic.llm import enable_cache, disable_cache
from episodic.config import config
from episodic.cli_helpers import _parse_flag_value, _has_flag, _remove_flag_and_value
from episodic.commands.navigation import format_role_display
from episodic.configuration import DEFAULT_MODEL, DEFAULT_SYSTEM_MESSAGE


class TestCLIHelpers(unittest.TestCase):
    """Test CLI helper functions."""
    
    def test_parse_flag_value(self):
        """Test parsing flag values from argument lists."""
        # Test successful flag parsing
        args = ["command", "--model", "gpt-4", "text"]
        result = _parse_flag_value(args, ["--model", "-m"])
        self.assertEqual(result, "gpt-4")
        
        # Test alternative flag name
        args = ["command", "-m", "claude", "text"]
        result = _parse_flag_value(args, ["--model", "-m"])
        self.assertEqual(result, "claude")
        
        # Test missing flag
        args = ["command", "text"]
        result = _parse_flag_value(args, ["--model", "-m"])
        self.assertIsNone(result)
        
        # Test flag without value
        args = ["command", "--model"]
        result = _parse_flag_value(args, ["--model"])
        self.assertIsNone(result)
    
    def test_remove_flag_and_value(self):
        """Test removing flags and values from argument lists."""
        # Test removing flag with value
        args = ["command", "--parent", "abc123", "text"]
        result = _remove_flag_and_value(args, ["--parent", "-p"])
        self.assertEqual(result, ["command", "text"])
        
        # Test removing alternative flag
        args = ["command", "-p", "abc123", "text"]
        result = _remove_flag_and_value(args, ["--parent", "-p"])
        self.assertEqual(result, ["command", "text"])
        
        # Test removing flag with text that looks like a value
        args = ["command", "--flag", "text"]
        result = _remove_flag_and_value(args, ["--flag"])
        self.assertEqual(result, ["command"])
        
        # Test no flag to remove
        args = ["command", "text"]
        result = _remove_flag_and_value(args, ["--nonexistent"])
        self.assertEqual(result, ["command", "text"])
    
    def test_has_flag(self):
        """Test checking for flag presence."""
        args = ["command", "--verbose", "text"]
        self.assertTrue(_has_flag(args, ["--verbose", "-v"]))
        self.assertTrue(_has_flag(args, ["-v", "--verbose"]))
        self.assertFalse(_has_flag(args, ["--quiet", "-q"]))
    
    def test_format_role_display(self):
        """Test role formatting for display."""
        # format_role_display was moved to navigation module and now shows emoji/model info
        result = format_role_display("user")
        self.assertIn("👤", result)
        
        result = format_role_display("assistant")
        self.assertIn("🤖", result)


class TestCLICommands(unittest.TestCase):
    """Test CLI commands that don't require database setup."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = config.get("database_path")
        config.set("database_path", os.path.join(self.test_dir, "test.db"))
        
        # Reset global state - these don't exist in refactored CLI
        # Instead, set them in config
        config.set("model", DEFAULT_MODEL)
        config.set("system_message", DEFAULT_SYSTEM_MESSAGE)
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original database path
        if self.original_db_path:
            config.set("database_path", self.original_db_path)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('typer.secho')
    @patch('episodic.commands.prompts.get_available_prompts')
    @patch('episodic.commands.prompts.load_prompt')
    @patch('episodic.commands.prompts.get_active_prompt')
    def test_prompts_list_command(self, mock_get_active, mock_load, mock_get_available, mock_secho):
        """Test prompts list command."""
        mock_get_available.return_value = ["default", "comedian", "technical"]
        mock_get_active.return_value = "default"
        mock_load.return_value = {"description": "Test prompt"}
        
        from episodic.commands.prompts import prompts
        prompts("list")
        
        # Verify functions were called
        mock_get_available.assert_called_once()
        
        # Verify output was shown
        self.assertTrue(mock_secho.called)
    
    @patch('typer.secho')
    @patch('episodic.commands.prompts.get_available_prompts')
    @patch('episodic.commands.prompts.load_prompt')
    @patch('episodic.commands.prompts.conversation_manager')
    def test_prompts_use_command(self, mock_conv_manager, mock_load, mock_get_available, mock_secho):
        """Test prompts use command."""
        mock_get_available.return_value = ["default", "comedian"]
        mock_load.return_value = {"content": "Test prompt content", "description": "Test prompt"}
        
        from episodic.commands.prompts import prompts
        prompts("use", "comedian")
        
        # Verify config was updated
        self.assertEqual(config.get("active_prompt"), "comedian")
        
        # Verify conversation manager's prompt was updated
        self.assertEqual(mock_conv_manager.system_prompt, "Test prompt content")


class TestCLIInitialization(unittest.TestCase):
    """Test CLI initialization functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = config.get("database_path")
        config.set("database_path", os.path.join(self.test_dir, "test.db"))
    
    def tearDown(self):
        """Clean up test environment."""
        if self.original_db_path:
            config.set("database_path", self.original_db_path)
        shutil.rmtree(self.test_dir, ignore_errors=True)


class TestCLIConfiguration(unittest.TestCase):
    """Test CLI configuration management."""
    
    def test_set_command_debug(self):
        """Test setting debug mode."""
        # Test enabling debug
        set_command("debug", "on")
        self.assertTrue(config.get("debug"))
        
        set_command("debug", "true")
        self.assertTrue(config.get("debug"))
        
        # Test disabling debug
        set_command("debug", "off")
        self.assertFalse(config.get("debug"))
        
        set_command("debug", "false")
        self.assertFalse(config.get("debug"))
    
    @patch('episodic.commands.settings.disable_cache')
    @patch('episodic.commands.settings.enable_cache')
    @patch('typer.echo')  # Need to patch echo too
    def test_set_command_cache(self, mock_echo, mock_enable, mock_disable):
        """Test setting cache mode."""
        # Test enabling cache
        set_command("cache", "on")
        mock_enable.assert_called_once()
        
        # Reset mocks
        mock_enable.reset_mock()
        mock_disable.reset_mock()
        
        # Test disabling cache
        set_command("cache", "off")
        mock_disable.assert_called_once()
    
    def test_set_command_cost_display(self):
        """Test setting cost display."""
        set_command("cost", "on")
        self.assertTrue(config.get("show_cost"))
        
        set_command("cost", "off")
        self.assertFalse(config.get("show_cost"))
    
    @patch('typer.echo')
    def test_set_command_invalid(self, mock_echo):
        """Test setting invalid configuration."""
        set_command("invalid_setting", "value")
        
        # Should show error message - the first call is "Unknown parameter: invalid_setting"
        mock_echo.assert_called()
        # Get the first call (there are multiple calls for the help text)
        first_call = mock_echo.call_args_list[0][0][0]
        self.assertIn("Unknown parameter", first_call)


class TestCLISessionManagement(unittest.TestCase):
    """Test CLI session and state management."""
    
    def test_session_cost_accumulation(self):
        """Test that session costs accumulate correctly."""
        from episodic.conversation import ConversationManager
        
        # Create a new conversation manager instance for testing
        cm = ConversationManager()
        
        # Initialize session costs
        cm.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }
        
        # Simulate cost updates
        cost_info = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.005
        }
        
        # Update costs twice to test accumulation
        for _ in range(2):
            # In the real code, this happens inside _handle_chat_message_impl
            cm.session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
            cm.session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
            cm.session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
            cm.session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
        
        # Verify accumulation
        costs = cm.get_session_costs()
        self.assertEqual(costs["total_input_tokens"], 200)
        self.assertEqual(costs["total_output_tokens"], 100)
        self.assertEqual(costs["total_tokens"], 300)
        self.assertAlmostEqual(costs["total_cost_usd"], 0.01, places=3)


    def test_markdown_command_aliases(self):
        """Test that markdown command aliases work correctly."""
        from episodic.cli_command_router import _handle_export, _handle_import, _handle_ls
        from unittest.mock import patch, MagicMock
        
        # Test /ex alias for /export
        with patch('episodic.commands.markdown_export.export_command') as mock_export:
            _handle_export(['current'])
            mock_export.assert_called_once()
        
        # Test /im alias for /import
        with patch('episodic.commands.markdown_import.import_command') as mock_import:
            _handle_import(['test.md'])
            mock_import.assert_called_once()
        
        # Test /ls alias for /files
        with patch('episodic.commands.ls.ls_command') as mock_ls:
            _handle_ls([])
            mock_ls.assert_called_once()


if __name__ == '__main__':
    unittest.main()