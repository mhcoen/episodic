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
from episodic.config import config
from episodic.cli_constants import *


class TestCLIHelpers(unittest.TestCase):
    """Test CLI helper functions."""
    
    def test_parse_flag_value(self):
        """Test parsing flag values from argument lists."""
        # Test successful flag parsing
        args = ["command", "--model", "gpt-4", "text"]
        result = cli._parse_flag_value(args, ["--model", "-m"])
        self.assertEqual(result, "gpt-4")
        
        # Test alternative flag name
        args = ["command", "-m", "claude", "text"]
        result = cli._parse_flag_value(args, ["--model", "-m"])
        self.assertEqual(result, "claude")
        
        # Test missing flag
        args = ["command", "text"]
        result = cli._parse_flag_value(args, ["--model", "-m"])
        self.assertIsNone(result)
        
        # Test flag without value
        args = ["command", "--model"]
        result = cli._parse_flag_value(args, ["--model"])
        self.assertIsNone(result)
    
    def test_remove_flag_and_value(self):
        """Test removing flags and values from argument lists."""
        # Test removing flag with value
        args = ["command", "--parent", "abc123", "text"]
        result = cli._remove_flag_and_value(args, ["--parent", "-p"])
        self.assertEqual(result, ["command", "text"])
        
        # Test removing alternative flag
        args = ["command", "-p", "abc123", "text"]
        result = cli._remove_flag_and_value(args, ["--parent", "-p"])
        self.assertEqual(result, ["command", "text"])
        
        # Test removing flag without value
        args = ["command", "--flag", "text"]
        result = cli._remove_flag_and_value(args, ["--flag"])
        self.assertEqual(result, ["command", "text"])
        
        # Test no flag to remove
        args = ["command", "text"]
        result = cli._remove_flag_and_value(args, ["--nonexistent"])
        self.assertEqual(result, ["command", "text"])
    
    def test_has_flag(self):
        """Test checking for flag presence."""
        args = ["command", "--verbose", "text"]
        self.assertTrue(cli._has_flag(args, ["--verbose", "-v"]))
        self.assertTrue(cli._has_flag(args, ["-v", "--verbose"]))
        self.assertFalse(cli._has_flag(args, ["--quiet", "-q"]))
    
    def test_format_role_display(self):
        """Test role formatting for display."""
        self.assertEqual(cli.format_role_display("user"), "User")
        self.assertEqual(cli.format_role_display("assistant"), "Assistant")
        self.assertEqual(cli.format_role_display("system"), "System")
        self.assertEqual(cli.format_role_display(None), "Unknown")
        self.assertEqual(cli.format_role_display(""), "Unknown")


class TestCLICommands(unittest.TestCase):
    """Test CLI commands that don't require database setup."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = config.get("database_path")
        config.set("database_path", os.path.join(self.test_dir, "test.db"))
        
        # Reset global state
        cli.current_node_id = None
        cli.default_model = DEFAULT_MODEL
        cli.default_system = DEFAULT_SYSTEM_MESSAGE
        cli.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original database path
        if self.original_db_path:
            config.set("database_path", self.original_db_path)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('episodic.cli.typer.echo')
    def test_version_command(self, mock_echo):
        """Test version command."""
        cli.version()
        mock_echo.assert_called_once()
        call_args = mock_echo.call_args[0][0]
        self.assertIn("Episodic CLI", call_args)
    
    @patch('episodic.cli.typer.echo')
    @patch('episodic.cli.get_current_provider')
    @patch('episodic.cli.get_available_providers')
    def test_providers_command(self, mock_get_providers, mock_get_current, mock_echo):
        """Test providers command."""
        mock_get_providers.return_value = ["openai", "anthropic", "ollama"]
        mock_get_current.return_value = "openai"
        
        cli.providers()
        
        # Check that provider information was displayed
        self.assertTrue(mock_echo.called)
        echo_calls = [call[0][0] for call in mock_echo.call_args_list]
        
        # Verify current provider is shown
        current_provider_shown = any("openai" in call and "current" in call.lower() for call in echo_calls)
        self.assertTrue(current_provider_shown)
    
    @patch('episodic.cli.typer.echo')
    @patch('episodic.cli.PromptManager')
    def test_prompts_list_command(self, mock_prompt_manager, mock_echo):
        """Test prompts list command."""
        mock_manager = MagicMock()
        mock_manager.list.return_value = ["default", "comedian", "technical"]
        mock_manager.get_metadata.return_value = {"description": "Test prompt"}
        mock_prompt_manager.return_value = mock_manager
        
        cli.prompts("list")
        
        # Verify prompt manager was called
        mock_manager.list.assert_called_once()
        
        # Verify output was shown
        self.assertTrue(mock_echo.called)
    
    @patch('episodic.cli.typer.echo')
    @patch('episodic.cli.PromptManager')
    def test_prompts_use_command(self, mock_prompt_manager, mock_echo):
        """Test prompts use command."""
        mock_manager = MagicMock()
        mock_manager.list.return_value = ["default", "comedian"]
        mock_manager.get_active_prompt_content.return_value = "Test prompt content"
        mock_prompt_manager.return_value = mock_manager
        
        cli.prompts("use", "comedian")
        
        # Verify config was updated
        self.assertEqual(config.get("active_prompt"), "comedian")
        
        # Verify global default_system was updated
        self.assertEqual(cli.default_system, "Test prompt content")


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
    
    @patch('episodic.cli.PromptManager')
    def test_initialize_prompt(self, mock_prompt_manager):
        """Test prompt initialization."""
        mock_manager = MagicMock()
        mock_manager.get_active_prompt_content.return_value = "Custom prompt content"
        mock_prompt_manager.return_value = mock_manager
        
        # Store original value
        original_system = cli.default_system
        
        # Test initialization
        cli._initialize_prompt()
        
        # Verify prompt was updated
        self.assertEqual(cli.default_system, "Custom prompt content")
        
        # Test fallback on exception
        mock_manager.get_active_prompt_content.side_effect = Exception("Test error")
        cli._initialize_prompt()
        
        # Should fall back to default
        self.assertEqual(cli.default_system, DEFAULT_SYSTEM_MESSAGE)
    
    @patch('episodic.cli.ensure_provider_matches_model')
    @patch('episodic.cli.typer.echo')
    def test_initialize_model(self, mock_echo, mock_ensure_provider):
        """Test model initialization."""
        cli._initialize_model()
        
        # Verify provider matching was called
        mock_ensure_provider.assert_called_once()
        
        # Verify model information was displayed
        self.assertTrue(mock_echo.called)


class TestCLIConfiguration(unittest.TestCase):
    """Test CLI configuration management."""
    
    def test_set_command_debug(self):
        """Test setting debug mode."""
        # Test enabling debug
        cli.set("debug", "on")
        self.assertTrue(config.get("debug"))
        
        cli.set("debug", "true")
        self.assertTrue(config.get("debug"))
        
        # Test disabling debug
        cli.set("debug", "off")
        self.assertFalse(config.get("debug"))
        
        cli.set("debug", "false")
        self.assertFalse(config.get("debug"))
    
    @patch('episodic.cli.enable_cache')
    @patch('episodic.cli.disable_cache')
    def test_set_command_cache(self, mock_disable, mock_enable):
        """Test setting cache mode."""
        # Test enabling cache
        cli.set("cache", "on")
        mock_enable.assert_called_once()
        
        # Test disabling cache
        cli.set("cache", "off")
        mock_disable.assert_called_once()
    
    def test_set_command_cost_display(self):
        """Test setting cost display."""
        cli.set("cost", "on")
        self.assertTrue(config.get("show_cost"))
        
        cli.set("cost", "off")
        self.assertFalse(config.get("show_cost"))
    
    @patch('episodic.cli.typer.echo')
    def test_set_command_invalid(self, mock_echo):
        """Test setting invalid configuration."""
        cli.set("invalid_setting", "value")
        
        # Should show error message
        mock_echo.assert_called()
        error_msg = mock_echo.call_args[0][0]
        self.assertIn("Unknown setting", error_msg)


class TestCLISessionManagement(unittest.TestCase):
    """Test CLI session and state management."""
    
    def test_display_session_summary(self):
        """Test session summary display."""
        # Set up session costs
        cli.session_costs = {
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_tokens": 1500,
            "total_cost_usd": 0.025
        }
        
        with patch('episodic.cli.typer.echo') as mock_echo:
            cli.display_session_summary()
            
            # Verify summary was displayed
            self.assertTrue(mock_echo.called)
            calls = [call[0][0] for call in mock_echo.call_args_list]
            
            # Check that token and cost information is included
            summary_text = " ".join(calls)
            self.assertIn("1000", summary_text)  # Input tokens
            self.assertIn("500", summary_text)   # Output tokens
            self.assertIn("1500", summary_text)  # Total tokens
            self.assertIn("0.025", summary_text) # Cost
    
    def test_session_cost_accumulation(self):
        """Test that session costs accumulate correctly."""
        # Reset session costs
        cli.session_costs = {
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
            cli.session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
            cli.session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
            cli.session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
            cli.session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
        
        # Verify accumulation
        self.assertEqual(cli.session_costs["total_input_tokens"], 200)
        self.assertEqual(cli.session_costs["total_output_tokens"], 100)
        self.assertEqual(cli.session_costs["total_tokens"], 300)
        self.assertAlmostEqual(cli.session_costs["total_cost_usd"], 0.01, places=3)


if __name__ == '__main__':
    unittest.main()