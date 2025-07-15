#!/usr/bin/env python3
"""
Integration tests for CLI commands.

These tests run actual CLI commands and verify their behavior.
Unlike unit tests, these test the full command execution path.
"""

import unittest
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCLICommands(unittest.TestCase):
    """Test CLI commands by actually running them."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize database before running tests."""
        # Initialize the database
        result = subprocess.run(
            ["python", "-m", "episodic", "--init"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Failed to initialize database: {result.stderr}"
    
    def run_cli_command(self, command: str) -> subprocess.CompletedProcess:
        """Helper to run a CLI command and return the result."""
        # For interactive commands, use echo to pipe input
        if command.startswith("/"):
            cmd = f'echo "{command}" | python -m episodic'
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
        else:
            result = subprocess.run(
                ["python", "-m", "episodic"] + command.split(),
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
        return result
    
    def assertCommandSucceeds(self, command: str, expected_in_output: str = None):
        """Assert that a command runs successfully."""
        result = self.run_cli_command(command)
        self.assertEqual(result.returncode, 0, 
                        f"Command '{command}' failed: {result.stderr}")
        if expected_in_output:
            self.assertIn(expected_in_output, result.stdout,
                         f"Expected '{expected_in_output}' not found in output")
        # Check for error messages
        self.assertNotIn("Error executing command:", result.stdout)
        self.assertNotIn("Unknown command:", result.stdout)
    
    def test_help_command(self):
        """Test /help command."""
        self.assertCommandSucceeds("/help", "Type messages directly to chat")
    
    def test_help_shortcut(self):
        """Test /h shortcut for help."""
        self.assertCommandSucceeds("/h", "Type messages directly to chat")
    
    def test_model_commands(self):
        """Test model-related commands."""
        self.assertCommandSucceeds("/model", "Current models:")
        self.assertCommandSucceeds("/model list", "Available models:")
    
    def test_mset_commands(self):
        """Test model parameter commands."""
        self.assertCommandSucceeds("/mset", "Model Parameters")
        self.assertCommandSucceeds("/mset chat", "chat.")
        self.assertCommandSucceeds("/mset chat.temperature 0.7", "Updated")
    
    def test_config_commands(self):
        """Test configuration commands."""
        self.assertCommandSucceeds("/config", "Current Configuration")
        self.assertCommandSucceeds("/config-docs", "Configuration Options")
        self.assertCommandSucceeds("/verify", "Configuration verification")
    
    def test_set_commands(self):
        """Test setting configuration values."""
        self.assertCommandSucceeds("/set debug on", "Debug mode enabled")
        self.assertCommandSucceeds("/set debug off", "Debug mode disabled")
        self.assertCommandSucceeds("/set text_wrap on", "Text wrapping enabled")
    
    def test_topic_commands(self):
        """Test topic-related commands."""
        self.assertCommandSucceeds("/topics", "Conversation Topics")
        self.assertCommandSucceeds("/topics list", "Conversation Topics")
        self.assertCommandSucceeds("/topics stats", "Topic Statistics")
    
    def test_compression_commands(self):
        """Test compression commands."""
        self.assertCommandSucceeds("/compression", "Compression Statistics")
        self.assertCommandSucceeds("/compression stats", "Compression Statistics")
    
    def test_rag_commands(self):
        """Test RAG (knowledge base) commands."""
        result = self.run_cli_command("/rag")
        # RAG might be disabled, but command should still work
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("Error executing command:", result.stdout)
    
    def test_web_commands(self):
        """Test web search commands."""
        self.assertCommandSucceeds("/web test query", "")
        # Note: actual search might fail without API keys, but command should parse
    
    def test_muse_commands(self):
        """Test muse mode commands."""
        self.assertCommandSucceeds("/muse", "Muse mode")
    
    def test_utility_commands(self):
        """Test utility commands."""
        self.assertCommandSucceeds("/about", "About Episodic")
        self.assertCommandSucceeds("/welcome", "Welcome to Episodic")
        self.assertCommandSucceeds("/cost", "Session Costs")
        self.assertCommandSucceeds("/reset")  # Reset doesn't output much
    
    def test_history_commands(self):
        """Test history-related commands."""
        self.assertCommandSucceeds("/history", "Recent messages from conversation")
        self.assertCommandSucceeds("/tree", "Conversation tree")
    
    def test_exit_command(self):
        """Test exit command."""
        result = self.run_cli_command("/exit")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Goodbye", result.stdout)


class TestCLIErrorHandling(unittest.TestCase):
    """Test CLI error handling."""
    
    def run_cli_command(self, command: str) -> subprocess.CompletedProcess:
        """Helper to run a CLI command and return the result."""
        cmd = f'echo "{command}" | python -m episodic'
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=5
        )
        return result
    
    def test_unknown_command(self):
        """Test handling of unknown commands."""
        result = self.run_cli_command("/nonexistent")
        self.assertIn("Unknown command", result.stdout)
    
    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = self.run_cli_command("/topics invalid_action")
        # Should either show error or usage
        self.assertTrue(
            "Unknown" in result.stdout or 
            "Available actions" in result.stdout
        )


if __name__ == "__main__":
    unittest.main()