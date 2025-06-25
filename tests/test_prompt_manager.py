#!/usr/bin/env python3
"""
Unit tests for the prompt manager.

Tests prompt loading, management, and configuration integration.
"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.prompt_manager import PromptManager
from episodic.config import config


class TestPromptManager(unittest.TestCase):
    """Test PromptManager functionality."""
    
    def setUp(self):
        """Set up test environment with temporary prompts directory."""
        self.test_dir = tempfile.mkdtemp()
        self.prompts_dir = os.path.join(self.test_dir, "prompts")
        os.makedirs(self.prompts_dir)
        
        # Create test prompt files
        self.create_test_prompt("default.md", """---
name: default
description: Default system prompt
version: 1.0
---

You are a helpful assistant.""")
        
        self.create_test_prompt("technical.md", """---
name: technical
description: Technical assistant prompt
version: 1.1
tags: [technical, coding]
---

You are a technical assistant specializing in software development.""")
        
        self.create_test_prompt("simple.txt", "You are a simple assistant.")
        
        # Create prompt manager
        self.manager = PromptManager(self.prompts_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_prompt(self, filename, content):
        """Helper method to create test prompt files."""
        filepath = os.path.join(self.prompts_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def test_prompt_loading(self):
        """Test that prompts are loaded correctly."""
        prompts = self.manager.list()
        
        # Check that all test prompts were loaded
        self.assertIn("default", prompts)
        self.assertIn("technical", prompts)
        self.assertIn("simple", prompts)
        self.assertEqual(len(prompts), 3)
    
    def test_prompt_content_retrieval(self):
        """Test retrieving prompt content."""
        # Test markdown prompt with frontmatter
        default_content = self.manager.get("default")
        self.assertEqual(default_content, "You are a helpful assistant.")
        
        # Test technical prompt
        technical_content = self.manager.get("technical")
        self.assertEqual(technical_content, "You are a technical assistant specializing in software development.")
        
        # Test simple text prompt
        simple_content = self.manager.get("simple")
        self.assertEqual(simple_content, "You are a simple assistant.")
        
        # Test non-existent prompt
        missing_content = self.manager.get("nonexistent")
        self.assertIsNone(missing_content)
    
    def test_metadata_parsing(self):
        """Test YAML frontmatter metadata parsing."""
        # Test default prompt metadata
        default_meta = self.manager.get_metadata("default")
        self.assertIsNotNone(default_meta)
        self.assertEqual(default_meta["name"], "default")
        self.assertEqual(default_meta["description"], "Default system prompt")
        self.assertEqual(default_meta["version"], 1.0)
        
        # Test technical prompt metadata
        technical_meta = self.manager.get_metadata("technical")
        self.assertIsNotNone(technical_meta)
        self.assertEqual(technical_meta["name"], "technical")
        self.assertEqual(technical_meta["description"], "Technical assistant prompt")
        self.assertEqual(technical_meta["version"], 1.1)
        self.assertEqual(technical_meta["tags"], ["technical", "coding"])
        
        # Test simple prompt (no metadata)
        simple_meta = self.manager.get_metadata("simple")
        self.assertEqual(simple_meta, {})
        
        # Test non-existent prompt
        missing_meta = self.manager.get_metadata("nonexistent")
        self.assertIsNone(missing_meta)
    
    def test_get_active_prompt(self):
        """Test getting active prompt name from config."""
        # Test with configured active prompt
        mock_config_get = MagicMock(return_value="technical")
        active = self.manager.get_active_prompt(mock_config_get)
        self.assertEqual(active, "technical")
        mock_config_get.assert_called_with("active_prompt", "default")
        
        # Test with default fallback (config returns the default when key not found)
        mock_config_get = MagicMock(side_effect=lambda key, default: default)
        active = self.manager.get_active_prompt(mock_config_get, "default")
        self.assertEqual(active, "default")
    
    def test_get_active_prompt_content(self):
        """Test getting active prompt content."""
        mock_config_get = MagicMock()
        
        # Test with existing active prompt
        mock_config_get.return_value = "technical"
        content = self.manager.get_active_prompt_content(mock_config_get)
        self.assertEqual(content, "You are a technical assistant specializing in software development.")
        
        # Test with non-existent active prompt (should fall back to default)
        mock_config_get.return_value = "nonexistent"
        content = self.manager.get_active_prompt_content(mock_config_get, "default")
        self.assertEqual(content, "You are a helpful assistant.")
        
        # Test with no prompts available (should return fallback)
        empty_manager = PromptManager(os.path.join(self.test_dir, "empty"))
        content = empty_manager.get_active_prompt_content(mock_config_get)
        self.assertEqual(content, "You are a helpful assistant.")
    
    def test_reload_functionality(self):
        """Test prompt reloading."""
        # Initial state
        initial_prompts = self.manager.list()
        self.assertEqual(len(initial_prompts), 3)
        
        # Add a new prompt file
        self.create_test_prompt("new_prompt.md", """---
name: new_prompt
description: A new test prompt
---

You are a newly added assistant.""")
        
        # Reload and check
        self.manager.reload()
        updated_prompts = self.manager.list()
        self.assertEqual(len(updated_prompts), 4)
        self.assertIn("new_prompt", updated_prompts)
        
        # Verify new prompt content
        new_content = self.manager.get("new_prompt")
        self.assertEqual(new_content, "You are a newly added assistant.")
    
    def test_empty_prompts_directory(self):
        """Test behavior with empty prompts directory."""
        empty_dir = os.path.join(self.test_dir, "empty")
        empty_manager = PromptManager(empty_dir)
        
        # Should create directory
        self.assertTrue(os.path.exists(empty_dir))
        
        # Should have no prompts
        prompts = empty_manager.list()
        self.assertEqual(len(prompts), 0)
        
        # Should return None for any prompt
        content = empty_manager.get("anything")
        self.assertIsNone(content)
    
    @patch('episodic.prompt_manager.YAML_AVAILABLE', False)
    def test_yaml_unavailable_fallback(self):
        """Test behavior when PyYAML is not available."""
        # Create a new manager without YAML support
        fallback_manager = PromptManager(self.prompts_dir)
        
        # Should still load prompts, but treat frontmatter as content
        prompts = fallback_manager.list()
        self.assertIn("default", prompts)
        
        # Content should include the frontmatter
        default_content = fallback_manager.get("default")
        self.assertIn("---", default_content)
        self.assertIn("name: default", default_content)
    
    def test_malformed_frontmatter(self):
        """Test handling of malformed YAML frontmatter."""
        # Create prompt with malformed frontmatter
        self.create_test_prompt("malformed.md", """---
invalid yaml: [unclosed bracket
name: malformed
---

You are an assistant with malformed metadata.""")
        
        # Reload to pick up new file
        self.manager.reload()
        
        # Should still load the prompt (treating entire content as prompt)
        prompts = self.manager.list()
        self.assertIn("malformed", prompts)
        
        # Content should include the whole file
        content = self.manager.get("malformed")
        self.assertIn("invalid yaml", content)
        self.assertIn("You are an assistant", content)


class TestPromptManagerIntegration(unittest.TestCase):
    """Test PromptManager integration with the config system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.prompts_dir = os.path.join(self.test_dir, "prompts")
        os.makedirs(self.prompts_dir)
        
        # Create test prompts
        with open(os.path.join(self.prompts_dir, "default.md"), 'w') as f:
            f.write("You are a default assistant.")
        
        with open(os.path.join(self.prompts_dir, "custom.md"), 'w') as f:
            f.write("You are a custom assistant.")
        
        self.manager = PromptManager(self.prompts_dir)
        
        # Store original config values
        self.original_active_prompt = config.get("active_prompt")
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original config
        if self.original_active_prompt:
            config.set("active_prompt", self.original_active_prompt)
        else:
            config.delete("active_prompt")
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_integration(self):
        """Test integration with config system."""
        # Set active prompt in config
        config.set("active_prompt", "custom")
        
        # Get active prompt content
        content = self.manager.get_active_prompt_content(config.get)
        self.assertEqual(content, "You are a custom assistant.")
        
        # Change active prompt
        config.set("active_prompt", "default")
        content = self.manager.get_active_prompt_content(config.get)
        self.assertEqual(content, "You are a default assistant.")
    
    def test_fallback_behavior(self):
        """Test fallback behavior when active prompt is not found."""
        # Set non-existent active prompt
        config.set("active_prompt", "nonexistent")
        
        # Should fall back to default
        content = self.manager.get_active_prompt_content(config.get, "default")
        self.assertEqual(content, "You are a default assistant.")
        
        # If default also doesn't exist, should return hardcoded fallback
        config.set("active_prompt", "nonexistent")
        content = self.manager.get_active_prompt_content(config.get, "also_nonexistent")
        self.assertEqual(content, "You are a helpful assistant.")


if __name__ == '__main__':
    unittest.main()