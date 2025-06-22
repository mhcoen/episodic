#!/usr/bin/env python3
"""
Unit tests for configuration management.

Tests the config module and CLI integration.
"""

import unittest
import tempfile
import shutil
import os
import sys
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.config import Config


class TestConfig(unittest.TestCase):
    """Test Config class functionality."""
    
    def setUp(self):
        """Set up test environment with temporary config file."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.json")
        self.config = Config(self.config_file)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initial_config_creation(self):
        """Test that config file is created if it doesn't exist."""
        # Config file should be created
        self.assertTrue(os.path.exists(self.config_file))
        
        # Should contain empty JSON object
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(data, {})
    
    def test_set_and_get_values(self):
        """Test setting and getting configuration values."""
        # Set simple value
        self.config.set("test_key", "test_value")
        self.assertEqual(self.config.get("test_key"), "test_value")
        
        # Set different types
        self.config.set("bool_key", True)
        self.config.set("int_key", 42)
        self.config.set("float_key", 3.14)
        self.config.set("list_key", [1, 2, 3])
        self.config.set("dict_key", {"nested": "value"})
        
        # Verify all types
        self.assertEqual(self.config.get("bool_key"), True)
        self.assertEqual(self.config.get("int_key"), 42)
        self.assertEqual(self.config.get("float_key"), 3.14)
        self.assertEqual(self.config.get("list_key"), [1, 2, 3])
        self.assertEqual(self.config.get("dict_key"), {"nested": "value"})
    
    def test_get_with_default(self):
        """Test getting values with default fallback."""
        # Non-existent key should return default
        self.assertEqual(self.config.get("nonexistent", "default"), "default")
        self.assertIsNone(self.config.get("nonexistent"))
        
        # Existing key should return actual value, not default
        self.config.set("existing", "actual")
        self.assertEqual(self.config.get("existing", "default"), "actual")
    
    def test_persistence(self):
        """Test that configuration persists across instances."""
        # Set values in first config instance
        self.config.set("persistent_key", "persistent_value")
        self.config.set("number", 123)
        
        # Create new config instance pointing to same file
        new_config = Config(self.config_file)
        
        # Values should persist
        self.assertEqual(new_config.get("persistent_key"), "persistent_value")
        self.assertEqual(new_config.get("number"), 123)
    
    def test_file_synchronization(self):
        """Test that changes are immediately written to file."""
        self.config.set("sync_test", "sync_value")
        
        # Read file directly
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data["sync_test"], "sync_value")
    
    def test_overwrite_values(self):
        """Test overwriting existing values."""
        # Set initial value
        self.config.set("overwrite_key", "initial")
        self.assertEqual(self.config.get("overwrite_key"), "initial")
        
        # Overwrite with different type
        self.config.set("overwrite_key", 42)
        self.assertEqual(self.config.get("overwrite_key"), 42)
        
        # Overwrite again
        self.config.set("overwrite_key", {"new": "structure"})
        self.assertEqual(self.config.get("overwrite_key"), {"new": "structure"})
    
    def test_delete_functionality(self):
        """Test deleting configuration values."""
        # Set a value
        self.config.set("delete_me", "to_be_deleted")
        self.assertEqual(self.config.get("delete_me"), "to_be_deleted")
        
        # Delete it
        self.config.delete("delete_me")
        self.assertIsNone(self.config.get("delete_me"))
        
        # Deleting non-existent key should not raise error
        self.config.delete("never_existed")  # Should not raise
    
    def test_malformed_config_file(self):
        """Test handling of malformed config file."""
        # Write malformed JSON to file
        with open(self.config_file, 'w') as f:
            f.write("{ invalid json")
        
        # Creating config should handle the error gracefully
        try:
            malformed_config = Config(self.config_file)
            # Should start with empty config
            self.assertIsNone(malformed_config.get("any_key"))
        except Exception as e:
            self.fail(f"Config should handle malformed JSON gracefully, but raised: {e}")
    
    def test_config_file_permissions(self):
        """Test behavior when config file has permission issues."""
        # This test is platform-dependent and may not work on all systems
        if os.name == 'posix':  # Unix-like systems
            # Set config file to read-only
            os.chmod(self.config_file, 0o444)
            
            try:
                # Attempting to set should handle permission error
                self.config.set("permission_test", "value")
                # If no exception was raised, restore permissions and continue
                os.chmod(self.config_file, 0o644)
            except PermissionError:
                # Expected behavior - restore permissions
                os.chmod(self.config_file, 0o644)
            except Exception as e:
                # Restore permissions and re-raise
                os.chmod(self.config_file, 0o644)
                raise e
    
    def test_nested_configuration(self):
        """Test handling of nested configuration structures."""
        nested_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "user",
                    "password": "pass"
                }
            },
            "features": {
                "caching": True,
                "debug": False
            }
        }
        
        self.config.set("app_config", nested_config)
        retrieved = self.config.get("app_config")
        
        self.assertEqual(retrieved["database"]["host"], "localhost")
        self.assertEqual(retrieved["database"]["port"], 5432)
        self.assertEqual(retrieved["database"]["credentials"]["username"], "user")
        self.assertEqual(retrieved["features"]["caching"], True)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters in config."""
        unicode_values = {
            "emoji": "🤖🎯✅",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "mixed": "Hello 世界 🌍"
        }
        
        for key, value in unicode_values.items():
            self.config.set(key, value)
            self.assertEqual(self.config.get(key), value)
    
    def test_large_configuration(self):
        """Test handling of large configuration values."""
        # Create a large list
        large_list = list(range(10000))
        self.config.set("large_list", large_list)
        
        retrieved = self.config.get("large_list")
        self.assertEqual(len(retrieved), 10000)
        self.assertEqual(retrieved[0], 0)
        self.assertEqual(retrieved[-1], 9999)
        
        # Create a large dictionary
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        self.config.set("large_dict", large_dict)
        
        retrieved = self.config.get("large_dict")
        self.assertEqual(len(retrieved), 1000)
        self.assertEqual(retrieved["key_0"], "value_0")
        self.assertEqual(retrieved["key_999"], "value_999")


class TestConfigIntegration(unittest.TestCase):
    """Test Config integration with the rest of the system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "integration_config.json")
        
        # Import the global config instance
        from episodic.config import config
        self.global_config = config
        
        # Store original config file path
        self.original_file = self.global_config.config_file
        
        # Temporarily switch to test config file
        self.global_config.config_file = self.config_file
        self.global_config.config_data = {}
        self.global_config._save()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original config
        self.global_config.config_file = self.original_file
        self.global_config._load()
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_global_config_usage(self):
        """Test usage of the global config instance."""
        # Set value using global config
        self.global_config.set("global_test", "global_value")
        
        # Verify it persists
        self.assertEqual(self.global_config.get("global_test"), "global_value")
        
        # Verify it's written to file
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(data["global_test"], "global_value")
    
    def test_cli_configuration_values(self):
        """Test configuration values used by CLI."""
        cli_configs = {
            "debug": False,
            "show_cost": True,
            "use_context_cache": True,
            "active_prompt": "default",
            "history_file": "~/.episodic_history"
        }
        
        for key, value in cli_configs.items():
            self.global_config.set(key, value)
            self.assertEqual(self.global_config.get(key), value)


if __name__ == '__main__':
    unittest.main()