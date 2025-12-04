"""
Unit tests for configuration management.

Tests the config module functionality using pytest.
"""

import pytest
import os
import json

from episodic.config import Config


class TestConfig:
    """Test Config class functionality."""

    @pytest.fixture(autouse=True)
    def setup_config(self, tmp_path):
        """Set up test environment with temporary config file."""
        self.test_dir = tmp_path
        self.config_file = str(tmp_path / "test_config.json")
        self.config = Config(self.config_file)

    def test_initial_config_creation(self):
        """Test that config file is created if it doesn't exist."""
        # Config file should be created
        assert os.path.exists(self.config_file)

        # Should contain default values
        with open(self.config_file, 'r') as f:
            data = json.load(f)

        # Check that key default values are set (config now has many more defaults)
        assert data.get("active_prompt") == "default"
        assert data.get("debug") is False
        assert data.get("show_cost") is False
        assert data.get("show_drift") is True
        assert data.get("auto_compress_topics") is True
        assert data.get("stream_responses") is True
        assert data.get("stream_rate") == 15
        assert data.get("stream_constant_rate") is False

    def test_set_and_get_values(self):
        """Test setting and getting configuration values."""
        # Set simple value
        self.config.set("test_key", "test_value")
        assert self.config.get("test_key") == "test_value"

        # Set different types
        self.config.set("bool_key", True)
        self.config.set("int_key", 42)
        self.config.set("float_key", 3.14)
        self.config.set("list_key", [1, 2, 3])
        self.config.set("dict_key", {"nested": "value"})

        # Verify all types
        assert self.config.get("bool_key") is True
        assert self.config.get("int_key") == 42
        assert self.config.get("float_key") == 3.14
        assert self.config.get("list_key") == [1, 2, 3]
        assert self.config.get("dict_key") == {"nested": "value"}

    def test_get_with_default(self):
        """Test getting values with default fallback."""
        # Non-existent key should return default
        assert self.config.get("nonexistent", "default") == "default"
        assert self.config.get("nonexistent") is None

        # Existing key should return actual value, not default
        self.config.set("existing", "actual")
        assert self.config.get("existing", "default") == "actual"

    def test_persistence(self):
        """Test that configuration persists across instances using save_setting."""
        # Set values using save_setting for persistence
        self.config.save_setting("persistent_key", "persistent_value")
        self.config.save_setting("number", 123)

        # Create new config instance pointing to same file
        new_config = Config(self.config_file)

        # Values should persist
        assert new_config.get("persistent_key") == "persistent_value"
        assert new_config.get("number") == 123

    def test_set_is_runtime_only(self):
        """Test that set() is runtime-only and does not persist."""
        self.config.set("runtime_key", "runtime_value")
        assert self.config.get("runtime_key") == "runtime_value"

        # Create new config instance - runtime value should not persist
        new_config = Config(self.config_file)
        # Since runtime values aren't persisted, it should not be in new instance
        assert new_config.get("runtime_key") is None

    def test_file_synchronization(self):
        """Test that save_setting writes immediately to file."""
        self.config.save_setting("sync_test", "sync_value")

        # Read file directly
        with open(self.config_file, 'r') as f:
            data = json.load(f)

        assert data["sync_test"] == "sync_value"

    def test_overwrite_values(self):
        """Test overwriting existing values."""
        # Set initial value
        self.config.set("overwrite_key", "initial")
        assert self.config.get("overwrite_key") == "initial"

        # Overwrite with different type
        self.config.set("overwrite_key", 42)
        assert self.config.get("overwrite_key") == 42

        # Overwrite again
        self.config.set("overwrite_key", {"new": "structure"})
        assert self.config.get("overwrite_key") == {"new": "structure"}

    def test_delete_functionality(self):
        """Test deleting configuration values."""
        # Set a value
        self.config.set("delete_me", "to_be_deleted")
        assert self.config.get("delete_me") == "to_be_deleted"

        # Delete it
        self.config.delete("delete_me")
        assert self.config.get("delete_me") is None

        # Deleting non-existent key should not raise error
        self.config.delete("never_existed")  # Should not raise

    def test_malformed_config_file(self):
        """Test handling of malformed config file."""
        # Write malformed JSON to file
        with open(self.config_file, 'w') as f:
            f.write("{ invalid json")

        # Creating config should handle the error gracefully
        malformed_config = Config(self.config_file)
        # Should start with empty config
        assert malformed_config.get("any_key") is None

    @pytest.mark.skipif(os.name != 'posix', reason="POSIX-specific test")
    def test_config_file_permissions(self):
        """Test behavior when config file has permission issues."""
        # Set config file to read-only
        os.chmod(self.config_file, 0o444)

        try:
            # Attempting to set should handle permission error
            self.config.set("permission_test", "value")
        finally:
            # Restore permissions
            os.chmod(self.config_file, 0o644)

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

        assert retrieved["database"]["host"] == "localhost"
        assert retrieved["database"]["port"] == 5432
        assert retrieved["database"]["credentials"]["username"] == "user"
        assert retrieved["features"]["caching"] is True

    def test_unicode_handling(self):
        """Test handling of Unicode characters in config."""
        unicode_values = {
            "emoji": "\U0001f916\U0001f3af\u2705",
            "chinese": "\u4f60\u597d\u4e16\u754c",
            "arabic": "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
            "mixed": "Hello \u4e16\u754c \U0001f30d"
        }

        for key, value in unicode_values.items():
            self.config.set(key, value)
            assert self.config.get(key) == value

    def test_large_configuration(self):
        """Test handling of large configuration values."""
        # Create a large list
        large_list = list(range(10000))
        self.config.set("large_list", large_list)

        retrieved = self.config.get("large_list")
        assert len(retrieved) == 10000
        assert retrieved[0] == 0
        assert retrieved[-1] == 9999

        # Create a large dictionary
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        self.config.set("large_dict", large_dict)

        retrieved = self.config.get("large_dict")
        assert len(retrieved) == 1000
        assert retrieved["key_0"] == "value_0"
        assert retrieved["key_999"] == "value_999"


class TestConfigIntegration:
    """Test Config integration with the rest of the system."""

    @pytest.fixture(autouse=True)
    def setup_integration(self, tmp_path):
        """Set up test environment with isolated global config."""
        from pathlib import Path
        self.test_dir = tmp_path
        self.config_file = str(tmp_path / "integration_config.json")

        # Import the global config instance
        from episodic.config import config
        self.global_config = config

        # Store original config file path and data
        self.original_file = self.global_config.config_file
        self.original_config = dict(self.global_config.config)

        # Temporarily switch to test config file
        self.global_config.config_file = Path(self.config_file)
        self.global_config.config = self.global_config._template_defaults.copy()
        self.global_config._save()

        yield

        # Restore original config
        self.global_config.config_file = self.original_file
        self.global_config.config = self.original_config

    def test_global_config_usage(self):
        """Test usage of the global config instance."""
        # Set value using global config (runtime only)
        self.global_config.set("global_test", "global_value")

        # Verify it's accessible at runtime
        assert self.global_config.get("global_test") == "global_value"

        # Use save_setting to persist and verify it's written to file
        self.global_config.save_setting("persisted_test", "persisted_value")
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        assert data["persisted_test"] == "persisted_value"

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
            assert self.global_config.get(key) == value
