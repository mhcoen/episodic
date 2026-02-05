"""
Unit tests for configuration management.

Tests the config module and CLI integration.
"""

import pytest
import os
import json

from episodic.config import Config


@pytest.fixture
def config_instance(tmp_path):
    """Create a Config instance with a temporary config file."""
    config_file = tmp_path / "test_config.json"
    return Config(str(config_file)), config_file


class TestConfig:
    """Test Config class functionality."""

    def test_initial_config_creation(self, config_instance):
        """Test that config file is created if it doesn't exist."""
        config, config_file = config_instance

        assert config_file.exists()

        with open(config_file, 'r') as f:
            data = json.load(f)

        expected_defaults = config.get_template_defaults()
        assert data == expected_defaults

    def test_set_and_get_values(self, config_instance):
        """Test setting and getting configuration values."""
        config, _ = config_instance

        config.set("test_key", "test_value")
        assert config.get("test_key") == "test_value"

        config.set("bool_key", True)
        config.set("int_key", 42)
        config.set("float_key", 3.14)
        config.set("list_key", [1, 2, 3])
        config.set("dict_key", {"nested": "value"})

        assert config.get("bool_key") is True
        assert config.get("int_key") == 42
        assert config.get("float_key") == 3.14
        assert config.get("list_key") == [1, 2, 3]
        assert config.get("dict_key") == {"nested": "value"}

    def test_get_with_default(self, config_instance):
        """Test getting values with default fallback."""
        config, _ = config_instance

        assert config.get("nonexistent", "default") == "default"
        assert config.get("nonexistent") is None

        config.set("existing", "actual")
        assert config.get("existing", "default") == "actual"

    def test_persistence(self, config_instance):
        """Test that configuration persists across instances."""
        config, config_file = config_instance

        config.save_setting("persistent_key", "persistent_value")
        config.save_setting("number", 123)

        new_config = Config(str(config_file))

        assert new_config.get("persistent_key") == "persistent_value"
        assert new_config.get("number") == 123

    def test_file_synchronization(self, config_instance):
        """Test that changes are immediately written to file."""
        config, config_file = config_instance

        config.save_setting("sync_test", "sync_value")

        with open(config_file, 'r') as f:
            data = json.load(f)

        assert data["sync_test"] == "sync_value"

    def test_overwrite_values(self, config_instance):
        """Test overwriting existing values."""
        config, _ = config_instance

        config.set("overwrite_key", "initial")
        assert config.get("overwrite_key") == "initial"

        config.set("overwrite_key", 42)
        assert config.get("overwrite_key") == 42

        config.set("overwrite_key", {"new": "structure"})
        assert config.get("overwrite_key") == {"new": "structure"}

    def test_delete_functionality(self, config_instance):
        """Test deleting configuration values."""
        config, _ = config_instance

        config.set("delete_me", "to_be_deleted")
        assert config.get("delete_me") == "to_be_deleted"

        config.delete("delete_me")
        assert config.get("delete_me") is None

        # Deleting non-existent key should not raise error
        config.delete("never_existed")

    def test_malformed_config_file(self, tmp_path):
        """Test handling of malformed config file."""
        config_file = tmp_path / "malformed.json"
        config_file.write_text("{ invalid json")

        malformed_config = Config(str(config_file))
        assert malformed_config.get("any_key") is None

    @pytest.mark.skipif(os.name != 'posix', reason="Unix-specific test")
    def test_config_file_permissions(self, config_instance):
        """Test behavior when config file has permission issues."""
        config, config_file = config_instance

        os.chmod(config_file, 0o444)
        try:
            config.set("permission_test", "value")
        except PermissionError:
            pass  # Expected behavior
        finally:
            os.chmod(config_file, 0o644)

    def test_nested_configuration(self, config_instance):
        """Test handling of nested configuration structures."""
        config, _ = config_instance

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

        config.set("app_config", nested_config)
        retrieved = config.get("app_config")

        assert retrieved["database"]["host"] == "localhost"
        assert retrieved["database"]["port"] == 5432
        assert retrieved["database"]["credentials"]["username"] == "user"
        assert retrieved["features"]["caching"] is True

    def test_unicode_handling(self, config_instance):
        """Test handling of Unicode characters in config."""
        config, _ = config_instance

        unicode_values = {
            "emoji": "🤖🎯✅",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "mixed": "Hello 世界 🌍"
        }

        for key, value in unicode_values.items():
            config.set(key, value)
            assert config.get(key) == value

    def test_large_configuration(self, config_instance):
        """Test handling of large configuration values."""
        config, _ = config_instance

        large_list = list(range(10000))
        config.set("large_list", large_list)

        retrieved = config.get("large_list")
        assert len(retrieved) == 10000
        assert retrieved[0] == 0
        assert retrieved[-1] == 9999

        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        config.set("large_dict", large_dict)

        retrieved = config.get("large_dict")
        assert len(retrieved) == 1000
        assert retrieved["key_0"] == "value_0"
        assert retrieved["key_999"] == "value_999"


class TestConfigIntegration:
    """Test Config integration with the rest of the system."""

    @pytest.fixture
    def global_config_setup(self, tmp_path):
        """Set up test environment with global config."""
        from episodic.config import config as global_config

        config_file = tmp_path / "integration_config.json"
        original_file = global_config.config_file

        global_config.config_file = str(config_file)
        global_config.config_data = {}
        global_config._save()

        yield global_config, config_file

        global_config.config_file = original_file
        global_config._load()

    def test_global_config_usage(self, global_config_setup):
        """Test usage of the global config instance."""
        global_config, config_file = global_config_setup

        global_config.save_setting("global_test", "global_value")

        assert global_config.get("global_test") == "global_value"

        with open(config_file, 'r') as f:
            data = json.load(f)
        assert data["global_test"] == "global_value"

    def test_cli_configuration_values(self, global_config_setup):
        """Test configuration values used by CLI."""
        global_config, _ = global_config_setup

        cli_configs = {
            "debug": False,
            "show_cost": True,
            "use_context_cache": True,
            "active_prompt": "default",
            "history_file": "~/.episodic_history"
        }

        for key, value in cli_configs.items():
            global_config.set(key, value)
            assert global_config.get(key) == value
