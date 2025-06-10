import os
import tempfile
from pathlib import Path
from episodic.config import Config

def test_debug_default_off():
    """Test that debug mode is off by default."""
    # Create a temporary file path that doesn't exist yet
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp_path = tmp.name

    # Make sure the file doesn't exist
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    try:
        # Create a new Config instance with the temporary file path
        # This should create a new config file with default values
        config = Config(tmp_path)

        # Check that debug is False by default
        debug_value = config.get("debug", None)
        print(f"Debug value: {debug_value}")

        # Verify that debug is explicitly set to False in the config
        assert debug_value is False, f"Expected debug to be False, got {debug_value}"

        # Check the raw config dictionary
        print(f"Raw config: {config.config}")

        # Verify that debug is explicitly set to False in the raw config
        assert "debug" in config.config, "Debug key not found in config"
        assert config.config["debug"] is False, f"Expected config['debug'] to be False, got {config.config['debug']}"

        print("All tests passed! Debug mode is off by default.")
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    test_debug_default_off()
