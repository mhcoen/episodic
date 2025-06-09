import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    def __init__(self, config_file: str = None):
        """Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file. If None, uses the default location.
        """
        if config_file is None:
            # Use default location: ~/.episodic/config.json
            home_dir = Path.home()
            config_dir = home_dir / ".episodic"
            config_dir.mkdir(exist_ok=True)
            self.config_file = config_dir / "config.json"
        else:
            self.config_file = Path(config_file)
        
        # Initialize or load the configuration
        self.config: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load the configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted, start with an empty config
                self.config = {}
        else:
            # If the file doesn't exist, create it with default values
            self.config = {
                "active_prompt": "default"
            }
            self._save()
    
    def _save(self) -> None:
        """Save the configuration to disk."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: The configuration key to get
            default: The default value to return if the key doesn't exist
            
        Returns:
            The configuration value, or the default if the key doesn't exist
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: The configuration key to set
            value: The value to set
        """
        self.config[key] = value
        self._save()

# Create a global instance for easy access
config = Config()