import json
from pathlib import Path
from typing import Any, Dict
from .config_defaults import DEFAULT_CONFIG, CONFIG_DOCS

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

                # Check for any missing defaults and add them
                config_changed = False
                for key, value in DEFAULT_CONFIG.items():
                    if key not in self.config:
                        self.config[key] = value
                        config_changed = True
                
                # Save if we added any defaults
                if config_changed:
                    self._save()
            except json.JSONDecodeError:
                # If the file is corrupted, start with defaults
                self.config = DEFAULT_CONFIG.copy()
                self._save()
        else:
            # If the file doesn't exist, create it with default values
            self.config = DEFAULT_CONFIG.copy()
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
        # Handle model parameter syntax like "main.temp" or "topic.temperature"
        if '.' in key:
            parts = key.split('.', 1)
            param_set = parts[0]
            param_name = parts[1]
            
            # Map parameter sets
            param_set_map = {
                'main': 'main_params',
                'topic': 'topic_params',
                'comp': 'compression_params',
                'compression': 'compression_params'
            }
            
            # Map abbreviated parameter names
            param_name_map = {
                'temp': 'temperature',
                'max': 'max_tokens',
                'top': 'top_p',
                'presence': 'presence_penalty',
                'freq': 'frequency_penalty'
            }
            
            # Get the actual parameter set name
            actual_param_set = param_set_map.get(param_set, param_set + '_params')
            
            # Handle reset command
            if param_name == 'reset' or param_name == '*':
                if actual_param_set in DEFAULT_CONFIG:
                    self.config[actual_param_set] = DEFAULT_CONFIG[actual_param_set].copy()
                    self._save()
                return
            
            # Get the actual parameter name
            actual_param_name = param_name_map.get(param_name, param_name)
            
            # Ensure the parameter set exists
            if actual_param_set not in self.config:
                raise ValueError(f"Unknown parameter set: {param_set}")
            
            # Validate parameter values
            if actual_param_name == 'temperature' and not (0 <= float(value) <= 2):
                raise ValueError("Temperature must be between 0 and 2")
            elif actual_param_name == 'top_p' and not (0 <= float(value) <= 1):
                raise ValueError("Top_p must be between 0 and 1")
            elif actual_param_name in ['presence_penalty', 'frequency_penalty'] and not (-2 <= float(value) <= 2):
                raise ValueError(f"{actual_param_name} must be between -2 and 2")
            elif actual_param_name == 'max_tokens' and value is not None and int(value) < 1:
                raise ValueError("max_tokens must be positive or None")
            elif actual_param_name == 'stop':
                if isinstance(value, str):
                    if value == '[]' or value == '':
                        value = []  # Handle empty list
                    else:
                        value = [value]  # Convert single string to list
                elif not isinstance(value, list):
                    raise ValueError("stop must be a string or list of strings")
            
            # Convert numeric values
            if actual_param_name in ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty']:
                value = float(value)
            elif actual_param_name == 'max_tokens' and value is not None:
                value = int(value)
            
            # Set the parameter
            self.config[actual_param_set][actual_param_name] = value
            self._save()
        else:
            # Standard configuration setting
            self.config[key] = value
            self._save()

    def get_model_params(self, param_set: str, model: str = None) -> Dict[str, Any]:
        """Get model parameters for a specific context.
        
        Args:
            param_set: The parameter set name ('main', 'topic', 'compression')
            model: Optional model name to filter parameters for compatibility
        
        Returns:
            Dictionary of model parameters
        """
        param_set_map = {
            'main': 'main_params',
            'topic': 'topic_params',
            'comp': 'compression_params',
            'compression': 'compression_params'
        }
        
        actual_param_set = param_set_map.get(param_set, param_set + '_params')
        
        # Return parameters, filtering out None values
        params = self.config.get(actual_param_set, {})
        filtered_params = {k: v for k, v in params.items() if v is not None}
        
        # Filter out unsupported parameters for specific providers
        if model and 'ollama' in model.lower():
            # Ollama doesn't support presence_penalty or frequency_penalty
            ollama_unsupported = ['presence_penalty', 'frequency_penalty']
            filtered_params = {k: v for k, v in filtered_params.items() 
                             if k not in ollama_unsupported}
        
        return filtered_params

    def delete(self, key: str) -> None:
        """Delete a configuration value.

        Args:
            key: The configuration key to delete
        """
        if key in self.config:
            del self.config[key]
            self._save()
    
    def get_doc(self, key: str) -> str:
        """Get documentation for a configuration key.
        
        Args:
            key: The configuration key
        
        Returns:
            Documentation string or 'No documentation available'
        """
        return CONFIG_DOCS.get(key, "No documentation available")
    
    def list_all(self) -> Dict[str, Any]:
        """Get all configuration values with their documentation.
        
        Returns:
            Dictionary mapping keys to (value, documentation) tuples
        """
        result = {}
        for key, value in self.config.items():
            doc = self.get_doc(key)
            result[key] = (value, doc)
        return result

# Create a global instance for easy access
config = Config()

# Disable hybrid topic detection temporarily to use the standard detection
config.set("use_hybrid_topic_detection", False)
