import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict
from .config_defaults import CONFIG_DOCS
from .param_mappings import ENV_VAR_MAPPING

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
            self.default_config_file = config_dir / "config.default.json"
        else:
            self.config_file = Path(config_file)
            self.default_config_file = self.config_file.parent / "config.default.json"

        # Get the template defaults first
        self._template_defaults = self._load_template_defaults()
        
        # Initialize or load the configuration
        self.config: Dict[str, Any] = {}
        self._load()
        
        # Ensure config.default.json exists
        self._ensure_default_config()

    def _load_template_defaults(self) -> Dict[str, Any]:
        """Load the default configuration from the template file."""
        # Get the path to the template file (in the same directory as this module)
        template_path = Path(__file__).parent / "config_template.json"
        
        try:
            with open(template_path, 'r') as f:
                template_data = json.load(f)
                # Filter out comment keys that start with "_comment"
                return {k: v for k, v in template_data.items() if not k.startswith('_comment')}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config template: {e}")
            # Fallback to basic defaults if template is missing
            return {
                "debug": False,
                "show_cost": False,
                "stream_responses": True,
                "model": "gpt-4o-mini",
                "topic_detection_model": "ollama/llama3:instruct",
                "context_depth": 5,
                "text_wrap": True,
                "color_mode": "full"
            }

    def _load(self) -> None:
        """Load the configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)

                # Check for any missing defaults and add them in memory only
                for key, value in self._template_defaults.items():
                    if key not in self.config:
                        self.config[key] = value
                        # Don't save - just use defaults in memory to preserve comments
            except json.JSONDecodeError:
                # If the file is corrupted, use defaults but don't overwrite
                print("Warning: Config file is corrupted, using defaults")
                self.config = self._template_defaults.copy()
                # Don't save - let user fix their config file
        else:
            # If the file doesn't exist, create it with default values
            self.config = self._template_defaults.copy()
            self._save()

    def _save(self) -> None:
        """Save the configuration to disk."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def save_setting(self, key: str, value: Any) -> None:
        """Save a specific setting to disk without affecting other runtime values.
        
        This method updates only the specified key in the config file,
        preserving all other settings and avoiding saving runtime-only values.
        """
        # Read the current file config
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            file_config = self._template_defaults.copy()
        
        # Update only the specific key
        file_config[key] = value
        
        # Write back to file
        with open(self.config_file, 'w') as f:
            json.dump(file_config, f, indent=2)
        
        # Also update runtime config
        self.config[key] = value
    
    def _ensure_default_config(self) -> None:
        """Ensure config.default.json exists by copying from template."""
        if not self.default_config_file.exists():
            # Copy the template file to the user's directory
            template_path = Path(__file__).parent / "config_template.json"
            if template_path.exists():
                shutil.copy2(template_path, self.default_config_file)
            else:
                # Fallback: create from template defaults
                with open(self.default_config_file, 'w') as f:
                    json.dump(self._template_defaults, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value, checking environment variables first.

        Args:
            key: The configuration key to get
            default: The default value to return if the key doesn't exist

        Returns:
            The configuration value, or the default if the key doesn't exist
        """
        # Check environment variable first
        if key in ENV_VAR_MAPPING:
            env_value = os.environ.get(ENV_VAR_MAPPING[key])
            if env_value is not None:
                # Convert boolean strings
                if env_value.lower() in ['true', '1', 'yes', 'on']:
                    return True
                elif env_value.lower() in ['false', '0', 'no', 'off']:
                    return False
                # Try to convert numbers
                try:
                    if '.' in env_value:
                        return float(env_value)
                    else:
                        return int(env_value)
                except ValueError:
                    return env_value
        
        # Fall back to config file
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value (runtime only, not persisted).

        Args:
            key: The configuration key to set
            value: The value to set
        """
        # Validate synthesis configuration values (now muse_*)
        if key == 'muse_style' and value not in ['concise', 'standard', 'comprehensive', 'exhaustive']:
            raise ValueError(f"Invalid synthesis style: {value}. Must be one of: concise, standard, comprehensive, exhaustive")
        elif key == 'muse_detail' and value not in ['minimal', 'moderate', 'detailed', 'maximum']:
            raise ValueError(f"Invalid synthesis detail: {value}. Must be one of: minimal, moderate, detailed, maximum")
        elif key == 'muse_format' and value not in ['paragraph', 'bulleted', 'mixed', 'academic']:
            raise ValueError(f"Invalid synthesis format: {value}. Must be one of: paragraph, bulleted, mixed, academic")
        elif key == 'muse_sources' and value not in ['first-only', 'top-three', 'all-relevant', 'selective']:
            raise ValueError(f"Invalid synthesis sources: {value}. Must be one of: first-only, top-three, all-relevant, selective")
        elif key == 'muse_max_tokens' and value is not None:
            try:
                tokens = int(value)
                if tokens < 50 or tokens > 4000:
                    raise ValueError("muse_max_tokens must be between 50 and 4000")
                value = tokens
            except (ValueError, TypeError):
                raise ValueError("muse_max_tokens must be a number or None")
        
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
                if actual_param_set in self._template_defaults:
                    self.config[actual_param_set] = self._template_defaults[actual_param_set].copy()
                    # Don't save - runtime only
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
            
            # Set the parameter in memory only
            self.config[actual_param_set][actual_param_name] = value
            # Don't save - runtime only
        else:
            # Standard configuration setting - runtime only
            self.config[key] = value
            # Don't save - runtime only
            
            # If an API key was set, reload API keys into environment
            if key.endswith('_api_key') or key in ['azure_api_base', 'azure_api_version',
                                                     'bedrock_access_key_id', 'bedrock_secret_access_key',
                                                     'bedrock_region', 'vertex_project', 'vertex_location']:
                # Import here to avoid circular dependency
                from episodic.llm import load_api_keys_from_config
                load_api_keys_from_config()

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
    
    def get_template_defaults(self) -> Dict[str, Any]:
        """Get the template default values.
        
        Returns:
            Dictionary of template default values
        """
        return self._template_defaults.copy()

# Create a global instance for easy access
config = Config()
