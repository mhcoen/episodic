import os
import glob
from typing import Dict, List, Optional
from .configuration import PROMPT_FILE_EXTENSIONS

# Try to import yaml, but provide a fallback if it's not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML is not installed. YAML frontmatter in prompts will not be parsed.")
    print("To enable full functionality, install PyYAML with: pip install pyyaml")

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.reload()

    def reload(self) -> None:
        """Reload all prompts from disk."""
        self.prompts = {}
        self.metadata = {}

        # Create prompts directory if it doesn't exist
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir)

        # Get all supported prompt files
        prompt_files = []
        for ext in PROMPT_FILE_EXTENSIONS:
            prompt_files.extend(glob.glob(os.path.join(self.prompts_dir, ext)))

        for file_path in prompt_files:
            name = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for YAML frontmatter
            if content.startswith('---') and YAML_AVAILABLE:
                try:
                    # Extract and parse frontmatter
                    _, frontmatter, text = content.split('---', 2)
                    metadata = yaml.safe_load(frontmatter)
                    self.metadata[name] = metadata
                    self.prompts[name] = text.strip()
                except:
                    # If parsing fails, treat the whole file as prompt content
                    self.prompts[name] = content
            else:
                # If YAML is not available or the file doesn't have frontmatter,
                # treat the whole file as prompt content
                self.prompts[name] = content

    def list(self) -> List[str]:
        """Return a list of available prompt names."""
        return list(self.prompts.keys())

    def get(self, name: str) -> Optional[str]:
        """Get a prompt by name."""
        return self.prompts.get(name)

    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a prompt."""
        return self.metadata.get(name)

    def get_active_prompt(self, config_get_func, default: str = "default") -> str:
        """Get the active prompt name from config, or use default if none is set."""
        return config_get_func("active_prompt", default)

    def get_active_prompt_content(self, config_get_func, default: str = "default") -> str:
        """Get the content of the active prompt."""
        active_prompt = self.get_active_prompt(config_get_func, default)
        content = self.get(active_prompt)
        if content is None and active_prompt != default:
            # If the active prompt doesn't exist, fall back to default
            content = self.get(default)
        return content or "You are a helpful assistant."
