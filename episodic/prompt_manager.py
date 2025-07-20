import os
import glob
import shutil
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
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            # Use user's episodic directory
            self.prompts_dir = os.path.expanduser("~/.episodic/prompts")
        else:
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
            # Copy default prompts on first run
            self._copy_default_prompts()

        # Get all supported prompt files (including subdirectories)
        prompt_files = []
        for ext in PROMPT_FILE_EXTENSIONS:
            # Search in root directory
            prompt_files.extend(glob.glob(os.path.join(self.prompts_dir, ext)))
            # Search in subdirectories
            prompt_files.extend(glob.glob(os.path.join(self.prompts_dir, "**", ext), recursive=True))

        for file_path in prompt_files:
            # Get relative path from prompts_dir and remove extension
            rel_path = os.path.relpath(file_path, self.prompts_dir)
            name = os.path.splitext(rel_path)[0]
            # Convert path separators to forward slashes for consistency
            name = name.replace(os.sep, '/')
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
        return self.metadata.get(name, {} if name in self.prompts else None)

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
    
    def _copy_default_prompts(self):
        """Copy default prompts from project directory to user directory."""
        # Get the project prompts directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_prompts_dir = os.path.join(project_root, "prompts")
        
        if not os.path.exists(default_prompts_dir):
            return
        
        # Copy all files and subdirectories
        for root, dirs, files in os.walk(default_prompts_dir):
            # Get relative path from default_prompts_dir
            rel_path = os.path.relpath(root, default_prompts_dir)
            
            # Create corresponding directory in user's prompts
            if rel_path == ".":
                target_dir = self.prompts_dir
            else:
                target_dir = os.path.join(self.prompts_dir, rel_path)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Copy files
            for file in files:
                if any(file.endswith(ext.replace("*", "")) for ext in PROMPT_FILE_EXTENSIONS):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_dir, file)
                    
                    # Only copy if destination doesn't exist (don't overwrite user customizations)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)


# Global instance for convenience
_prompt_manager = None

def get_prompt_manager():
    """Get or create the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# Convenience functions for backward compatibility
def get_available_prompts() -> List[str]:
    """Get list of available prompt names."""
    return get_prompt_manager().list()


def load_prompt(name: str) -> Optional[Dict]:
    """Load a prompt by name, returning content and metadata."""
    pm = get_prompt_manager()
    content = pm.get(name)
    if content is None:
        return None
    
    metadata = pm.get_metadata(name) or {}
    return {
        'content': content,
        **metadata
    }


def get_active_prompt() -> str:
    """Get the active prompt name from config."""
    from .config import config
    return config.get("active_prompt", "default")
