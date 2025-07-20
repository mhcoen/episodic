"""
Named debugging system for Episodic.
Allows enabling/disabling debug output for specific subsystems.
"""

from typing import Dict, Set, Optional, List
from episodic.config import config


class DebugSystem:
    """Manages debug settings for different subsystems."""
    
    # Define debug categories and their descriptions
    CATEGORIES = {
        'all': 'All debug output',
        'memory': 'Memory system (RAG, context injection)',
        'topic': 'Topic detection and boundaries',
        'drift': 'Semantic drift calculations',
        'llm': 'LLM queries and responses',
        'rag': 'RAG search and indexing',
        'web': 'Web search operations',
        'stream': 'Streaming output control',
        'format': 'Text formatting and display',
        'db': 'Database operations',
        'cache': 'Caching operations',
        'cost': 'Token usage and costs',
        'perf': 'Performance benchmarking',
    }
    
    # Map old debug areas to new categories
    LEGACY_MAPPINGS = {
        'print': 'format',
        'streaming': 'stream',
        'benchmark': 'perf',
        'token': 'cost',
    }
    
    def __init__(self):
        """Initialize the debug system."""
        self._enabled_categories: Set[str] = set()
        self._load_from_config()
    
    def _load_from_config(self):
        """Load debug settings from config."""
        # Check legacy debug setting
        if config.get("debug", False):
            self._enabled_categories.add('all')
        
        # Load named debug settings
        debug_settings = config.get("debug_categories", {})
        if isinstance(debug_settings, dict):
            for category, enabled in debug_settings.items():
                if enabled:
                    self._enabled_categories.add(category)
        elif isinstance(debug_settings, list):
            # Support list format for backward compatibility
            self._enabled_categories.update(debug_settings)
    
    def is_enabled(self, category: str) -> bool:
        """Check if a debug category is enabled."""
        # Map legacy names
        category = self.LEGACY_MAPPINGS.get(category, category)
        
        # Check if category exists
        if category not in self.CATEGORIES and category != 'all':
            return False
        
        # 'all' enables everything
        if 'all' in self._enabled_categories:
            return True
        
        return category in self._enabled_categories
    
    def enable(self, category: str) -> bool:
        """Enable a debug category."""
        # Map legacy names
        category = self.LEGACY_MAPPINGS.get(category, category)
        
        # Validate category
        if category not in self.CATEGORIES:
            return False
        
        self._enabled_categories.add(category)
        self._save_to_config()
        return True
    
    def disable(self, category: str) -> bool:
        """Disable a debug category."""
        # Map legacy names
        category = self.LEGACY_MAPPINGS.get(category, category)
        
        if category == 'all':
            # 'all' clears everything
            self._enabled_categories.clear()
        else:
            self._enabled_categories.discard(category)
        
        self._save_to_config()
        return True
    
    def set_only(self, categories: List[str]) -> None:
        """Enable only the specified categories, disabling all others."""
        self._enabled_categories.clear()
        for cat in categories:
            cat = self.LEGACY_MAPPINGS.get(cat, cat)
            if cat in self.CATEGORIES:
                self._enabled_categories.add(cat)
        self._save_to_config()
    
    def toggle(self, category: str) -> bool:
        """Toggle a debug category."""
        if self.is_enabled(category):
            self.disable(category)
            return False
        else:
            self.enable(category)
            return True
    
    def get_enabled(self) -> List[str]:
        """Get list of enabled categories."""
        return sorted(list(self._enabled_categories))
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all categories."""
        status = {}
        for cat in self.CATEGORIES:
            status[cat] = self.is_enabled(cat)
        return status
    
    def _save_to_config(self):
        """Save debug settings to config."""
        # Convert to dict format
        debug_dict = {cat: True for cat in self._enabled_categories}
        config.set("debug_categories", debug_dict)
        
        # Update legacy debug flag
        config.set("debug", 'all' in self._enabled_categories)


# Global instance
debug_system = DebugSystem()


def debug_enabled(category: str) -> bool:
    """Check if a debug category is enabled."""
    return debug_system.is_enabled(category)


def debug_print(message: str, category: str = 'all', **kwargs):
    """Print a debug message if the category is enabled."""
    if debug_enabled(category):
        from episodic.color_utils import secho_color
        from episodic.configuration import get_system_color
        
        # Add category prefix if not 'all'
        if category != 'all':
            message = f"[{category.upper()}] {message}"
        
        # Default to system color for debug
        if 'fg' not in kwargs:
            kwargs['fg'] = get_system_color()
        
        secho_color(f"  [DEBUG] {message}", **kwargs)


def debug_set(setting: str) -> str:
    """
    Handle debug settings from /set command.
    
    Examples:
        /set debug true          -> Enable all
        /set debug false         -> Disable all
        /set debug memory        -> Enable only memory
        /set debug memory,topic  -> Enable memory and topic
        /set debug off           -> Disable all
    """
    setting = setting.lower().strip()
    
    # Handle boolean values
    if setting in ('true', 'on', '1', 'yes'):
        debug_system.enable('all')
        return "Debug enabled for all categories"
    elif setting in ('false', 'off', '0', 'no'):
        debug_system.disable('all')
        return "Debug disabled for all categories"
    
    # Handle category lists
    categories = [cat.strip() for cat in setting.split(',')]
    valid_cats = []
    invalid_cats = []
    
    for cat in categories:
        if cat in debug_system.CATEGORIES or cat in debug_system.LEGACY_MAPPINGS:
            valid_cats.append(debug_system.LEGACY_MAPPINGS.get(cat, cat))
        else:
            invalid_cats.append(cat)
    
    if invalid_cats:
        return f"Invalid debug categories: {', '.join(invalid_cats)}"
    
    # Enable only the specified categories
    debug_system.set_only(valid_cats)
    return f"Debug enabled for: {', '.join(valid_cats)}"


def format_debug_status() -> str:
    """Format the debug status for display."""
    lines = ["Debug Categories:"]
    status = debug_system.get_status()
    
    for cat, desc in debug_system.CATEGORIES.items():
        enabled = status.get(cat, False)
        marker = "✓" if enabled else "✗"
        lines.append(f"  {marker} {cat:<10} - {desc}")
    
    enabled = debug_system.get_enabled()
    if enabled:
        lines.append(f"\nEnabled: {', '.join(enabled)}")
    else:
        lines.append("\nAll debug output disabled")
    
    return "\n".join(lines)