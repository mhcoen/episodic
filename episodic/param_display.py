"""
Parameter display name mapping for cleaner UI.

This module provides mapping between internal parameter names (canonical)
and user-friendly display names (aliases) for better usability.
"""

# Map from display name (short) to internal name (canonical)
PARAM_ALIASES = {
    # Web search parameters
    "web-enabled": "web-search-enabled",
    "web-provider": "web-search-provider",
    
    # Topic detection parameters  
    "auto-topics": "automatic-topic-detection",
    "topic-threshold": "min-messages-before-topic-change",
    "sliding-window": "use-sliding-window-detection",
    "hybrid-topics": "use-hybrid-topic-detection",
    
    # Core settings (shorter versions)
    "cost": "show-costs",
    "streaming": "stream-responses", 
    "wrap": "text-wrap",
    "topics": "show-topics",
    
    # Muse mode (already reasonable, keep same)
    "muse-style": "muse-style",
    "muse-detail": "muse-detail",
    "muse-format": "muse-format",
    "muse-max-tokens": "muse-max-tokens",
    "muse-sources": "muse-sources",
    "muse-model": "muse-model",
    
    # RAG parameters
    "rag-enabled": "rag-enabled",
    "rag-auto": "rag-auto-search",
    "rag-results": "rag-max-results",
    
    # Advanced topic detection
    "drift-threshold": "drift-threshold",
    "drift-model": "drift-embedding-model",
    
    # Other core settings
    "debug": "debug",
    "depth": "context-depth",
    "cache": "use-context-cache",
    "benchmark": "benchmark",
}

# Reverse map for display purposes (internal -> display)
DISPLAY_NAMES = {v: k for k, v in PARAM_ALIASES.items()}

# Parameter descriptions for help
PARAM_DESCRIPTIONS = {
    "debug": "Show debug output and verbose information",
    "cost": "Display token costs after each response",
    "topics": "Show topic transitions in conversation",
    "streaming": "Enable real-time response streaming",
    "muse-mode": "Enable web search synthesis mode",
    "muse-style": "Response length (concise/standard/comprehensive/exhaustive)",
    "muse-detail": "Detail level (minimal/moderate/detailed/maximum)",
    "depth": "Number of conversation turns to include in context",
    "wrap": "Enable text wrapping for long responses",
    "rag-enabled": "Enable knowledge base search integration",
    "web-enabled": "Enable web search functionality",
    "auto-topics": "Automatically detect topic changes in conversation",
    "topic-threshold": "Minimum messages before allowing topic change",
}


def get_canonical_name(param_name: str) -> str:
    """Convert display name to canonical internal name."""
    return PARAM_ALIASES.get(param_name, param_name)


def get_display_name(canonical_name: str) -> str:
    """Convert canonical name to user-friendly display name."""
    return DISPLAY_NAMES.get(canonical_name, canonical_name)


def get_param_description(display_name: str) -> str:
    """Get description for a parameter using its display name."""
    return PARAM_DESCRIPTIONS.get(display_name, "")


def get_all_display_aliases() -> dict:
    """Get all display name aliases for validation."""
    return PARAM_ALIASES.copy()