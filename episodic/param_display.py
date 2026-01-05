"""
Parameter display name mapping for cleaner UI.

This module provides mapping between internal parameter names (canonical)
and user-friendly display names (aliases) for better usability.
"""

# Map from display name (short) to internal name (canonical)
PARAM_ALIASES = {
    # Web search parameters
    "web-enabled": "web-search-enabled",
    "web-provider": "web-search-providers",  # Singular alias maps to plural
    "web-providers": "web-search-providers",
    
    # Topic detection parameters  
    "auto-topics": "automatic-topic-detection",
    "topic-threshold": "min-messages-before-topic-change",
    "sliding-window": "use-sliding-window-detection",
    "hybrid-topics": "use-hybrid-topic-detection",
    
    # Core settings (shorter versions)
    "cost": "show-cost",
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
    # Core settings
    "debug": "Show debug output and verbose information",
    "cost": "Display token costs after each response",
    "model": "Primary LLM model for chat",
    "streaming": "Enable real-time response streaming",
    "depth": "Number of conversation turns to include in context",
    "wrap": "Enable text wrapping for long responses",

    # Topic detection
    "topics": "Show topic transitions in conversation",
    "auto-topics": "Automatically detect topic changes",
    "topic-threshold": "Minimum messages before topic change",

    # Muse mode (web synthesis)
    "muse-mode": "Enable web search synthesis mode",
    "muse-style": "Response length: concise/standard/comprehensive",
    "muse-detail": "Detail level: minimal/moderate/detailed/maximum",

    # RAG / Knowledge base
    "rag-enabled": "Enable knowledge base search",
    "rag-auto": "Auto-search knowledge base for each query",

    # Web search
    "web-enabled": "Enable web search functionality",
    "web-auto": "Auto-enhance responses with web search",
    "web.providers": "Web search providers (comma-separated)",

    # Voice settings
    "voice_stt_provider": "Speech-to-text provider",
    "voice_tts_provider": "Text-to-speech provider",
    "voice_tts_enabled": "Enable text-to-speech output",

    # Display settings
    "color_mode": "Color output: full/basic/none",
    "stream_rate": "Streaming speed (words per second)",
}


def get_canonical_name(param_name: str) -> str:
    """Convert display name to canonical internal name."""
    return PARAM_ALIASES.get(param_name, param_name)


def resolve_param_name(param_name: str) -> str:
    """Resolve a display or alias name to an internal config key."""
    from episodic.param_mappings import normalize_param_name
    return normalize_param_name(get_canonical_name(param_name))


def get_display_name(canonical_name: str) -> str:
    """Convert canonical name to user-friendly display name."""
    return DISPLAY_NAMES.get(canonical_name, canonical_name)


def get_param_description(display_name: str) -> str:
    """Get description for a parameter using its display name."""
    return PARAM_DESCRIPTIONS.get(display_name, "")


def get_all_display_aliases() -> dict:
    """Get all display name aliases for validation."""
    return PARAM_ALIASES.copy()
