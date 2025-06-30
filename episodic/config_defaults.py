"""Default configuration values for Episodic.

This module centralizes all default configuration values, making it easier to:
1. See all available settings in one place
2. Understand what each setting does
3. Maintain consistency across the codebase
4. Avoid hardcoded values scattered throughout the code
"""

# Core application settings
CORE_DEFAULTS = {
    "active_prompt": "default",
    "debug": False,
    "show_cost": False,
    "show_drift": True,
}

# Topic detection settings
TOPIC_DEFAULTS = {
    # Automatic topic detection
    "automatic_topic_detection": True,
    "auto_compress_topics": True,
    "min_messages_before_topic_change": 8,  # Minimum user messages before allowing topic change
    "running_topic_guess": True,  # Show topic predictions in responses (not yet implemented)
    "show_topics": False,  # Show topic evolution in responses
    
    # Topic boundary analysis
    "analyze_topic_boundaries": True,  # Analyze recent messages to find actual topic transition
    "use_llm_boundary_analysis": True,  # Use LLM for boundary analysis (vs heuristics)
    
    # Manual indexing
    "manual_index_window_size": 3,  # Default window size for /index command
    "manual_index_threshold": 0.75,  # Drift score threshold for boundary detection
    
    # Hybrid topic detection (deprecated, keeping for compatibility)
    "use_hybrid_topic_detection": False,
    "hybrid_topic_weights": {
        "semantic_drift": 0.6,
        "keyword_explicit": 0.25,
        "keyword_domain": 0.1,
        "message_gap": 0.025,
        "conversation_flow": 0.025
    },
    "hybrid_topic_threshold": 0.55,
    "hybrid_llm_threshold": 0.3,
}

# Streaming settings
STREAMING_DEFAULTS = {
    "stream_responses": True,
    "stream_rate": 15,  # Words per second for constant-rate streaming
    "stream_constant_rate": False,  # Whether to use constant-rate streaming
}

# Model parameter defaults for different contexts
MODEL_PARAMS_DEFAULTS = {
    "main_params": {
        "temperature": 0.7,
        "max_tokens": None,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop": []
    },
    "topic_params": {
        "temperature": 0.3,  # Lower for more consistent topic detection
        "max_tokens": 50,   # Topic detection needs minimal tokens
        "top_p": 0.9,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop": []
    },
    "compression_params": {
        "temperature": 0.5,  # Balanced for accurate summaries
        "max_tokens": 500,   # Summaries need more tokens
        "top_p": 1.0,
        "presence_penalty": 0.1,  # Slight penalty to avoid repetition
        "frequency_penalty": 0.1,
        "stop": []
    }
}

# Combine all defaults
DEFAULT_CONFIG = {
    **CORE_DEFAULTS,
    **TOPIC_DEFAULTS,
    **STREAMING_DEFAULTS,
    **MODEL_PARAMS_DEFAULTS
}

# Configuration value documentation
CONFIG_DOCS = {
    # Core settings
    "active_prompt": "The active system prompt to use for conversations",
    "debug": "Enable debug output for troubleshooting",
    "show_cost": "Display token usage and cost information",
    "show_drift": "Show semantic drift scores in topic detection",
    
    # Topic detection
    "automatic_topic_detection": "Enable automatic topic detection during conversations",
    "auto_compress_topics": "Automatically compress topics when they end",
    "min_messages_before_topic_change": "Minimum user messages required before allowing a topic change",
    "running_topic_guess": "Show ongoing topic predictions (not yet implemented)",
    "show_topics": "Display topic evolution in responses",
    "analyze_topic_boundaries": "Analyze recent messages to find precise topic boundaries",
    "use_llm_boundary_analysis": "Use LLM for boundary analysis (more accurate than heuristics)",
    "manual_index_window_size": "Default sliding window size for manual topic indexing",
    "manual_index_threshold": "Drift score threshold for detecting topic boundaries",
    
    # Hybrid detection (deprecated)
    "use_hybrid_topic_detection": "Use multi-signal topic detection (deprecated)",
    "hybrid_topic_weights": "Weight distribution for hybrid detection signals",
    "hybrid_topic_threshold": "Overall threshold for hybrid topic change detection",
    "hybrid_llm_threshold": "Threshold below which to use LLM fallback",
    
    # Streaming
    "stream_responses": "Enable streaming responses for better UX",
    "stream_rate": "Words per second for constant-rate streaming",
    "stream_constant_rate": "Use constant rate streaming instead of token-based",
    
    # Model parameters
    "main_params": "Model parameters for main conversation",
    "topic_params": "Model parameters for topic detection",
    "compression_params": "Model parameters for topic compression",
}

# Dynamic threshold behavior (from topics.py)
TOPIC_THRESHOLD_BEHAVIOR = {
    "description": "First 2 topics use half the configured threshold",
    "first_n_topics": 2,
    "reduced_threshold_factor": 0.5
}