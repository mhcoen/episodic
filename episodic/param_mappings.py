"""
Parameter name mappings for CLI usability.

This module provides mappings between user-friendly CLI parameter names
(using dashes) and internal configuration names (using underscores).
"""

# Parameter name mapping from CLI (dash) to internal (underscore)
PARAM_MAPPING = {
    # Display settings
    'stream-rate': 'stream_rate',
    'stream-constant-rate': 'stream_constant_rate', 
    'stream-natural-rhythm': 'stream_natural_rhythm',
    'stream-char-mode': 'stream_char_mode',
    'stream-char-rate': 'stream_char_rate',
    'stream-line-delay': 'stream_line_delay',
    'text-wrap': 'text_wrap',
    'show-cost': 'show_cost',
    'show-topics': 'show_topics',
    'show-drift': 'show_drift',
    'hybrid-topics': 'use_hybrid_topic_detection',
    'color-mode': 'color_mode',
    
    # Topic detection
    'automatic-topic-detection': 'automatic_topic_detection',
    'topic-detection-model': 'topic_detection_model',
    'auto-compress-topics': 'auto_compress_topics',
    'compression-model': 'compression_model',
    'compression-min-nodes': 'compression_min_nodes',
    'show-compression-notifications': 'show_compression_notifications',
    'min-messages-before-topic-change': 'min_messages_before_topic_change',
    'use-hybrid-topic-detection': 'use_hybrid_topic_detection',
    'analyze-topic-boundaries': 'analyze_topic_boundaries',
    'use-llm-boundary-analysis': 'use_llm_boundary_analysis',
    'use-sliding-window-detection': 'use_sliding_window_detection',
    'sliding-window-size': 'sliding_window_size',
    'drift-threshold': 'drift_threshold',
    
    # RAG settings
    'rag-enabled': 'rag_enabled',
    'rag-auto-search': 'rag_auto_search',
    'rag-search-threshold': 'rag_search_threshold',
    'rag-max-results': 'rag_max_results',
    'rag-chunk-size': 'rag_chunk_size',
    'rag-chunk-overlap': 'rag_chunk_overlap',
    
    # Web search settings
    'web-search-enabled': 'web_search_enabled',
    'web-search-provider': 'web_search_provider',
    'web-search-auto-enhance': 'web_search_auto_enhance',
    'web-search-max-results': 'web_search_max_results',
    'web-search-synthesize': 'web_search_synthesize',
    'web-search-extract-content': 'web_search_extract_content',
    'web-search-rate-limit': 'web_search_rate_limit',
    'web-search-cache-ttl': 'web_search_cache_ttl',
    'web-show-sources': 'web_show_sources',
    'google-api-key': 'google_api_key',
    'google-search-engine-id': 'google_search_engine_id',
    'bing-api-key': 'bing_api_key',
    'searx-instance-url': 'searx_instance_url',
    
    # Other settings
    'benchmark-display': 'benchmark_display',
    'use-context-cache': 'use_context_cache',
    'active-prompt': 'active_prompt',
    'running-topic-guess': 'running_topic_guess',
}

# Short aliases for common parameters
SHORT_ALIASES = {
    # Web search
    'web-provider': 'web_search_provider',
    'web-enabled': 'web_search_enabled',
    'web-auto': 'web_search_auto_enhance',
    'web-results': 'web_search_max_results',
    'web-synthesize': 'web_search_synthesize',
    'web-extract': 'web_search_extract_content',
    'web-rate-limit': 'web_search_rate_limit',
    'web-cache-ttl': 'web_search_cache_ttl',
    
    # RAG
    'rag': 'rag_enabled',
    'rag-auto': 'rag_auto_search',
    'rag-threshold': 'rag_search_threshold',
    'rag-results': 'rag_max_results',
    'rag-chunk': 'rag_chunk_size',
    'rag-overlap': 'rag_chunk_overlap',
    
    # Topic detection
    'topic-model': 'topic_detection_model',
    'topic-min': 'min_messages_before_topic_change',
    'topic-auto': 'automatic_topic_detection',
    'topic-boundaries': 'analyze_topic_boundaries',
    'topic-llm-analysis': 'use_llm_boundary_analysis',
    'topic-window': 'use_sliding_window_detection',
    'window-size': 'sliding_window_size',
    'drift': 'drift_threshold',
    
    # Compression
    'comp-model': 'compression_model',
    'comp-min': 'compression_min_nodes',
    'comp-auto': 'auto_compress_topics',
    'comp-notify': 'show_compression_notifications',
    
    # Display
    'wrap': 'text_wrap',
    'cost': 'show_cost',
    'topics': 'show_topics',
    'color': 'color_mode',
    'stream': 'stream_responses',
    'stream-rate': 'stream_rate',
    'stream-constant': 'stream_constant_rate',
    'stream-natural': 'stream_natural_rhythm',
    'stream-char': 'stream_char_mode',
    
    # Core settings
    'cache': 'use_context_cache',
    'debug': 'debug',
    'benchmark': 'benchmark',
    'prompt': 'active_prompt',
}

# Environment variable mapping
ENV_VAR_MAPPING = {
    # Web search
    'web_search_provider': 'EPISODIC_WEB_PROVIDER',
    'web_search_enabled': 'EPISODIC_WEB_ENABLED',
    'web_search_auto_enhance': 'EPISODIC_WEB_AUTO',
    'web_search_max_results': 'EPISODIC_WEB_RESULTS',
    'google_api_key': 'GOOGLE_API_KEY',
    'google_search_engine_id': 'GOOGLE_SEARCH_ENGINE_ID',
    'bing_api_key': 'BING_API_KEY',
    'searx_instance_url': 'SEARX_INSTANCE_URL',
    
    # RAG
    'rag_enabled': 'EPISODIC_RAG_ENABLED',
    'rag_auto_search': 'EPISODIC_RAG_AUTO',
    'rag_search_threshold': 'EPISODIC_RAG_THRESHOLD',
    'rag_max_results': 'EPISODIC_RAG_RESULTS',
    
    # Topic detection
    'topic_detection_model': 'EPISODIC_TOPIC_MODEL',
    'automatic_topic_detection': 'EPISODIC_TOPIC_AUTO',
    'min_messages_before_topic_change': 'EPISODIC_TOPIC_MIN',
    
    # Compression
    'compression_model': 'EPISODIC_COMPRESSION_MODEL',
    'auto_compress_topics': 'EPISODIC_COMPRESS_AUTO',
    'compression_min_nodes': 'EPISODIC_COMPRESS_MIN',
    
    # Display
    'stream_rate': 'EPISODIC_STREAM_RATE',
    'color_mode': 'EPISODIC_COLOR_MODE',
    'show_cost': 'EPISODIC_SHOW_COST',
    'show_topics': 'EPISODIC_SHOW_TOPICS',
    
    # Core settings
    'debug': 'EPISODIC_DEBUG',
    'use_context_cache': 'EPISODIC_CACHE',
    'active_prompt': 'EPISODIC_PROMPT',
}

def normalize_param_name(param: str) -> str:
    """Convert CLI parameter name to internal name.
    
    Args:
        param: Parameter name from CLI (may have dashes or be a short alias)
        
    Returns:
        Internal parameter name with underscores
    """
    # First check if it's a short alias
    if param in SHORT_ALIASES:
        return SHORT_ALIASES[param]
    
    # Then check parameter mapping
    if param in PARAM_MAPPING:
        return PARAM_MAPPING[param]
    
    # If it has dashes, try converting to underscores
    if '-' in param:
        underscore_version = param.replace('-', '_')
        return underscore_version
    
    # Return as-is if no mapping found
    return param

def get_display_name(internal_name: str) -> str:
    """Get the display name (with dashes) for an internal parameter name.
    
    Args:
        internal_name: Internal parameter name with underscores
        
    Returns:
        Display name with dashes, or original if no mapping exists
    """
    # Check if there's a short alias
    for alias, internal in SHORT_ALIASES.items():
        if internal == internal_name:
            return alias
    
    # Check regular mappings
    for cli_name, internal in PARAM_MAPPING.items():
        if internal == internal_name:
            return cli_name
    
    # Convert underscores to dashes as fallback
    return internal_name.replace('_', '-')