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
    "analyze_topic_boundaries": False,  # Disabled - causes incorrect boundary placement
    "use_llm_boundary_analysis": False,  # Use LLM for boundary analysis (vs heuristics)
    
    # Manual indexing
    "manual_index_window_size": 3,  # Default window size for /index command
    "manual_index_threshold": 0.75,  # Drift score threshold for boundary detection
    
    # Hybrid topic detection
    "use_hybrid_topic_detection": False,
    "use_sliding_window_detection": True,  # Use sliding 3-window detection by default
    "sliding_window_size": 3,  # 3-3 window size
    "hybrid_topic_weights": {
        "semantic_drift": 0.6,
        "keyword_explicit": 0.25,
        "keyword_domain": 0.1,
        "message_gap": 0.025,
        "conversation_flow": 0.025
    },
    "hybrid_topic_threshold": 0.55,
    "hybrid_llm_threshold": 0.3,
    "drift_threshold": 0.9,  # Threshold for sliding window detection (0.9+ = topic change)
    "keyword_threshold": 0.5,
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

# RAG (Retrieval Augmented Generation) settings
RAG_DEFAULTS = {
    "rag_enabled": False,  # Enable RAG for enhanced responses with external knowledge
    "rag_auto_search": True,  # Automatically search knowledge base for each user message
    "rag_search_threshold": 0.7,  # Minimum relevance score for including search results
    "rag_max_results": 5,  # Maximum number of search results to include
    "rag_embedding_model": "all-MiniLM-L6-v2",  # Sentence transformer model for embeddings
    "rag_include_citations": True,  # Include source citations in responses
    "rag_context_prefix": "\n\n[Relevant context from knowledge base]:\n",  # Prefix for RAG context in prompts
    "rag_chunk_size": 500,  # Number of words per document chunk
    "rag_chunk_overlap": 100,  # Number of overlapping words between chunks
    "rag_max_file_size": 10 * 1024 * 1024,  # Maximum file size for indexing (10MB)
    "rag_show_citations": True,  # Show which documents were used in responses
    "rag_citation_style": "inline",  # How to display citations: 'inline' or 'footnote'
    "rag_allowed_file_types": [".txt", ".md", ".pdf", ".rst"],  # Allowed file extensions for indexing
}

# Web search settings
WEB_SEARCH_DEFAULTS = {
    "web_search_enabled": False,  # Enable web search functionality
    "web_search_provider": "duckduckgo",  # Search provider: duckduckgo, searx, google, bing
    "web_search_auto_enhance": False,  # Auto-search when no good local results
    "web_search_cache_duration": 3600,  # Cache search results for 1 hour
    "web_search_max_results": 5,  # Maximum results per search
    "web_search_rate_limit": 60,  # Maximum searches per hour
    "web_search_index_results": True,  # Index web results into RAG for future use
    "web_search_timeout": 10,  # Search timeout in seconds
    "web_search_require_confirmation": False,  # Ask before performing searches
    "web_search_excluded_domains": [],  # Domains to exclude from results
    "web_search_show_urls": True,  # Show URLs in search results
    
    # Provider-specific settings
    "searx_instance_url": "https://searx.be",  # Searx instance URL (can be self-hosted)
    
    # Google Custom Search (requires setup at https://developers.google.com/custom-search)
    "google_api_key": None,  # Or set GOOGLE_API_KEY env var
    "google_search_engine_id": None,  # Or set GOOGLE_SEARCH_ENGINE_ID env var
    
    # Bing Search (requires Azure Cognitive Services)
    "bing_api_key": None,  # Or set BING_API_KEY env var
    "bing_endpoint": "https://api.bing.microsoft.com/v7.0/search",
}

# Combine all defaults
DEFAULT_CONFIG = {
    **CORE_DEFAULTS,
    **TOPIC_DEFAULTS,
    **STREAMING_DEFAULTS,
    **MODEL_PARAMS_DEFAULTS,
    **RAG_DEFAULTS,
    **WEB_SEARCH_DEFAULTS
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
    
    # Topic detection methods
    "use_hybrid_topic_detection": "Use multi-signal topic detection (combines drift, keywords, etc)",
    "use_sliding_window_detection": "Use sliding window detection (compares groups of messages)",
    "sliding_window_size": "Size of sliding windows for topic detection (default: 3)",
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
    
    # RAG settings
    "rag_enabled": "Enable RAG for enhanced responses with external knowledge",
    "rag_auto_search": "Automatically search knowledge base for each user message",
    "rag_search_threshold": "Minimum relevance score for including search results",
    "rag_max_results": "Maximum number of search results to include in context",
    "rag_embedding_model": "Sentence transformer model for embeddings",
    "rag_include_citations": "Include source citations in responses",
    "rag_context_prefix": "Prefix added before RAG context in prompts",
    "rag_chunk_size": "Number of words per document chunk",
    "rag_chunk_overlap": "Number of overlapping words between chunks",
    "rag_max_file_size": "Maximum file size for indexing (in bytes)",
    "rag_show_citations": "Show which documents were used in responses",
    "rag_citation_style": "How to display citations: 'inline' or 'footnote'",
    "rag_allowed_file_types": "List of allowed file extensions for indexing",
    
    # Web search settings
    "web_search_enabled": "Enable web search functionality for current information",
    "web_search_provider": "Search provider to use (duckduckgo, searx, google, bing)",
    "web_search_auto_enhance": "Automatically search web when local results insufficient",
    "web_search_cache_duration": "How long to cache search results (seconds)",
    "web_search_max_results": "Maximum number of web results to retrieve",
    "web_search_rate_limit": "Maximum searches allowed per hour",
    "web_search_index_results": "Index web search results into RAG for future use",
    "web_search_timeout": "Timeout for web searches (seconds)",
    "web_search_require_confirmation": "Ask user before performing web searches",
    "web_search_excluded_domains": "List of domains to exclude from search results",
    "web_search_show_urls": "Display URLs in search result output",
    
    # Provider-specific settings
    "searx_instance_url": "URL of Searx/SearxNG instance to use (default: https://searx.be)",
    "google_api_key": "Google Custom Search API key (or set GOOGLE_API_KEY env var)",
    "google_search_engine_id": "Google Custom Search Engine ID (or set GOOGLE_SEARCH_ENGINE_ID)",
    "bing_api_key": "Bing Search API key from Azure (or set BING_API_KEY env var)",
    "bing_endpoint": "Bing Search API endpoint URL",
}

# REMOVED - No special threshold behavior needed
# Topics change when drift score exceeds drift_threshold (default 0.9)