"""Configuration documentation and legacy support.

This module now contains only documentation strings for configuration values.
The actual default values are stored in config_template.json and managed by the Config class.

DEPRECATED: The hardcoded default dictionaries in this file are no longer used.
Use config_template.json for maintaining default values.
"""

# DEPRECATED - Kept only for backward compatibility during transition
# Use config_template.json for actual defaults
CORE_DEFAULTS = {}
TOPIC_DEFAULTS = {}
STREAMING_DEFAULTS = {}
MODEL_PARAMS_DEFAULTS = {}
RAG_DEFAULTS = {}
WEB_SEARCH_DEFAULTS = {}
CACHE_DEFAULTS = {}
COMPRESSION_DEFAULTS = {}
MODEL_SELECTION_DEFAULTS = {}
LLM_DEFAULTS = {}
DISPLAY_DEFAULTS = {}
DRIFT_EMBEDDING_DEFAULTS = {}

# Combine all defaults - now empty since values come from config_template.json
DEFAULT_CONFIG = {
    **CORE_DEFAULTS,
    **TOPIC_DEFAULTS,
    **STREAMING_DEFAULTS,
    **MODEL_PARAMS_DEFAULTS,
    **CACHE_DEFAULTS,
    **COMPRESSION_DEFAULTS,
    **MODEL_SELECTION_DEFAULTS,
    **LLM_DEFAULTS,
    **DISPLAY_DEFAULTS,
    **RAG_DEFAULTS,
    **WEB_SEARCH_DEFAULTS,
    **DRIFT_EMBEDDING_DEFAULTS
}

# Configuration value documentation
CONFIG_DOCS = {
    # Core settings
    "active_prompt": "The active system prompt to use for conversations",
    "debug": "Enable debug output for troubleshooting",
    "show_cost": "Display token usage and cost information",
    "show_drift": "Show semantic drift scores in topic detection",
    "muse_mode": "Muse mode: treat all input as web search queries (like Perplexity)",
    "response_style": "Global response style for all modes: concise, standard, comprehensive, custom",
    "response_format": "Global response format for all modes: paragraph, bulleted, mixed, academic",
    
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
    "stream_natural_rhythm": "Add natural pauses at punctuation for more human-like streaming",
    "stream_char_mode": "Stream character by character instead of word by word",
    "stream_char_rate": "Characters per second when using character-mode streaming",
    "stream_line_delay": "Delay in seconds between lines when streaming multi-line responses",
    
    # Model parameters
    "main_params": "Model parameters for main conversation",
    "topic_params": "Model parameters for topic detection",
    "compression_params": "Model parameters for topic compression",
    
    # LLM Provider API Keys
    "openai_api_key": "OpenAI API key for GPT models",
    "anthropic_api_key": "Anthropic API key for Claude models",
    "google_api_key": "Google API key for Gemini models",
    "groq_api_key": "Groq API key for fast inference",
    "together_api_key": "Together AI API key for open source models",
    "mistral_api_key": "Mistral AI API key for Mistral models",
    "cohere_api_key": "Cohere API key for Command models",
    "deepseek_api_key": "DeepSeek API key for Chinese language models",
    "deepinfra_api_key": "DeepInfra API key for various open models",
    "perplexity_api_key": "Perplexity API key for online models",
    "fireworks_api_key": "Fireworks AI API key for fast inference",
    "anyscale_api_key": "Anyscale API key for scalable models",
    "replicate_api_key": "Replicate API key for various models",
    "huggingface_api_key": "HuggingFace API key for open models",
    "ai21_api_key": "AI21 Labs API key for Jurassic models",
    "voyage_api_key": "Voyage AI API key for embeddings",
    "openrouter_api_key": "OpenRouter API key for access to multiple providers through one API",
    "azure_api_key": "Azure OpenAI API key",
    "azure_api_base": "Azure OpenAI API base URL",
    "azure_api_version": "Azure OpenAI API version",
    "bedrock_access_key_id": "AWS Bedrock access key ID",
    "bedrock_secret_access_key": "AWS Bedrock secret access key",
    "bedrock_region": "AWS Bedrock region",
    "vertex_project": "Google Vertex AI project ID",
    "vertex_location": "Google Vertex AI location",
    "openrouter_api_base": "OpenRouter API base URL (default: https://openrouter.ai/api/v1)",
    "openrouter_site_url": "Your app's URL for OpenRouter tracking (optional)",
    "openrouter_app_name": "Your app name for OpenRouter tracking (optional)",
    "openrouter_default_models": "List of popular OpenRouter models to show when API key is set",
    
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
    "web_search_providers": "Ordered list of providers for fallback (e.g., ['google', 'duckduckgo'])",
    "web_search_fallback_enabled": "Enable automatic fallback to next provider on errors",
    "web_search_fallback_cache_minutes": "Cache working provider for N minutes after successful search",
    "web_search_auto_enhance": "Automatically search web when local results insufficient",
    "web_search_cache_duration": "How long to cache search results (seconds)",
    "web_search_max_results": "Maximum number of web results to retrieve",
    "web_search_rate_limit": "Maximum searches allowed per hour",
    "web_search_index_results": "Index web search results into RAG for future use",
    "web_search_timeout": "Timeout for web searches (seconds)",
    "web_search_require_confirmation": "Ask user before performing web searches",
    "web_search_excluded_domains": "List of domains to exclude from search results",
    "web_search_show_urls": "Display URLs in search result output",
    "web_search_extract_content": "Extract actual page content from search results for better information",
    "web_search_synthesize": "Synthesize search results into comprehensive answer using LLM",
    "web_show_sources": "Show source URLs when displaying synthesized answers",
    "web_show_raw": "Show raw search results instead of synthesizing (overrides synthesis)",
    
    # Muse mode synthesis configuration  
    "muse_detail": "Detail level: minimal (facts only), moderate (with context), detailed (explanations), maximum (all nuances)",
    "muse_max_tokens": "Direct token limit for synthesis (overrides style setting if specified)",
    "muse_sources": "Source selection: first-only, top-three (default), all-relevant, selective",
    "muse_model": "Specific model for synthesis (None = use main conversation model)",
    "muse_context_depth": "Number of previous messages to include as context in muse mode",
    
    # Provider-specific settings
    "searx_instance_url": "URL of Searx/SearxNG instance to use (default: https://searx.be)",
    "google_api_key": "Google Custom Search API key (or set GOOGLE_API_KEY env var)",
    "google_search_engine_id": "Google Custom Search Engine ID (or set GOOGLE_SEARCH_ENGINE_ID)",
    "bing_api_key": "Bing Search API key from Azure (or set BING_API_KEY env var)",
    "bing_endpoint": "Bing Search API endpoint URL",
    "brave_api_key": "Brave Search API key from https://api.search.brave.com/ (or set BRAVE_API_KEY env var)",
    
    # Caching and performance
    "use_context_cache": "Enable prompt caching to reduce API costs when supported",
    "benchmark": "Enable performance benchmarking for operations",
    "benchmark_display": "Automatically display benchmark results after commands",
    
    # Compression settings
    "compression_model": "LLM model to use for compressing/summarizing topics",
    "compression_min_nodes": "Minimum conversation nodes required before compression is allowed",
    "show_compression_notifications": "Display notifications when topics are automatically compressed",
    
    # Model selection
    "topic_detection_model": "Specific model to use for topic detection (can differ from main model)",
    "model": "Current conversation model (default: gpt-4o-mini, change via /model command)",
    
    # Display settings
    "color_mode": "Terminal color mode: 'none' (no colors), 'basic' (8 colors), or 'full' (256 colors)",
    "text_wrap": "Enable wrapping of long lines to fit terminal width",
    "context_depth": "Default number of previous messages to include in conversation context",
    "show_input_box": "Display user input in a styled box after typing",
    "use_unicode_boxes": "Use Unicode box-drawing characters (disable for basic terminals)",
    "export_directory": "Directory for saving exported markdown files (~ is expanded to home directory)",
    
    # Detection thresholds
    "drift_threshold": "Semantic drift threshold for topic changes (0.9+ indicates topic change)",
    "keyword_threshold": "Keyword-based topic detection threshold",
}

# REMOVED - No special threshold behavior needed
# Topics change when drift score exceeds drift_threshold (default 0.9)