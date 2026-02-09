"""
Centralized constants for Episodic.

All hardcoded lists that are used in multiple places should be defined here
to ensure consistency and make it easy to add new options.
"""

# Web search providers
WEB_SEARCH_PROVIDERS = ['duckduckgo', 'google', 'bing', 'brave', 'searx']

# Response styles for /style command
RESPONSE_STYLES = ['concise', 'standard', 'comprehensive', 'custom']

# Response formats for /format command
RESPONSE_FORMATS = ['paragraph', 'bulleted', 'mixed', 'academic']

# Detail levels for /detail command
DETAIL_LEVELS = ['minimal', 'moderate', 'detailed', 'maximum']

# Topic subcommands
TOPIC_ACTIONS = ['list', 'rename', 'compress', 'index', 'scores', 'stats', 'reanalyze', 'delete']

# Topic granularity levels for neural segmentation
TOPIC_GRANULARITY_LEVELS = ['fine', 'medium', 'coarse']

# Compression subcommands
COMPRESSION_ACTIONS = ['stats', 'queue', 'compress', 'api-stats', 'reset-api']

# Voice subcommands
VOICE_ACTIONS = ['on', 'off', 'status', 'stt', 'tts', 'info']

# RAG subcommands
RAG_ACTIONS = ['on', 'off', 'stats']

# Docs subcommands
DOCS_ACTIONS = ['list', 'show', 'remove', 'rm', 'clear']

# Prompt subcommands
PROMPT_ACTIONS = ['list', 'use', 'show']

# Reset options
RESET_ACTIONS = ['all']

# KG (knowledge graph) subcommands
KG_ACTIONS = [
    'status', 'visualize', 'entities', 'entity', 'edges',
    'search', 'update', 'rebuild', 'skip', 'patch', 'stats',
]

# Dev subcommands
DEV_ACTIONS = ['reindex-help']

# Migrate subcommands
MIGRATE_ACTIONS = ['run', 'dry-run', 'rollback']

# Summary length options
SUMMARY_LENGTHS = ['brief', 'short', 'standard', 'detailed', 'bulleted']

# Color modes
COLOR_MODES = ['full', 'basic', 'none']

# Compression methods
COMPRESSION_METHODS = ['tiered', 'simple', 'extractive']

# Local LLM providers (for detecting local vs cloud)
LOCAL_PROVIDERS = ['ollama', 'lmstudio', 'local', 'localai']

# STT providers
STT_PROVIDERS = ['local_whisper', 'openai_whisper', 'deepgram']

# TTS providers
TTS_PROVIDERS = ['local_piper', 'openai_tts', 'elevenlabs']

# Embedding providers for drift/RAG
EMBEDDING_PROVIDERS = ['sentence-transformers', 'openai', 'huggingface']

# Model parameter contexts
MODEL_CONTEXTS = ['chat', 'detection', 'compression', 'synthesis']

# Muse source options
MUSE_SOURCES = ['first-only', 'top-three', 'all-relevant', 'selective']

# Porcupine built-in wake words (free tier)
PORCUPINE_KEYWORDS = [
    'alexa', 'americano', 'blueberry', 'bumblebee', 'computer',
    'grapefruit', 'grasshopper', 'hey google', 'hey siri', 'jarvis',
    'ok google', 'picovoice', 'porcupine', 'terminator'
]
