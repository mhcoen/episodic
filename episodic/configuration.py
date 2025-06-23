"""
Project-wide configuration constants.
This file contains all default values, magic strings, and fixed configuration used throughout the project.
"""

# Default model and system configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_CONTEXT_DEPTH = 5

# File paths and directories
DEFAULT_HISTORY_FILE = "~/.episodic_history"

# Provider names
PROVIDER_OLLAMA = "ollama"
PROVIDER_LMSTUDIO = "lmstudio"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
LOCAL_PROVIDERS = [PROVIDER_OLLAMA, PROVIDER_LMSTUDIO]

# Pricing and token calculation
PRICING_TOKEN_COUNT = 1000  # Number of tokens to use for pricing calculations

# Server configuration
DEFAULT_VISUALIZATION_PORT = 5000

# Display configuration
DEFAULT_LIST_COUNT = 5  # Default number of nodes to list
MAX_CONTENT_DISPLAY_LENGTH = 50  # Max length for truncated content display

# Command keywords
EXIT_COMMANDS = ["exit", "quit"]

# Prompt styling
PROMPT_COLOR = "<ansigreen>> </ansigreen>"

# Cost display formatting
COST_PRECISION = 6  # Number of decimal places for cost display
ZERO_COST_PRECISION = 1  # Decimal places for zero cost display

# Server configuration
SERVER_SHUTDOWN_DELAY = 0.1  # seconds

# Polling and update intervals
HTTP_POLLING_INTERVAL = 5000  # milliseconds
MAIN_LOOP_SLEEP_INTERVAL = 1  # seconds

# Cache configuration
CACHED_TOKEN_DISCOUNT_RATE = 0.5  # 50% cost reduction for cached tokens

# Database configuration
MAX_DATABASE_RETRIES = 3
FALLBACK_ID_LENGTH = 4
MIN_SHORT_ID_LENGTH = 2
SHORT_ID_MAX_LENGTH = 3
DATABASE_FILENAME = "episodic.db"

# ID generation
ID_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"  # ascii_lowercase + digits

# File patterns
PROMPT_FILE_EXTENSIONS = ["*.txt", "*.md"]

# Visualization configuration
VISUALIZATION_VERTICAL_SPACING = 2  # multiplier for level spacing
VISUALIZATION_DEFAULT_WIDTH = 1000  # pixels when width is "100%"
VISUALIZATION_NODE_SIZE = 20
VISUALIZATION_CONTENT_TRUNCATE_LENGTH = 50  # characters

# Visualization colors
CURRENT_NODE_COLOR = "#FFA500"  # Orange
DEFAULT_NODE_COLOR = "#97c2fc"  # Light blue
VIRTUAL_ROOT_COLOR = "#CCCCCC"  # Gray

# UI interaction thresholds
CLICK_THRESHOLD_TOUCH = 100  # pixels
CLICK_THRESHOLD_MOUSE = 50   # pixels

# Server defaults
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5000