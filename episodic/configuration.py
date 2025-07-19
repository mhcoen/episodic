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
PROMPT_COLOR = "<ansicyan>\n> </ansicyan>"

# Color schemes for different modes
COLOR_SCHEMES = {
    "dark": {
        "llm_response": "BRIGHT_CYAN",       # LLM responses - bright cyan for better visibility
        "system_info": "BRIGHT_CYAN",       # All episodic system information
        "prompt": "CYAN",                   # Prompt symbol color
        "text": "WHITE",                    # General text (labels, etc)
        "heading": "BRIGHT_YELLOW"          # Section headings - changed to yellow for better bold visibility
    },
    "light": {
        "llm_response": "BLUE",             # LLM responses for light background
        "system_info": "MAGENTA",           # All episodic system information  
        "prompt": "BLUE",                   # Prompt symbol color
        "text": "BLACK",                    # General text (labels, etc)
        "heading": "BRIGHT_BLUE"            # Section headings - changed to bright blue for better visibility
    },
    "gemini": {
        "llm_response": "BRIGHT_BLUE",      # Gemini-inspired blue for AI responses
        "system_info": "BRIGHT_CYAN",       # Bright cyan for system messages
        "prompt": "BRIGHT_GREEN",           # Bright green prompt
        "text": "WHITE",                    # White text
        "heading": "BRIGHT_CYAN"            # Cyan headings for tech feel
    },
    "solarized_dark": {
        "llm_response": "CYAN",             # Solarized cyan
        "system_info": "BLUE",              # Solarized blue
        "prompt": "GREEN",                  # Solarized green
        "text": "WHITE",                    # Base0
        "heading": "YELLOW"                 # Solarized yellow
    },
    "solarized_light": {
        "llm_response": "CYAN",             # Solarized cyan
        "system_info": "BLUE",              # Solarized blue
        "prompt": "GREEN",                  # Solarized green
        "text": "BLACK",                    # Base00
        "heading": "YELLOW"                 # Solarized yellow
    },
    "monokai": {
        "llm_response": "WHITE",            # White for main text
        "system_info": "CYAN",              # Cyan for system
        "prompt": "GREEN",                  # Green prompt
        "text": "WHITE",                    # White text
        "heading": "YELLOW"                 # Yellow headings
    },
    "dracula": {
        "llm_response": "WHITE",            # Foreground white
        "system_info": "CYAN",              # Dracula cyan
        "prompt": "GREEN",                  # Dracula green
        "text": "WHITE",                    # Foreground
        "heading": "BRIGHT_MAGENTA"         # Dracula pink
    },
    "nord": {
        "llm_response": "BRIGHT_CYAN",      # Nord8 - bright cyan
        "system_info": "CYAN",              # Nord7 - cyan
        "prompt": "GREEN",                  # Nord14 - green
        "text": "WHITE",                    # Snow storm
        "heading": "BRIGHT_BLUE"            # Nord9 - bright blue
    },
    "gruvbox": {
        "llm_response": "BRIGHT_YELLOW",    # Gruvbox yellow
        "system_info": "CYAN",              # Gruvbox aqua
        "prompt": "GREEN",                  # Gruvbox green
        "text": "WHITE",                    # Light
        "heading": "YELLOW"                 # Gruvbox yellow
    },
    "one_dark": {
        "llm_response": "CYAN",             # One Dark cyan
        "system_info": "BRIGHT_BLUE",       # One Dark blue
        "prompt": "GREEN",                  # One Dark green
        "text": "WHITE",                    # One Dark foreground
        "heading": "BRIGHT_MAGENTA"         # One Dark purple
    },
    "tokyo_night": {
        "llm_response": "BRIGHT_CYAN",      # Tokyo Night cyan
        "system_info": "BLUE",              # Tokyo Night blue
        "prompt": "GREEN",                  # Tokyo Night green
        "text": "WHITE",                    # Tokyo Night foreground
        "heading": "BRIGHT_MAGENTA"         # Tokyo Night magenta
    },
    "catppuccin": {
        "llm_response": "BRIGHT_BLUE",      # Catppuccin sky
        "system_info": "CYAN",              # Catppuccin teal
        "prompt": "GREEN",                  # Catppuccin green
        "text": "WHITE",                    # Catppuccin text
        "heading": "BRIGHT_MAGENTA"         # Catppuccin mauve
    },
    "material": {
        "llm_response": "CYAN",             # Material cyan
        "system_info": "BRIGHT_CYAN",       # Material light cyan
        "prompt": "GREEN",                  # Material green
        "text": "WHITE",                    # Material foreground
        "heading": "YELLOW"                 # Material yellow
    },
    "ayu_dark": {
        "llm_response": "BRIGHT_YELLOW",    # Ayu orange
        "system_info": "CYAN",              # Ayu cyan
        "prompt": "GREEN",                  # Ayu green
        "text": "WHITE",                    # Ayu foreground
        "heading": "BRIGHT_YELLOW"          # Ayu yellow
    },
    "palenight": {
        "llm_response": "BRIGHT_CYAN",      # Palenight cyan
        "system_info": "BLUE",              # Palenight blue
        "prompt": "GREEN",                  # Palenight green
        "text": "WHITE",                    # Palenight foreground
        "heading": "BRIGHT_MAGENTA"         # Palenight purple
    },
    "tomorrow_night": {
        "llm_response": "CYAN",             # Tomorrow cyan
        "system_info": "BLUE",              # Tomorrow blue
        "prompt": "GREEN",                  # Tomorrow green
        "text": "WHITE",                    # Tomorrow foreground
        "heading": "YELLOW"                 # Tomorrow yellow
    },
    "synthwave": {
        "llm_response": "BRIGHT_MAGENTA",   # Synthwave pink
        "system_info": "BRIGHT_CYAN",       # Synthwave cyan
        "prompt": "BRIGHT_MAGENTA",         # Synthwave magenta
        "text": "WHITE",                    # Synthwave foreground
        "heading": "BRIGHT_YELLOW"          # Synthwave yellow
    },
    "cobalt2": {
        "llm_response": "BRIGHT_YELLOW",    # Cobalt2 yellow
        "system_info": "BRIGHT_CYAN",       # Cobalt2 cyan
        "prompt": "GREEN",                  # Cobalt2 green
        "text": "WHITE",                    # Cobalt2 foreground
        "heading": "BRIGHT_YELLOW"          # Cobalt2 orange
    },
    "night_owl": {
        "llm_response": "CYAN",             # Night Owl cyan
        "system_info": "BRIGHT_BLUE",       # Night Owl blue
        "prompt": "GREEN",                  # Night Owl green
        "text": "WHITE",                    # Night Owl foreground
        "heading": "BRIGHT_YELLOW"          # Night Owl yellow
    },
    "zenburn": {
        "llm_response": "BRIGHT_YELLOW",    # Zenburn yellow
        "system_info": "CYAN",              # Zenburn cyan
        "prompt": "GREEN",                  # Zenburn green
        "text": "WHITE",                    # Zenburn foreground
        "heading": "YELLOW"                 # Zenburn yellow
    },
    "everforest": {
        "llm_response": "GREEN",            # Everforest aqua
        "system_info": "CYAN",              # Everforest blue
        "prompt": "GREEN",                  # Everforest green
        "text": "WHITE",                    # Everforest foreground
        "heading": "BRIGHT_GREEN"           # Everforest green
    },
    "kanagawa": {
        "llm_response": "CYAN",             # Kanagawa wave blue
        "system_info": "BRIGHT_BLUE",       # Kanagawa spring blue
        "prompt": "GREEN",                  # Kanagawa spring green
        "text": "WHITE",                    # Kanagawa foreground
        "heading": "BRIGHT_YELLOW"          # Kanagawa autumn yellow
    },
    "rosepine": {
        "llm_response": "BRIGHT_CYAN",      # Rose Pine foam
        "system_info": "BRIGHT_MAGENTA",    # Rose Pine iris
        "prompt": "GREEN",                  # Rose Pine pine
        "text": "WHITE",                    # Rose Pine text
        "heading": "BRIGHT_MAGENTA"         # Rose Pine rose
    }
}

# Default color mode
DEFAULT_COLOR_MODE = "dark"

# Model context limits (approximate token limits for common models)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    # Local models - conservative estimates
    "llama3": 8192,
    "mistral": 32768,
    "codellama": 16384,
    "phi3": 4096,
    "local-model": 8192
}

def get_model_context_limit(model_name: str) -> int:
    """Get the context limit for a model, with fallback for unknown models."""
    # Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
    clean_model = model_name.split('/')[-1] if '/' in model_name else model_name
    return MODEL_CONTEXT_LIMITS.get(clean_model, 8192)  # Default to 8192 if unknown

def get_color_scheme():
    """Get the current color scheme based on configuration."""
    from episodic.config import config
    mode = config.get("color_mode", DEFAULT_COLOR_MODE)
    return COLOR_SCHEMES.get(mode, COLOR_SCHEMES[DEFAULT_COLOR_MODE])

def get_llm_color():
    """Get the color for LLM responses."""
    import typer
    color_name = get_color_scheme()["llm_response"]
    return getattr(typer.colors, color_name)

def get_system_color():
    """Get the color for system information."""
    import typer
    color_name = get_color_scheme()["system_info"]
    return getattr(typer.colors, color_name)

def get_prompt_color():
    """Get the color for the prompt."""
    color_name = get_color_scheme()["prompt"]
    return f"<ansi{color_name.lower()}>\n> </ansi{color_name.lower()}>"

def get_text_color():
    """Get the color for general text (labels, etc)."""
    import typer
    color_name = get_color_scheme()["text"]
    return getattr(typer.colors, color_name)

def get_heading_color():
    """Get the color for section headings."""
    import typer
    color_name = get_color_scheme()["heading"]
    return getattr(typer.colors, color_name)

def get_error_color():
    """Get the color for error messages."""
    import typer
    return typer.colors.RED

def get_warning_color():
    """Get the color for warning messages."""
    import typer
    return typer.colors.YELLOW

def get_success_color():
    """Get the color for success messages."""
    import typer
    return typer.colors.GREEN

def get_info_color():
    """Get the color for informational messages."""
    import typer
    # Use system color for info messages to match theme
    return get_system_color()

# Cost display formatting
COST_PRECISION = 6  # Number of decimal places for cost display
ZERO_COST_PRECISION = 1  # Decimal places for zero cost display

def format_cost(cost_usd: float) -> str:
    """Format cost for display with appropriate precision."""
    if cost_usd == 0.0:
        return "$0.000000"
    elif cost_usd < 0.000001:  # Less than $0.000001
        return f"${cost_usd:.8f}"  # 8 decimal places for very small costs
    elif cost_usd < 0.001:  # Less than $0.001
        return f"${cost_usd:.6f}"  # 6 decimal places
    elif cost_usd < 1.0:  # Less than $1
        return f"${cost_usd:.4f}"  # 4 decimal places
    else:
        return f"${cost_usd:.2f}"  # 2 decimal places for larger costs

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