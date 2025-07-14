"""
Enhanced CLI command handling using the command registry.

This module provides a cleaner command handling system that uses
the centralized command registry.
"""

import shlex
import typer
from typing import List
from episodic.commands.registry import command_registry, register_all_commands
from episodic.configuration import (
    EXIT_COMMANDS, get_system_color, get_heading_color, get_text_color
)

# Ensure commands are registered
_registry_initialized = False

def _ensure_registry_initialized():
    global _registry_initialized
    if not _registry_initialized:
        register_all_commands()
        _registry_initialized = True


def handle_command_with_registry(command_str: str) -> bool:
    """
    Handle a command string using the command registry.
    
    Returns:
        bool: True if should exit, False otherwise
    """
    _ensure_registry_initialized()
    
    # Parse the command
    try:
        parts = shlex.split(command_str)
    except ValueError as e:
        typer.secho(f"Error parsing command: {e}", fg="red")
        return False
    
    if not parts:
        return False
    
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    
    # Remove leading slash if present
    if cmd.startswith('/'):
        cmd = cmd[1:]
    
    # Check for exit commands
    if cmd in EXIT_COMMANDS or cmd == "q":
        return True
    
    # Look up command in registry
    cmd_info = command_registry.get_command(cmd)
    
    if not cmd_info:
        typer.secho(f"Unknown command: /{cmd}", fg="red")
        typer.echo("Type /help for available commands")
        return False
    
    # Check if deprecated
    if cmd_info.deprecated:
        typer.secho(
            f"⚠️  Warning: /{cmd} is deprecated. Use /{cmd_info.replacement} instead.",
            fg="yellow"
        )
    
    # Handle the command based on its type
    try:
        # Special handling for unified commands
        if cmd in ["topics", "compression"]:
            # These commands expect action as first argument
            if args:
                action = args[0]
                remaining_args = args[1:]
                # Call with action and parse remaining args
                cmd_info.handler(action, *remaining_args)
            else:
                # Default action
                cmd_info.handler()
        else:
            # Legacy command handling - needs specific argument parsing
            # This is where we'd need command-specific logic
            # For now, pass through to original handler
            return handle_legacy_command(cmd, args)
    
    except Exception as e:
        typer.secho(f"Error executing command: {e}", fg="red")
        if typer.get_app().get("debug", False):
            import traceback
            traceback.print_exc()
    
    return False


def handle_legacy_command(cmd: str, args: List[str]) -> bool:
    """Handle legacy commands that aren't yet converted to new style."""
    # Import the original handle_command logic
    from episodic.cli import handle_command
    
    # Reconstruct command string
    if args:
        # Properly quote arguments that contain spaces
        quoted_args = []
        for arg in args:
            if ' ' in arg:
                quoted_args.append(f'"{arg}"')
            else:
                quoted_args.append(arg)
        command_str = f"/{cmd} {' '.join(quoted_args)}"
    else:
        command_str = f"/{cmd}"
    
    # Use original handler
    return handle_command(command_str)


def _format_aligned_commands(commands_and_descriptions, max_width=None):
    """Format a list of (command, description) tuples with perfect alignment."""
    if not commands_and_descriptions:
        return ""
    
    # Use provided max_width or find the longest command in this list
    if max_width is None:
        max_width = max(len(cmd) for cmd, _ in commands_and_descriptions)
    
    # Format each line with perfect alignment
    lines = []
    for cmd, desc in commands_and_descriptions:
        padding = ' ' * (max_width - len(cmd) + 1)  # +1 for space after colon
        lines.append(f"- **{cmd}**:{padding}{desc}")
    
    return '\n'.join(lines)


def show_help_with_categories():
    """Show basic help information with common commands and categories."""
    _ensure_registry_initialized()
    from episodic.text_formatter import display_help_content
    
    # Essential commands
    essential_commands = [
        ("/muse", "Enable web search synthesis mode"),
        ("/chat", "Enable normal LLM conversation mode"), 
        ("/topics", "List conversation topics"),
        ("/list", "Show recent conversation nodes"),
        ("/config", "View current system configuration"),
        ("/set", "Change configuration settings"),
        ("/reset", "Reset configuration to defaults")
    ]
    
    # Command categories  
    categories = [
        ("/help chat", "Mode switching and conversation management"),
        ("/help settings", "Configuration and system management"),
        ("/help search", "Knowledge base and muse configuration"),
        ("/help history", "Navigation and conversation history"),
        ("/help topics", "Topic detection and management")
    ]
    
    # Other options
    other_options = [
        ("/help all", "Show all available commands"),
        ("/help <command>", "Get detailed help for a specific command")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = essential_commands + categories + other_options
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""⌨️  Just type to chat.

Or interact with : /<command> [options]

## 💬 Essential Commands:
{_format_aligned_commands(essential_commands, max_width)}

## 📚 Command Categories:
Use '/help <category>' for detailed commands in each area.

{_format_aligned_commands(categories, max_width)}

## 📖 Other options:
{_format_aligned_commands(other_options, max_width)}

🚪 Type '/exit' or '/quit' to leave
"""
    
    display_help_content(content)


def show_category_help(category: str):
    """Show help for a specific category."""
    _ensure_registry_initialized()
    
    category = category.lower()
    if category == "chat":
        show_chat_help()
    elif category == "settings":
        show_settings_help()
    elif category == "search":
        show_search_help()
    elif category == "history":
        show_history_help()
    elif category == "topics":
        show_topics_help()
    else:
        typer.secho(f"Unknown help category: {category}", fg="red")
        typer.secho("Available categories: chat, settings, search, history, topics", fg=get_text_color())


def show_chat_help():
    """Show chat and conversation management commands."""
    from episodic.text_formatter import display_help_content
    
    # Commands
    commands = [
        ("/chat", "Enable normal LLM conversation mode"),
        ("/muse", "Enable web search synthesis mode (like Perplexity)"),
        ("/topics", "List conversation topics"),
        ("/topics list", "List all topics with details"),
        ("/topics rename", "Rename ongoing topics"),
        ("/summary", "Summarize recent conversation"),
        ("/cost", "Show token usage and costs"),
        ("/set muse-style <style>", "Set muse response length (concise/standard/comprehensive)")
    ]
    
    # Examples
    examples = [
        ("/muse", "Switch to web search mode"),
        ("/set muse-style concise", "Set shorter responses"),
        ("/topics", "See conversation topics")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""## 💬 Chat & Conversation Management
Mode switching and conversation flow control.

Commands:
{_format_aligned_commands(commands, max_width)}

### Examples:
{_format_aligned_commands(examples, max_width)}
"""
    
    display_help_content(content)


def show_settings_help():
    """Show configuration and system management commands.""" 
    from episodic.text_formatter import display_help_content
    
    # Commands
    commands = [
        ("/config", "View current system configuration"),
        ("/set", "Show commonly changed settings"),
        ("/set <param> <value>", "Change a configuration parameter"),
        ("/model", "Show current models for all contexts"),
        ("/model chat <name>", "Set the main chat model"),
        ("/model detection <name>", "Set the topic detection model"),
        ("/mset", "Show model parameters"),
        ("/mset chat.temperature 0.7", "Set model-specific parameters"),
        ("/script <file>", "Execute commands from a script file")
    ]
    
    # Common settings
    common_settings = [
        ("/set debug true", "Enable debug output"),
        ("/set cost true", "Show token costs"),
        ("/set streaming false", "Disable response streaming")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + common_settings
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""## ⚙️ Settings & System Management
Configure the system and manage models.

Commands:
{_format_aligned_commands(commands, max_width)}

### Common Settings:
{_format_aligned_commands(common_settings, max_width)}
"""
    
    display_help_content(content)


def show_search_help():
    """Show knowledge base and muse configuration commands."""
    from episodic.text_formatter import display_help_content
    
    # Commands
    commands = [
        ("/rag", "Show RAG (knowledge base) status"),
        ("/rag on", "Enable knowledge base integration"),
        ("/search <query>", "Search the knowledge base"),
        ("/index <file>", "Add a file to the knowledge base"),
        ("/index --text \"<content>\"", "Add text directly to knowledge base"),
        ("/docs", "List documents in knowledge base"),
        ("/docs show <id>", "Show a specific document"),
        ("/docs remove <id>", "Remove a document"),
        ("/web", "Show muse web search provider configuration"),
        ("/web provider <name>", "Set web search provider for muse mode")
    ]
    
    # Examples
    examples = [
        ("/index ~/documents/notes.md", "Index a file"),
        ("/search python functions", "Search knowledge base"),
        ("/set rag-enabled true", "Enable RAG integration")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""## 🔍 Knowledge Base & Muse Configuration
Search your knowledge base and configure muse web search.

Commands:
{_format_aligned_commands(commands, max_width)}

### Examples:
{_format_aligned_commands(examples, max_width)}
"""
    
    display_help_content(content)


def show_history_help():
    """Show navigation and conversation history commands."""
    from episodic.text_formatter import display_help_content
    
    # Commands
    commands = [
        ("/list", "Show recent conversation nodes"),
        ("/list 10", "Show last 10 nodes"),
        ("/last", "Show the last exchange"),
        ("/show <id>", "Show details of a specific node"),
        ("/print", "Print current node content"),
        ("/print <id>", "Print specific node content"),
        ("/head", "Show current node"),
        ("/head <id>", "Set current node"),
        ("/history", "Show conversation history (alias for /list)"),
        ("/tree", "Show conversation tree structure")
    ]
    
    # Navigation examples
    navigation = [
        ("/list", "See recent exchanges"),
        ("/show AB", "View details of node AB"),
        ("/head CD", "Continue from node CD")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + navigation
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""## 🧭 Navigation & History
Navigate through conversation history and nodes.

Commands:
{_format_aligned_commands(commands, max_width)}

### Navigation:
{_format_aligned_commands(navigation, max_width)}
"""
    
    display_help_content(content)


def show_topics_help():
    """Show topic detection and management commands."""
    from episodic.text_formatter import display_help_content
    
    # Commands
    commands = [
        ("/topics", "List conversation topics (default action)"),
        ("/topics list", "List all topics with details and node boundaries"),
        ("/topics rename", "Rename ongoing topics interactively"),
        ("/topics compress", "Compress current topic to save space"),
        ("/topics index <n>", "Manual topic detection with window size"),
        ("/topics scores", "Show topic detection scores and analysis"),
        ("/topics stats", "Show topic statistics and completion status")
    ]
    
    # Examples
    examples = [
        ("/topics", "List current topics"),
        ("/topics index 5", "Detect topics with 5-node window"),
        ("/topics stats", "View topic analytics")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    content = f"""## 📑 Topic Detection & Management
Manage conversation topics and analyze topic detection.

Commands:
{_format_aligned_commands(commands, max_width)}

### Examples:
{_format_aligned_commands(examples, max_width)}
"""
    
    display_help_content(content)


def show_advanced_help():
    """Show all available commands organized by categories."""
    _ensure_registry_initialized()
    from episodic.text_formatter import display_help_content
    
    # Get commands by category
    categories = command_registry.get_commands_by_category()
    
    # Define category order for advanced view
    category_order = [
        "Navigation", "Conversation", "Topics", "Configuration",
        "Knowledge Base", "Compression", "Utility"
    ]
    
    # Collect all commands and muse settings for uniform alignment
    all_command_tuples = []
    
    # Collect active commands from all categories
    for category in category_order:
        if category not in categories:
            continue
        commands = categories[category]
        if not commands:
            continue
        active_commands = [cmd for cmd in commands if not cmd.deprecated]
        for cmd_info in active_commands:
            cmd_display = f"/{cmd_info.name}"
            if cmd_info.aliases:
                cmd_display += f" (/{', /'.join(cmd_info.aliases)})"
            all_command_tuples.append((cmd_display, cmd_info.description))
    
    # Add muse configuration settings
    muse_settings = [
        ("muse-style", "Response length: concise (~150 words), standard (~300), comprehensive (~500), exhaustive (800+)"),
        ("muse-detail", "Detail level: minimal, moderate, detailed, maximum"),
        ("muse-format", "Output format: paragraph, bullet-points, mixed, academic"),
        ("muse-max-tokens", "Direct token limit (overrides style if set)"),
        ("muse-sources", "Source selection: first-only, top-three, all-relevant"),
        ("muse-model", "Model for synthesis (None = use main model)")
    ]
    
    for setting, description in muse_settings:
        setting_display = f"/set {setting}"
        all_command_tuples.append((setting_display, description))
    
    # Find the longest command across ALL sections for uniform alignment
    max_width = max(len(cmd) for cmd, _ in all_command_tuples)
    
    # Build content
    content = "# 📚 Episodic Commands (Advanced)\n\n"
    
    for category in category_order:
        if category not in categories:
            continue
            
        commands = categories[category]
        if not commands:
            continue
        
        # Skip deprecated commands completely
        active_commands = [cmd for cmd in commands if not cmd.deprecated]
        if not active_commands:
            continue
        
        # Category header
        icon = get_category_icon(category)
        content += f"## {icon} {category}\n"
        
        # Collect commands for this category
        category_commands = []
        for cmd_info in active_commands:
            cmd_display = f"/{cmd_info.name}"
            if cmd_info.aliases:
                cmd_display += f" (/{', /'.join(cmd_info.aliases)})"
            category_commands.append((cmd_display, cmd_info.description))
        
        # Format with uniform alignment
        content += _format_aligned_commands(category_commands, max_width) + "\n\n"
    
    # Show muse configuration details
    content += "## 🎭 Muse Mode Configuration\n"
    muse_command_tuples = [(f"/set {setting}", description) for setting, description in muse_settings]
    content += _format_aligned_commands(muse_command_tuples, max_width) + "\n"
    
    content += """\n## 💡 Quick Tips
• Type messages directly to chat
• Common settings: **/set debug off**, **/set cost on**, **/set topics on**  
• Muse mode length: **/set muse-style concise|standard|comprehensive**
• Type **/exit** or **/quit** to leave
"""
    
    display_help_content(content)


def get_category_icon(category: str) -> str:
    """Get emoji icon for command category."""
    icons = {
        "Navigation": "🧭",
        "Conversation": "💬",
        "Topics": "📑",
        "Configuration": "⚙️",
        "Knowledge Base": "📚",
        "Compression": "📦",
        "Utility": "🛠️"
    }
    return icons.get(category, "📌")
