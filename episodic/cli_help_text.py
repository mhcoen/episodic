"""
Help text templates for CLI commands.

This module contains the top-level help routing functions
(show_help_with_categories, show_category_help, show_advanced_help,
show_simple_help, get_category_icon) and re-exports the shared
_display_aligned_commands utility and individual category help functions
from cli_help_categories.py.

Extracted from cli_registry.py to keep all files under 500 lines.
"""

import typer
from episodic.configuration import get_heading_color, get_text_color

# Re-export from cli_help_categories so callers can import from either module.
from episodic.cli_help_categories import (  # noqa: F401
    _display_aligned_commands,
    show_chat_help,
    show_settings_help,
    show_search_help,
    show_history_help,
    show_topics_help,
    show_markdown_help,
    show_voice_help,
    show_assistant_help,
    show_calendar_email_help,
    show_mcp_help,
)


def show_help_with_categories():
    """Show basic help information with common commands and categories."""
    from episodic.cli_registry import _ensure_registry_initialized
    _ensure_registry_initialized()

    # Check if we're in simple mode
    from episodic.config import config
    if config.get("interface_mode", "advanced") == "simple":
        show_simple_help()
        return

    # Essential commands
    essential_commands = [
        ("/muse", "Enable web search synthesis mode"),
        ("/chat", "Enable normal LLM conversation mode"),
        ("/voice", "Toggle voice mode (speech input/output)"),
        ("/mode", "Switch between local/cloud provider modes"),
        ("/style", "Set global response style (concise/standard/comprehensive/custom)"),
        ("/format", "Set global response format (paragraph/bulleted/mixed/academic)"),
        ("/topics", "List conversation topics"),
        ("/out", "Export conversation to markdown"),
        ("/list", "Show recent conversation nodes"),
        ("/config", "View current system configuration"),
        ("/set", "Change configuration settings"),
        ("/reset", "Reset configuration to defaults")
    ]

    # Command categories
    categories = [
        ("/help chat", "Mode switching and conversation management"),
        ("/help voice", "Voice mode, STT/TTS providers"),
        ("/help assistant", "Timers, alarms, weather, news, and utilities"),
        ("/help settings", "Configuration and system management"),
        ("/help search", "Knowledge base and muse configuration"),
        ("/help history", "Navigation and conversation history"),
        ("/help topics", "Topic detection and management"),
        ("/help markdown", "Markdown file operations"),
        ("/help calendar", "Calendar and email commands (Google Workspace)"),
        ("/help mcp", "MCP server, tokens, and external tool access")
    ]

    # Other options
    other_options = [
        ("/help all", "Show all available commands"),
        ("/help <command>", "Get detailed help for a specific command")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = essential_commands + categories + other_options
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("⌨️  Just type to chat.", fg=get_text_color())
    typer.echo()
    typer.secho("Or interact with : /<command> [options]", fg=get_text_color())
    typer.echo()

    # Display essential commands
    typer.secho("💬 Essential Commands:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(essential_commands, max_width)
    typer.echo()

    # Display command categories
    typer.secho("📚 Command Categories:", fg=get_heading_color(), bold=True)
    typer.secho("Use '/help <category>' for detailed commands in each area.", fg=get_text_color())
    typer.echo()
    _display_aligned_commands(categories, max_width)
    typer.echo()

    # Display other options
    typer.secho("📖 Other options:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(other_options, max_width)
    typer.echo()

    typer.secho("🚪 Type '/exit' or '/quit' to leave", fg=get_text_color())


def show_category_help(category: str):
    """Show help for a specific category."""
    from episodic.cli_registry import _ensure_registry_initialized
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
    elif category == "markdown":
        show_markdown_help()
    elif category == "voice":
        show_voice_help()
    elif category == "assistant":
        show_assistant_help()
    elif category in ("calendar", "email", "cal"):
        show_calendar_email_help()
    elif category == "mcp":
        show_mcp_help()
    else:
        typer.secho(f"Unknown help category: {category}", fg="red")
        typer.secho("Available categories: chat, settings, search, history, topics, markdown, voice, assistant, calendar, mcp", fg=get_text_color())


def show_advanced_help():
    """Show all available commands organized by categories."""
    from episodic.cli_registry import _ensure_registry_initialized
    from episodic.commands.registry import command_registry
    _ensure_registry_initialized()

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
        ("muse-detail", "Detail level: minimal, moderate, detailed, maximum"),
        ("muse-max-tokens", "Direct token limit (overrides global style if set)"),
        ("muse-sources", "Source selection: first-only, top-three, all-relevant"),
        ("muse-model", "Model for synthesis (None = use main model)")
    ]

    for setting, description in muse_settings:
        setting_display = f"/set {setting}"
        all_command_tuples.append((setting_display, description))

    # Find the longest command across ALL sections for uniform alignment
    max_width = max(len(cmd) for cmd, _ in all_command_tuples)

    # Display header
    typer.secho("📚 Episodic Commands (Advanced)", fg=get_heading_color(), bold=True)
    typer.echo()

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
        typer.secho(f"{icon} {category}", fg=get_heading_color(), bold=True)

        # Collect commands for this category
        category_commands = []
        for cmd_info in active_commands:
            cmd_display = f"/{cmd_info.name}"
            if cmd_info.aliases:
                cmd_display += f" (/{', /'.join(cmd_info.aliases)})"
            category_commands.append((cmd_display, cmd_info.description))

        # Display with uniform alignment
        _display_aligned_commands(category_commands, max_width)
        typer.echo()

    # Show muse configuration details
    typer.secho("🎭 Muse Mode Configuration", fg=get_heading_color(), bold=True)
    muse_command_tuples = [(f"/set {setting}", description) for setting, description in muse_settings]
    _display_aligned_commands(muse_command_tuples, max_width)
    typer.echo()

    # Display quick tips
    typer.secho("💡 Quick Tips", fg=get_heading_color(), bold=True)
    typer.secho("• Type messages directly to chat", fg=get_text_color())
    typer.secho("• Common settings: ", fg=get_text_color(), nl=False)
    typer.secho("/set debug off", fg="cyan", bold=True, nl=False)
    typer.secho(", ", fg=get_text_color(), nl=False)
    typer.secho("/set cost on", fg="cyan", bold=True, nl=False)
    typer.secho(", ", fg=get_text_color(), nl=False)
    typer.secho("/set topics on", fg="cyan", bold=True)
    typer.secho("• Global response style: ", fg=get_text_color(), nl=False)
    typer.secho("/style concise|standard|comprehensive|custom", fg="cyan", bold=True)
    typer.secho("• Global response format: ", fg=get_text_color(), nl=False)
    typer.secho("/format paragraph|bulleted|mixed|academic", fg="cyan", bold=True)
    typer.secho("• Type ", fg=get_text_color(), nl=False)
    typer.secho("/exit", fg="cyan", bold=True, nl=False)
    typer.secho(" or ", fg=get_text_color(), nl=False)
    typer.secho("/quit", fg="cyan", bold=True, nl=False)
    typer.secho(" to leave", fg=get_text_color())


def show_simple_help():
    """Show help for simple mode - just the essential commands."""
    # Group commands by category
    conversation_commands = [
        ("/chat", "Normal conversation mode"),
        ("/muse", "Web search mode (like Perplexity)"),
        ("/new", "Start fresh topic")
    ]

    file_commands = [
        ("/save", "Save current topic"),
        ("/load", "Load a conversation"),
        ("/files", "List saved conversations")
    ]

    style_commands = [
        ("/style", "Set response length (concise/standard/comprehensive)"),
        ("/format", "Set response format (paragraph/bulleted/mixed/academic)")
    ]

    system_commands = [
        ("/theme", "Change color theme"),
        ("/help", "Show this help"),
        ("/exit", "Leave Episodic")
    ]

    # Find the longest command across all groups for uniform alignment
    all_commands = conversation_commands + file_commands + style_commands + system_commands
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("⌨️  Just type to chat.", fg=get_text_color())
    typer.echo()

    # Display conversation commands
    typer.secho("💬 Conversation", fg=get_heading_color(), bold=True)
    _display_aligned_commands(conversation_commands, max_width)
    typer.echo()

    # Display file commands
    typer.secho("📁 Files", fg=get_heading_color(), bold=True)
    _display_aligned_commands(file_commands, max_width)
    typer.echo()

    # Display style commands
    typer.secho("✨ Style", fg=get_heading_color(), bold=True)
    _display_aligned_commands(style_commands, max_width)
    typer.echo()

    # Display system commands
    typer.secho("⚙️  System", fg=get_heading_color(), bold=True)
    _display_aligned_commands(system_commands, max_width)
    typer.echo()

    typer.secho("💡 Type /advanced anytime to unlock all features", fg=get_text_color(), dim=False)


def get_category_icon(category: str) -> str:
    """Get emoji icon for command category."""
    icons = {
        "Navigation": "🧭",
        "Conversation": "💬",
        "Topics": "📑",
        "Configuration": "⚙️",
        "Knowledge Base": "📚",
        "Compression": "📦",
        "Utility": "🛠️",
        "MCP": "🔌"
    }
    return icons.get(category, "📌")
