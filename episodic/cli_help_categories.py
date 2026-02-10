"""
Individual category help pages for the CLI.

Each show_*_help() function renders one help category (chat, settings,
search, history, topics, markdown, voice, assistant, mcp).

Also defines _display_aligned_commands, the shared display utility used
by all help functions.  Top-level routing functions (show_help_with_categories,
show_category_help, show_advanced_help, show_simple_help) live in
cli_help_text.py.
"""

import shutil
import textwrap

import typer
from episodic.configuration import get_system_color, get_heading_color, get_text_color


def _display_aligned_commands(commands_and_descriptions, max_width=None):
    """Display a list of (command, description) tuples with perfect alignment and cyan descriptions."""
    if not commands_and_descriptions:
        return

    # Use provided max_width or find the longest command in this list
    if max_width is None:
        max_width = max(len(cmd) for cmd, _ in commands_and_descriptions)

    # Get terminal width for wrapping
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    # Display each line with perfect alignment and word wrapping
    for cmd, desc in commands_and_descriptions:
        padding = ' ' * max(2, max_width - len(cmd) + 2)  # Minimum 2 spaces between command and description

        # Calculate available width for description
        command_part_width = 1 + len(cmd) + len(padding)  # " " + command + padding
        desc_width = max(40, terminal_width - command_part_width - 4)  # Leave some margin

        # Wrap the description if needed
        wrapped_lines = textwrap.wrap(desc, width=desc_width)

        if not wrapped_lines:
            wrapped_lines = [""]

        # Display first line with command
        typer.secho(f" ", nl=False)
        typer.secho(f"{cmd}", bold=True, nl=False)
        typer.echo(padding, nl=False)
        typer.secho(wrapped_lines[0], fg="cyan", nl=True)

        # Display continuation lines if any
        if len(wrapped_lines) > 1:
            continuation_padding = ' ' * (command_part_width + 1)  # Add one extra space for readability
            for line in wrapped_lines[1:]:
                typer.echo(continuation_padding, nl=False)
                typer.secho(line, fg="cyan")


def show_chat_help():
    """Show chat and conversation management commands."""

    # Commands
    commands = [
        ("/chat", "Enable normal LLM conversation mode"),
        ("/muse", "Enable web search synthesis mode (like Perplexity)"),
        ("/voice", "Toggle voice mode (see /help voice for details)"),
        ("/style <style>", "Set global response style (concise/standard/comprehensive/custom)"),
        ("/format <format>", "Set global response format (paragraph/bulleted/mixed/academic)"),
        ("/topics", "List conversation topics"),
        ("/topics list", "List all topics with details"),
        ("/topics rename", "Rename ongoing topics"),
        ("/summary", "Summarize conversation (brief/short/standard/detailed/bulleted)"),
        ("/cost", "Show token usage and costs")
    ]

    # Examples
    examples = [
        ("/muse", "Switch to web search mode"),
        ("/style concise", "Set shorter responses for all modes"),
        ("/format bulleted", "Use bullet points for all modes"),
        ("/topics", "See conversation topics")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("💬 Chat & Conversation Management", fg=get_heading_color(), bold=True)
    typer.secho("Mode switching and conversation flow control.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)


def show_settings_help():
    """Show configuration and system management commands."""

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
        ("/script <file>", "Execute commands from a script file"),
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

    # Display header
    typer.secho("⚙️ Settings & System Management", fg=get_heading_color(), bold=True)
    typer.secho("Configure the system and manage models.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Common Settings:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(common_settings, max_width)


def show_search_help():
    """Show knowledge base and muse configuration commands."""

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
        ("/web provider <name>", "Set web search provider for muse mode"),
        ("/set muse-detail <level>", "Set muse detail level (minimal/moderate/detailed/maximum)"),
        ("/set web-search-max-results <n>", "Set number of search results for muse mode")
    ]

    # Examples
    examples = [
        ("/index ~/documents/notes.md", "Index a file"),
        ("/search python functions", "Search knowledge base"),
        ("/set rag-enabled true", "Enable RAG integration"),
        ("/set muse-detail detailed", "More detailed muse responses"),
        ("/style concise", "Set response length for all modes (chat, RAG, muse)"),
        ("/format academic", "Use academic format for all modes (chat, RAG, muse)")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("🔍 Knowledge Base & Muse Configuration", fg=get_heading_color(), bold=True)
    typer.secho("Search your knowledge base and configure muse web search.", fg=get_text_color())
    typer.echo()

    typer.secho("Note: Response style and format are now controlled globally with ", fg=get_text_color(), nl=False)
    typer.secho("/style", fg="cyan", bold=True, nl=False)
    typer.secho(" and ", fg=get_text_color(), nl=False)
    typer.secho("/format", fg="cyan", bold=True, nl=False)
    typer.secho(".", fg=get_text_color())
    typer.secho("Muse-specific settings control detail level and search behavior.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)


def show_history_help():
    """Show navigation and conversation history commands."""

    # Commands
    commands = [
        ("/list", "Show recent conversation nodes"),
        ("/list 10", "Show last 10 nodes"),
        ("/last", "Show the last exchange"),
        ("/show <id>", "Show details of a specific node"),
        ("/print", "Print current node content"),
        ("/print <id>", "Print specific node content"),
        ("/copy", "Copy last response to clipboard"),
        ("/copy <id>", "Copy specific node to clipboard"),
        ("/head", "Show current node"),
        ("/head <id>", "Set current node"),
        ("/history", "Show conversation history (alias for /list)"),
        ("/tree", "Show conversation tree structure")
    ]

    # Navigation examples
    navigation = [
        ("/list", "See recent exchanges"),
        ("/show AB", "View details of node AB"),
        ("/copy", "Copy last LLM response to clipboard"),
        ("/head CD", "Continue from node CD")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + navigation
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("🧭 Navigation & History", fg=get_heading_color(), bold=True)
    typer.secho("Navigate through conversation history and nodes.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Navigation:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(navigation, max_width)


def show_topics_help():
    """Show topic detection and management commands."""

    # Commands
    commands = [
        ("/topics", "List conversation topics (default action)"),
        ("/topics list", "List all topics with details and node boundaries"),
        ("/topics rename", "Rename ongoing topics interactively"),
        ("/topics compress", "Compress current topic to save space"),
        ("/topics index <n>", "Manual topic detection with window size"),
        ("/topics scores", "Show topic detection scores and analysis"),
        ("/topics stats", "Show topic statistics and completion status"),
        ("/topics reanalyze", "Re-detect topics using full conversation context"),
        ("/topics reanalyze apply", "Re-detect topics and save to database"),
        ("/topics reanalyze verbose", "Re-detect with detailed merge history"),
        ("/topics delete <name>", "Delete topic by exact name"),
        ("/topics delete --pattern <pat>", "Delete topics matching pattern"),
        ("/topics delete --time <expr>", "Delete topics by time range")
    ]

    # Examples
    examples = [
        ("/topics", "List current topics"),
        ("/topics index 5", "Detect topics with 5-node window"),
        ("/topics reanalyze", "Preview re-detected topics using elbow detection"),
        ("/topics reanalyze apply", "Apply re-detected topics to database"),
        ("/topics delete --pattern test", "Delete all topics containing 'test'"),
        ("/topics delete --time 'since yesterday'", "Delete topics from yesterday")
    ]

    # Find the longest command across ALL sections for uniform alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("📑 Topic Detection & Management", fg=get_heading_color(), bold=True)
    typer.secho("Manage conversation topics and analyze topic detection.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)


def show_markdown_help():
    """Show markdown file operation commands."""

    # Commands
    commands = [
        ("/out", "Export current topic to markdown"),
        ("/out <spec> [file]", "Export topics to markdown file"),
        ("/in <file>", "Import markdown conversation"),
        ("/files, /ls [dir]", "List markdown files in directory")
    ]

    # Topic specifications
    specs = [
        ("current", "Export current topic (default)"),
        ("3", "Export topic #3"),
        ("1-5", "Export topics 1 through 5"),
        ("1,3,5", "Export topics 1, 3, and 5"),
        ("all", "Export all topics")
    ]

    # Examples
    examples = [
        ("/out", "Save current topic with auto-name"),
        ("/out 1-3 meeting.md", "Save topics 1-3 to meeting.md"),
        ("/in research.md", "Load research.md conversation"),
        ("/in notes.md", "Load notes.md"),
        ("/files", "List markdown files in current directory"),
        ("/ls exports", "List files in exports directory (using alias)")
    ]

    # Find the longest command/spec across ALL sections for uniform alignment
    all_items = commands + [(f"  {s}", d) for s, d in specs] + examples
    max_width = max(len(item) for item, _ in all_items)

    # Display header
    typer.secho("📝 Markdown File Operations", fg=get_heading_color(), bold=True)
    typer.secho("Export, import, and manage markdown conversation files.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Topic Specifications:", fg=get_text_color())
    for spec, desc in specs:
        padding = ' ' * max(1, max_width - len(spec) - 2)
        typer.secho(f"  {spec}{padding}", fg=get_system_color(), nl=False)
        typer.secho(desc, fg=get_text_color())
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)


def show_voice_help():
    """Show voice mode commands."""

    # Commands
    commands = [
        ("/voice", "Toggle voice mode on/off"),
        ("/voice on", "Enable voice mode"),
        ("/voice off", "Disable voice mode"),
        ("/voice status", "Show voice mode status and current settings"),
        ("/voice info", "Show audio devices and test microphone access"),
        ("/voice stt", "Show and configure speech-to-text provider"),
        ("/voice tts", "Show and configure text-to-speech provider"),
    ]

    # STT providers
    stt_providers = [
        ("local_whisper", "Free, runs locally (default)"),
        ("openai_whisper", "Cloud API, excellent accuracy"),
        ("deepgram", "Cloud API, real-time streaming"),
    ]

    # TTS providers
    tts_providers = [
        ("local_piper", "Free, fast, lower quality (default)"),
        ("local_xtts", "Free, high quality, slow first load (~18s)"),
        ("openai_tts", "Cloud API, good quality"),
        ("elevenlabs", "Cloud API, highest quality"),
        ("azure_neural", "Cloud API, DragonHD voices"),
    ]

    # Examples
    examples = [
        ("/voice", "Toggle voice mode"),
        ("/voice info", "Check microphone before enabling"),
        ("/set voice_stt_provider openai_whisper", "Use OpenAI for STT"),
        ("/set voice_tts_provider local_xtts", "Use high-quality local TTS"),
        ("/set voice_wake_word jarvis", "Change wake word (see all with /set voice_wake_word)"),
    ]

    # Find the longest item for alignment
    all_items = commands + [(f"  {p}", d) for p, d in stt_providers] + [(f"  {p}", d) for p, d in tts_providers] + examples
    max_width = max(len(item) for item, _ in all_items)

    # Display header
    typer.secho("🎙️ Voice Mode", fg=get_heading_color(), bold=True)
    typer.secho("Speech input and text-to-speech output for hands-free interaction.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("STT Providers (speech-to-text):", fg=get_text_color())
    for provider, desc in stt_providers:
        padding = ' ' * max(1, max_width - len(provider) - 2)
        typer.secho(f"  {provider}{padding}", fg=get_system_color(), nl=False)
        typer.secho(desc, fg=get_text_color())
    typer.echo()

    typer.secho("TTS Providers (text-to-speech):", fg=get_text_color())
    for provider, desc in tts_providers:
        padding = ' ' * max(1, max_width - len(provider) - 2)
        typer.secho(f"  {provider}{padding}", fg=get_system_color(), nl=False)
        typer.secho(desc, fg=get_text_color())
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)
    typer.echo()

    typer.secho("Voice Commands (while in voice mode):", fg=get_text_color())
    typer.secho('  Say "exit voice" or "voice off" to disable voice mode', fg=get_system_color())


def show_assistant_help():
    """Show assistant utility commands (timers, alarms, weather, etc.)."""

    # Commands
    commands = [
        ("/time", "Show current time"),
        ("/timer <duration> [label]", "Set a timer (e.g., /timer 5m coffee)"),
        ("/timer", "Show active timers"),
        ("/alarm <time> [label]", "Set an alarm (e.g., /alarm 7am wake up)"),
        ("/alarm", "List active alarms"),
        ("/remind <text> in/at <time>", "Set a reminder (e.g., /remind call mom in 1h)"),
        ("/remind", "List active reminders"),
        ("/weather [location]", "Get current weather"),
        ("/forecast [location]", "Get weather forecast"),
        ("/news [category]", "Get news headlines (general, tech, business, science, health, politics, world)"),
        ("/calc <expression>", "Calculate expression (e.g., /calc 15% of 85)"),
        ("/note <text>", "Add a note"),
        ("/note", "List all notes"),
        ("/play <station>", "Play radio station (e.g., /play npr)"),
        ("/pause", "Pause media playback"),
        ("/stop", "Stop current TTS or media"),
        ("/cancel [timer|alarm]", "Cancel timer or alarm"),
        ("/undo", "Undo last utility action"),
        ("/dnd [on|off|duration]", "Do not disturb mode"),
        ("/status", "Show system status (active timers, alarms, media)"),
    ]

    # Examples
    examples = [
        ("/timer 5m tea", "Set a 5-minute tea timer"),
        ("/alarm 7:30am", "Set alarm for 7:30 AM"),
        ("/remind buy milk in 2h", "Reminder in 2 hours"),
        ("/weather", "Weather for current location"),
        ("/news tech", "Technology news headlines"),
        ("/calc 20% of 150", "Calculate 20% of 150"),
    ]

    # Find the longest command for alignment
    all_commands = commands + examples
    max_width = max(len(cmd) for cmd, _ in all_commands)

    # Display header
    typer.secho("🤖 Assistant Utilities", fg=get_heading_color(), bold=True)
    typer.secho("Timers, alarms, reminders, weather, news, and more.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)
    typer.echo()

    typer.secho("Voice Mode:", fg=get_text_color())
    typer.secho("  These commands also work via voice in /voice mode", fg=get_system_color())
    typer.secho("  Example: \"Set a timer for five minutes\"", fg=get_system_color())


def show_mcp_help():
    """Show MCP server commands."""

    # Commands
    commands = [
        ("/mcp", "Show MCP server status (default action)"),
        ("/mcp start", "Start server in background (port 51983)"),
        ("/mcp start --port <port>", "Start on a custom port"),
        ("/mcp start --foreground", "Start in foreground (blocks CLI)"),
        ("/mcp stop", "Stop the MCP server"),
        ("/mcp status", "Show status, PID, port, and uptime"),
        ("/mcp token create <id>", "Create auth token (shown once, save it!)"),
        ("/mcp token list", "List active tokens"),
        ("/mcp token revoke <id>", "Revoke a token by ID"),
        ("/mcp token rotate <id>", "Rotate: create new token, revoke old"),
        ("/mcp traces", "Show recent tool call audit log"),
        ("/mcp traces --tool <name>", "Filter traces by tool name"),
    ]

    # Tools
    tools = [
        ("get_model_info", "Current models and providers"),
        ("get_runtime_state", "Safe runtime config (no secrets)"),
        ("get_topics", "Conversation topics with metadata"),
        ("search_knowledge", "Search RAG knowledge base"),
        ("search_memory", "Search conversation memory"),
        ("ask_llm_stateless", "One-shot LLM query (optional RAG/memory)"),
        ("create_thread", "Create stateful conversation thread"),
        ("ask_llm_stateful", "Send message in a conversation thread"),
        ("index_document", "Add document to RAG knowledge base"),
    ]

    # Examples
    examples = [
        ("/mcp start", "Start the server on default port"),
        ("/mcp token create my-agent", "Create a token for 'my-agent'"),
        ("/mcp traces --tool search_knowledge", "View search_knowledge traces"),
    ]

    # Find the longest item for alignment
    all_items = commands + [(f"  {t}", d) for t, d in tools] + examples
    max_width = max(len(item) for item, _ in all_items)

    # Display header
    typer.secho("🔌 MCP Server (Model Context Protocol)", fg=get_heading_color(), bold=True)
    typer.secho("Expose conversation memory to external AI clients.", fg=get_text_color())
    typer.echo()

    typer.secho("Commands:", fg=get_text_color())
    _display_aligned_commands(commands, max_width)
    typer.echo()

    typer.secho("Available Tools (9):", fg=get_text_color())
    for tool, desc in tools:
        padding = ' ' * max(1, max_width - len(tool) - 2)
        typer.secho(f"  {tool}{padding}", fg=get_system_color(), nl=False)
        typer.secho(desc, fg=get_text_color())
    typer.echo()

    typer.secho("Examples:", fg=get_heading_color(), bold=True)
    _display_aligned_commands(examples, max_width)
