"""
Command registry for Episodic CLI.

This module provides a centralized registry of all available commands,
making it easier to discover, document, and maintain commands.
"""

from typing import Dict, Callable, List, Optional
from dataclasses import dataclass


@dataclass
class CommandInfo:
    """Information about a registered command."""
    name: str
    handler: Callable
    description: str
    category: str
    aliases: List[str] = None
    deprecated: bool = False
    replacement: Optional[str] = None


class CommandRegistry:
    """Central registry for all CLI commands."""
    
    def __init__(self):
        self._commands: Dict[str, CommandInfo] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str,
        category: str,
        aliases: List[str] = None,
        deprecated: bool = False,
        replacement: Optional[str] = None
    ):
        """Register a command."""
        info = CommandInfo(
            name=name,
            handler=handler,
            description=description,
            category=category,
            aliases=aliases or [],
            deprecated=deprecated,
            replacement=replacement
        )
        
        # Register main command
        self._commands[name] = info
        
        # Register aliases
        for alias in info.aliases:
            self._commands[alias] = info
        
        # Track by category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def get_command(self, name: str) -> Optional[CommandInfo]:
        """Get command info by name or alias."""
        return self._commands.get(name)
    
    def get_commands_by_category(self) -> Dict[str, List[CommandInfo]]:
        """Get all commands organized by category."""
        result = {}
        for category, names in self._categories.items():
            result[category] = [self._commands[name] for name in names]
        return result
    
    def is_deprecated(self, name: str) -> bool:
        """Check if a command is deprecated."""
        info = self.get_command(name)
        return info.deprecated if info else False
    


# Create global registry instance
command_registry = CommandRegistry()


def register_all_commands():
    """Register all available commands."""
    # Import commands
    from episodic.commands import (
        # Navigation
        init, add, show, print_node, head, ancestry, list_nodes, last_exchange,
        # Settings  
        set, verify, cost, model_params, config_docs, reset,
        # Topics
        compress,
        # Other
        prompts, summary, benchmark, help
    )
    
    # Import new utility commands
    from episodic.cli_command_router import (
        _handle_about, _handle_welcome, _handle_config,
        _handle_history, _handle_tree
    )
    
    # Import script commands
    from episodic.commands.scripts import scripts_command
    
    # Import save/load commands
    from episodic.commands.save_load import save_command, load_command, files_command
    
    # Import mode commands
    from episodic.commands.mode import handle_muse, handle_chat, mode_command
    
    # Import interface mode commands
    from episodic.commands.interface_mode import simple_mode_command, advanced_mode_command
    
    # Import new topic command
    from episodic.commands.new_topic import new_command, clear_command
    
    # Import style and format commands
    from episodic.commands.style import handle_style, handle_format
    
    # Import detail command
    from episodic.commands.detail import detail
    
    # Import theme command
    from episodic.commands.theme import theme_command
    
    # Import unified commands
    from episodic.commands.unified_topics import topics_command
    from episodic.commands.unified_compression import compression_command
    from episodic.commands.unified_model import model_command
    from episodic.commands.mset import mset_command
    
    # RAG commands will be imported lazily when needed
    rag_available = True  # We'll check availability when actually used
    
    # Import web provider commands
    try:
        from episodic.commands.web_provider import web_command
        web_available = True
    except ImportError:
        web_available = False
    
    # Import reflection command
    from episodic.commands.reflection import reflection_command
    
    # Import debug command
    from episodic.commands.debug import app as debug_app
    
    # Import dev command
    from episodic.commands.dev import dev
    
    # Register navigation commands
    command_registry.register("init", init, "Initialize the database", "Navigation")
    command_registry.register("add", add, "Add a new node manually", "Navigation")
    command_registry.register("show", show, "Show details of a specific node", "Navigation")
    command_registry.register("print", print_node, "Print node content", "Navigation")
    command_registry.register("head", head, "Set or show the current head node", "Navigation")
    command_registry.register("ancestry", ancestry, "Show the ancestry chain of a node", "Navigation")
    command_registry.register("list", list_nodes, "List recent nodes", "Navigation")
    command_registry.register("last", last_exchange, "Show the last conversation exchange", "Navigation")
    
    # Register unified commands (new style)
    command_registry.register(
        "topics", topics_command, 
        "Manage conversation topics", 
        "Topics"
    )
    command_registry.register(
        "compression", compression_command,
        "Manage conversation compression",
        "Compression"  
    )
    
    
    # Register settings commands (keep both old and new)
    command_registry.register("set", set, "Configure parameters", "Configuration")
    command_registry.register("verify", verify, "Verify configuration", "Configuration")
    command_registry.register("cost", cost, "Show session cost", "Configuration")
    command_registry.register("model-params", model_params, "Show/set model parameters", "Configuration", 
                            aliases=["mp"], deprecated=True, replacement="mset")
    command_registry.register("config-docs", config_docs, "Show configuration documentation", "Configuration")
    command_registry.register("reset", reset, "Reset parameters to defaults", "Configuration")
    
    # Register new unified model commands
    command_registry.register("model", model_command, "Manage models for all contexts (chat/detection/compression/synthesis)", "Configuration")
    command_registry.register("mset", mset_command, "Set model parameters (e.g., mset chat.temperature 0.7)", "Configuration")
    command_registry.register("style", handle_style, "Set global response style (concise/standard/comprehensive/custom)", "Configuration")
    command_registry.register("format", handle_format, "Set global response format (paragraph/bulleted/mixed/academic)", "Configuration")
    command_registry.register("detail", detail, "Set response detail level (minimal/moderate/detailed/maximum)", "Configuration")
    command_registry.register("theme", theme_command, "Manage color themes", "Configuration")
    command_registry.register("prompt", prompts, "Manage system prompts", "Conversation", aliases=["prompts"])
    command_registry.register("summary", summary, "Summarize recent conversation", "Conversation")
    command_registry.register("muse", handle_muse, "Enable muse mode (web search synthesis)", "Conversation")
    command_registry.register("chat", handle_chat, "Enable chat mode (normal LLM conversation)", "Conversation")

    # Critique command
    from episodic.commands.critique import critique_command
    command_registry.register("critique", critique_command, "Critique last response using critic model", "Conversation")
    command_registry.register("mode", mode_command, "Switch between local and cloud provider modes", "Configuration")

    # Voice command with lazy loading
    def lazy_voice_command(*args, **kwargs):
        from episodic.commands.voice import voice
        return voice(*args, **kwargs)

    command_registry.register("voice", lazy_voice_command, "Toggle voice mode (speech input/output)", "Conversation")
    command_registry.register("benchmark", benchmark, "Show performance statistics", "Utility")
    command_registry.register("help", help, "Show help information", "Utility", aliases=["h"])
    command_registry.register("about", _handle_about, "Show information about Episodic", "Utility")
    command_registry.register("welcome", _handle_welcome, "Show welcome message", "Utility")
    command_registry.register("config", _handle_config, "Show current configuration", "Configuration")
    command_registry.register("history", _handle_history, "Show conversation history", "Navigation")
    command_registry.register("tree", _handle_tree, "Show conversation tree structure", "Navigation")
    command_registry.register("script", scripts_command, "Manage session scripts (save/run/list)", "Utility")

    # Pause command for scripts
    def pause_command():
        """Pause script execution until user presses Enter."""
        input("Press Enter to continue...")

    command_registry.register("pause", pause_command, "Pause script execution until Enter is pressed", "Utility")
    command_registry.register("compress", compress, "Compress a topic or branch", "Compression")
    
    # Register simple mode commands
    command_registry.register("save", save_command, "Save conversation to markdown file", "Conversation")
    command_registry.register("load", load_command, "Load conversation from markdown file", "Conversation")
    command_registry.register("files", files_command, "List saved conversations", "Conversation")
    command_registry.register("new", new_command, "Start a fresh topic/conversation", "Conversation")
    command_registry.register("clear", clear_command, "Clear context and start fresh (alias for /new)", "Conversation")
    
    # Register interface mode switching commands
    command_registry.register("simple", simple_mode_command, "Switch to simple mode (10 essential commands)", "Configuration")
    command_registry.register("advanced", advanced_mode_command, "Switch to advanced mode (all commands)", "Configuration")
    
    # Register RAG commands with lazy loading wrappers
    def lazy_rag_toggle(*args, **kwargs):
        from episodic.commands.rag import rag_toggle
        return rag_toggle(*args, **kwargs)
    
    def lazy_search(*args, **kwargs):
        from episodic.commands.rag import search
        return search(*args, **kwargs)
    
    def lazy_index_file(*args, **kwargs):
        from episodic.commands.rag import index_file
        return index_file(*args, **kwargs)
    
    def lazy_docs_command(*args, **kwargs):
        from episodic.commands.rag import docs_command
        return docs_command(*args, **kwargs)
    
    if rag_available:
        command_registry.register("rag", lazy_rag_toggle, "Enable/disable RAG or show stats", "Knowledge Base")
        command_registry.register("search", lazy_search, "Search the knowledge base", "Knowledge Base", aliases=["s"])
        command_registry.register("index", lazy_index_file, "Index a file or text into knowledge base", "Knowledge Base", aliases=["i"])
        command_registry.register("docs", lazy_docs_command, "Manage documents (list/show/remove/clear)", "Knowledge Base")
    
    # Register web provider commands if available
    if web_available:
        command_registry.register("web", web_command, "Manage web search providers", "Configuration")
    
    # Register reflection command
    command_registry.register("reflect", reflection_command, "Enable multi-step reflection and reasoning", "Conversation")
    
    # Register debug command
    command_registry.register("debug", debug_app, "Manage debug output categories", "Configuration")
    
    # Developer commands - hidden from normal help
    command_registry.register("dev", dev, "Developer maintenance commands", "Developer")
    
    # Test mode command (in Utility so it shows in /help)
    from episodic.commands.test_mode import test_command
    command_registry.register("test", test_command, "Manage test mode for isolated testing", "Utility")

    # Clipboard command
    from episodic.commands.clipboard import copy_command
    command_registry.register("copy", copy_command, "Copy last response to clipboard", "Utility")

    # Register memory commands if RAG is available
    if rag_available:
        # Use lazy loading for memory commands
        def lazy_memory_command(*args, **kwargs):
            from episodic.commands.memory import memory_command
            return memory_command(*args, **kwargs)
        
        def lazy_forget_command(*args, **kwargs):
            from episodic.commands.memory import forget_command
            return forget_command(*args, **kwargs)
        
        def lazy_memory_stats_command(*args, **kwargs):
            from episodic.commands.memory import memory_stats_command
            return memory_stats_command(*args, **kwargs)
        
        command_registry.register("memory", lazy_memory_command, "View and search memory entries", "Knowledge Base")
        command_registry.register("forget", lazy_forget_command, "Remove memory entries", "Knowledge Base")
        command_registry.register("memory-stats", lazy_memory_stats_command, "Show memory system statistics", "Knowledge Base")
        
        # Migration command
        def lazy_migrate_command(*args, **kwargs):
            from episodic.commands.migrate import migrate_command
            return migrate_command(*args, **kwargs)
        
        command_registry.register("migrate", lazy_migrate_command, "Migrate RAG data to multi-collection system", "Utility")

    # Maintenance command (lazy loading)
    def lazy_maintenance_command(*args, **kwargs):
        from episodic.commands.maintenance import maintenance_command
        return maintenance_command(*args, **kwargs)

    command_registry.register("maintenance", lazy_maintenance_command, "Database maintenance tasks (summarize topics, etc.)", "Utility")

    # Evaluation command (lazy loading)
    def lazy_evaluate_command(*args, **kwargs):
        from episodic.commands.evaluate import evaluate_command
        return evaluate_command(*args, **kwargs)

    command_registry.register("evaluate", lazy_evaluate_command, "Evaluation and calibration tools (reactivation replay, etc.)", "Utility")

    # Doctor command (installation health check)
    def lazy_doctor_command(*args, **kwargs):
        from episodic.commands.doctor import doctor
        return doctor(*args, **kwargs)

    command_registry.register("doctor", lazy_doctor_command, "Run installation health checks", "Utility")

    # MCP server command (lazy loading)
    def lazy_mcp_command(*args, **kwargs):
        from episodic.commands.mcp_cmd import mcp_command
        return mcp_command(*args, **kwargs)

    command_registry.register("mcp", lazy_mcp_command, "Manage MCP server and external MCP clients", "Configuration")

    # KG command (lazy loading)
    def lazy_kg_command(*args, **kwargs):
        from episodic.commands.kg import kg_command
        return kg_command(*args, **kwargs)

    command_registry.register("kg", lazy_kg_command, "Knowledge graph visualization and queries", "Knowledge Base")

    # Rollback command (lazy loading)
    def lazy_rollback_command(*args, **kwargs):
        from episodic.commands.rollback import rollback_command
        return rollback_command(*args, **kwargs)

    command_registry.register("rollback", lazy_rollback_command, "Roll back conversation to a specific node", "Navigation")


# Don't initialize on import - will be called when needed
# register_all_commands()