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
    
    def get_replacement(self, name: str) -> Optional[str]:
        """Get replacement command for deprecated command."""
        info = self.get_command(name)
        return info.replacement if info else None


# Create global registry instance
command_registry = CommandRegistry()


def register_all_commands():
    """Register all available commands."""
    # Import commands
    from episodic.commands import (
        # Navigation
        init, add, show, print_node, head, ancestry, list_nodes,
        # Settings  
        set, verify, cost, model_params, config_docs,
        # Topics
        topics, compress_current_topic, rename_ongoing_topics,
        # Compression
        compress, compression_stats, compression_queue, api_call_stats, reset_api_stats,
        # Other
        visualize, prompts, summary, benchmark, help, handle_model
    )
    from episodic.commands.index_topics import index_topics
    from episodic.commands.debug_topics import topic_scores
    
    # Import unified commands
    from episodic.commands.unified_topics import topics_command
    from episodic.commands.unified_compression import compression_command
    from episodic.commands.unified_settings import settings_command
    
    # Import RAG commands
    try:
        from episodic.commands.rag import (
            search, index_text, index_file, rag_toggle, rag_stats, docs_command
        )
        rag_available = True
    except ImportError:
        rag_available = False
    
    # Import web search commands
    try:
        from episodic.commands.web_search import (
            websearch, websearch_command, websearch_toggle, 
            websearch_config, websearch_stats, websearch_cache_clear
        )
        websearch_available = True
    except ImportError:
        websearch_available = False
    
    # Register navigation commands
    command_registry.register("init", init, "Initialize the database", "Navigation")
    command_registry.register("add", add, "Add a new node manually", "Navigation")
    command_registry.register("show", show, "Show details of a specific node", "Navigation")
    command_registry.register("print", print_node, "Print node content", "Navigation")
    command_registry.register("head", head, "Set or show the current head node", "Navigation")
    command_registry.register("ancestry", ancestry, "Show the ancestry chain of a node", "Navigation")
    command_registry.register("list", list_nodes, "List recent nodes", "Navigation")
    
    # Register unified commands (new style)
    command_registry.register(
        "topics", topics_command, 
        "Manage topics (list/rename/compress/index/stats)", 
        "Topics"
    )
    command_registry.register(
        "compression", compression_command,
        "Manage compression (stats/queue/compress)",
        "Compression"  
    )
    command_registry.register(
        "settings", settings_command,
        "Manage settings (show/set/verify/docs)",
        "Configuration"
    )
    
    # Register old commands as deprecated
    command_registry.register(
        "rename-topics", rename_ongoing_topics,
        "Rename ongoing topics", "Topics",
        deprecated=True, replacement="topics rename"
    )
    command_registry.register(
        "compress-current-topic", compress_current_topic,
        "Compress current topic", "Topics",
        deprecated=True, replacement="topics compress"
    )
    command_registry.register(
        "index", index_topics,
        "Index topics manually", "Topics", 
        deprecated=True, replacement="topics index"
    )
    command_registry.register(
        "topic-scores", topic_scores,
        "Show topic detection scores", "Topics",
        deprecated=True, replacement="topics scores"
    )
    command_registry.register(
        "compression-stats", compression_stats,
        "Show compression statistics", "Compression",
        deprecated=True, replacement="compression stats"
    )
    command_registry.register(
        "compression-queue", compression_queue,
        "Show compression queue", "Compression",
        deprecated=True, replacement="compression queue"
    )
    command_registry.register(
        "api-stats", api_call_stats,
        "Show API call statistics", "Compression",
        deprecated=True, replacement="compression api-stats"
    )
    command_registry.register(
        "reset-api-stats", reset_api_stats,
        "Reset API statistics", "Compression",
        deprecated=True, replacement="compression reset-api"
    )
    
    # Register settings commands (keep both old and new)
    command_registry.register("set", set, "Configure parameters", "Configuration")
    command_registry.register("verify", verify, "Verify configuration", "Configuration")
    command_registry.register("cost", cost, "Show session cost", "Configuration")
    command_registry.register("model-params", model_params, "Show/set model parameters", "Configuration", aliases=["mp"])
    command_registry.register("config-docs", config_docs, "Show configuration documentation", "Configuration")
    
    # Register other commands
    command_registry.register("model", handle_model, "Switch or show language model", "Conversation")
    command_registry.register("prompts", prompts, "Manage system prompts", "Conversation")
    command_registry.register("summary", summary, "Summarize recent conversation", "Conversation")
    command_registry.register("visualize", visualize, "Visualize conversation graph", "Utility")
    command_registry.register("benchmark", benchmark, "Show performance statistics", "Utility")
    command_registry.register("help", help, "Show help information", "Utility")
    command_registry.register("compress", compress, "Compress a topic or branch", "Compression")
    
    # Register RAG commands if available
    if rag_available:
        command_registry.register("rag", rag_toggle, "Enable/disable RAG or show stats", "Knowledge Base")
        command_registry.register("search", search, "Search the knowledge base", "Knowledge Base", aliases=["s"])
        command_registry.register("index", index_file, "Index a file or text into knowledge base", "Knowledge Base", aliases=["i"])
        command_registry.register("docs", docs_command, "Manage documents (list/show/remove/clear)", "Knowledge Base")
    
    # Register web search commands if available
    if websearch_available:
        command_registry.register("websearch", websearch_command, "Search the web for current information", "Knowledge Base", aliases=["ws"])


# Initialize registry on import
register_all_commands()