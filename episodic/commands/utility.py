"""
Utility commands for the Episodic CLI.

Includes help, script execution, benchmarking, and other utilities.
"""

import typer
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color
)
from episodic.benchmark import display_benchmark_summary


def help(advanced: bool = False):
    """Show help information with available commands."""
    # Check if we should use the new registry-based help
    try:
        from episodic.cli_registry import show_help_with_categories, show_advanced_help
        if advanced:
            show_advanced_help()
        else:
            show_help_with_categories()
        return
    except ImportError:
        pass
    
    # Fallback to original help
    typer.secho("\n📚 Episodic Commands", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    
    # Navigation commands
    typer.secho("\n🧭 Navigation:", fg=get_heading_color(), bold=True)
    commands = [
        ("/init [--erase]", "Initialize the database"),
        ("/add <content>", "Add a new node manually"),
        ("/show <node_id>", "Show details of a specific node"),
        ("/print [node_id]", "Print node content (current if no ID)"),
        ("/head [node_id]", "Set or show the current head node"),
        ("/list [--count N]", "List recent nodes"),
        ("/ancestry <node_id>", "Show the ancestry chain of a node"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Conversation commands
    typer.secho("\n💬 Conversation:", fg=get_heading_color(), bold=True)
    commands = [
        ("/model [chat|list]", "Show/set models for all contexts"),
        ("/mset [context.param]", "Show/set model parameters"),
        ("/prompt [list|use <name>]", "Manage system prompts"),
        ("/summary [N|all]", "Summarize recent conversation"),
        ("/topics", "List all conversation topics"),
        ("/rename-topics", "Rename ongoing-discussion topics"),
        ("/index <n>", "Detect topics using sliding windows"),
        ("/cost", "Show session cost information"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Configuration commands
    typer.secho("\n⚙️  Configuration:", fg=get_heading_color(), bold=True)
    commands = [
        ("/set [param] [value]", "Configure parameters (e.g., debug, cost)"),
        ("/verify", "Verify database and configuration"),
        ("/config-docs", "Show configuration documentation"),
        ("/benchmark", "Show performance statistics"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Document commands
    typer.secho("\n📄 Documents:", fg=get_heading_color(), bold=True)
    commands = [
        ("/load <pdf_file>", "Load a PDF document"),
        ("/docs [status|enable|disable|clear]", "Manage loaded documents"),
        ("/search <query>", "Search loaded documents"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Compression commands
    typer.secho("\n📦 Compression:", fg=get_heading_color(), bold=True)
    commands = [
        ("/compress", "Compress conversation branch"),
        ("/compress-current-topic", "Compress the current topic"),
        ("/compression-stats", "Show compression statistics"),
        ("/api-stats", "Show LLM API call statistics by thread"),
        ("/reset-api-stats", "Reset LLM API call statistics"),
        ("/compression-queue", "Show pending compressions"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Debug commands
    typer.secho("\n🐛 Debug:", fg=get_heading_color(), bold=True)
    commands = [
        ("/topic-scores [node_id]", "View topic detection scores"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    # Utility commands
    typer.secho("\n🛠️  Utilities:", fg=get_heading_color(), bold=True)
    commands = [
        ("/visualize", "Open graph visualization in browser"),
        ("/script <file>", "Execute commands from a file"),
        ("/save <file>", "Save session commands to script"),
        ("/help", "Show this help message"),
        ("/help-reindex", "Reindex help documentation"),
        ("/exit, /quit, /q", "Exit the application"),
    ]
    
    for cmd, desc in commands:
        padding = ' ' * max(1, 30 - len(cmd) - 2)
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho(desc, fg=get_text_color())
    
    typer.secho("\n" + "=" * 60, fg=get_heading_color())
    typer.secho("💡 Type messages directly to chat, use '/' prefix for commands", 
               fg=get_text_color(), dim=True)
    typer.secho("📝 Examples: /model chat gpt-4, /mset chat.temperature 0.7", 
               fg=get_text_color(), dim=True)




def benchmark():
    """Display benchmark summary."""
    display_benchmark_summary()