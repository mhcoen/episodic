"""
Command routing for Episodic CLI.

This module handles parsing and routing of commands to their respective handlers.
"""

import shlex
import typer
from typing import List, Tuple

from episodic.config import config
from episodic.configuration import EXIT_COMMANDS, get_text_color, get_heading_color, get_system_color
from episodic.cli_helpers import _has_flag
from episodic.benchmark import display_pending_benchmark


def parse_command(command_str: str) -> Tuple[str, List[str]]:
    """
    Parse a command string into command and arguments.
    
    Returns:
        Tuple of (command, arguments)
    """
    try:
        parts = shlex.split(command_str)
    except ValueError:
        # If shlex fails (e.g., unmatched quotes), fall back to simple split
        # This handles contractions like "what's" better
        parts = command_str.split()
    
    if not parts:
        return "", []
    
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []
    
    return cmd, args


def handle_command(command_str: str) -> bool:
    """
    Handle a command string.
    
    Returns:
        bool: True if should exit, False otherwise
    """
    cmd, args = parse_command(command_str)
    
    if not cmd:
        return False
    
    # Check for exit commands (remove leading slash for comparison)
    cmd_without_slash = cmd[1:] if cmd.startswith('/') else cmd
    if cmd_without_slash in EXIT_COMMANDS or cmd_without_slash == "q":
        return True
    
    try:
        # Route to appropriate handler
        if cmd == "/init":
            _handle_init(args)
        elif cmd == "/add":
            _handle_add(args)
        elif cmd == "/show":
            _handle_show(args)
        elif cmd == "/print":
            _handle_print(args)
        elif cmd == "/visualize":
            _handle_visualize(args)
        elif cmd == "/cost":
            _handle_cost()
        elif cmd == "/model":
            _handle_model(args)
        elif cmd == "/mset":
            _handle_mset(args)
        elif cmd == "/style":
            _handle_style(args)
        elif cmd == "/format":
            _handle_format(args)
        elif cmd == "/set":
            _handle_set(args)
        elif cmd == "/reset":
            _handle_reset()
        elif cmd == "/topics":
            _handle_topics(args)
        elif cmd == "/compression":
            _handle_compression(args)
        elif cmd == "/rag":
            _handle_rag(args)
        elif cmd in ["/search", "/s"]:
            _handle_search(args)
        elif cmd in ["/index", "/i"]:
            _handle_index(args)
        elif cmd == "/docs":
            _handle_docs(args)
        elif cmd == "/web":
            _handle_web(args)
        elif cmd == "/muse":
            _handle_muse(args)
        elif cmd == "/prompt":
            _handle_prompt(args)
        elif cmd == "/execute":
            _handle_execute(args)
        elif cmd == "/save":
            _handle_save(args)
        elif cmd == "/export":
            _handle_export(args)
        elif cmd == "/import":
            _handle_import(args)
        elif cmd == "/ls":
            _handle_ls(args)
        elif cmd == "/benchmark":
            _handle_benchmark(args)
        elif cmd == "/reset-benchmarks":
            _handle_reset_benchmarks()
        elif cmd == "/config-docs":
            _handle_config_docs()
        elif cmd == "/verify":
            _handle_verify()
        elif cmd in ["/help", "/h"]:
            _handle_help(args)
        elif cmd == "/help-reindex":
            _handle_help_reindex()
        elif cmd == "/about":
            _handle_about()
        elif cmd == "/welcome":
            _handle_welcome()
        elif cmd == "/config":
            _handle_config()
        elif cmd == "/history":
            _handle_history(args)
        elif cmd == "/tree":
            _handle_tree(args)
        elif cmd == "/graph":
            _handle_graph(args)
        elif cmd == "/summary":
            _handle_summary(args)
        elif cmd == "/list":
            _handle_list(args)
        elif cmd == "/last":
            _handle_last(args)
        else:
            # Check if it's a deprecated command
            _handle_deprecated_commands(cmd, args)
    except Exception as e:
        typer.secho(f"Error executing command: {e}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")
    
    # Display any pending benchmarks after command execution
    display_pending_benchmark()
    
    # Add blank line before next prompt
    typer.echo()
    
    return False


# Individual command handlers
def _handle_init(args: List[str]):
    """Handle /init command."""
    from episodic.commands import init
    erase = _has_flag(args, ["--erase", "-e"])
    init(erase=erase)


def _handle_add(args: List[str]):
    """Handle /add command."""
    if not args:
        typer.secho("Usage: /add <content>", fg="red")
    else:
        from episodic.commands import add
        content = " ".join(args)
        add(content)


def _handle_show(args: List[str]):
    """Handle /show command."""
    if not args:
        typer.secho("Usage: /show <node_id>", fg="red")
    else:
        from episodic.commands import show
        show(args[0])


def _handle_print(args: List[str]):
    """Handle /print command."""
    from episodic.commands import print_node
    
    if args:
        node_id = args[0]
        print_node(node_id)
    else:
        print_node()


def _handle_visualize(args: List[str]):
    """Handle /visualize command."""
    from episodic.commands import visualize
    visualize()


def _handle_cost():
    """Handle /cost command."""
    from episodic.commands import cost
    cost()


def _handle_model(args: List[str]):
    """Handle /model command."""
    # Use unified model command if it exists, otherwise fall back to simple model command
    try:
        from episodic.commands.unified_model import model_command
        
        if not args:
            model_command(None, None)
        elif len(args) == 1:
            # Either "list" or a context name without model
            model_command(args[0], None)
        else:
            # Context and model name
            context = args[0]
            model_name = " ".join(args[1:])
            model_command(context, model_name)
    except ImportError:
        # Fall back to simple model command
        from episodic.commands import handle_model
        
        if not args:
            handle_model()
        else:
            # Just pass the model name for backward compatibility
            model_name = " ".join(args)
            handle_model(model_name)


def _handle_mset(args: List[str]):
    """Handle /mset command."""
    from episodic.commands.mset import mset_command
    
    if not args:
        mset_command(None, None)
    elif len(args) == 1:
        # Could be just context or embedding.list
        if args[0] == "embedding":
            mset_command(args[0], None)
        else:
            mset_command(args[0], None)
    elif args[0] == "embedding" and args[1] == "list" and len(args) == 2:
        # Special handling for embedding list
        from episodic.commands.mset import list_embedding_models
        list_embedding_models()
    elif len(args) == 2:
        # parameter value format
        mset_command(args[0], args[1])
    else:
        # Join remaining args as value
        mset_command(args[0], " ".join(args[1:]))


def _handle_style(args: List[str]):
    """Handle /style command."""
    from episodic.commands.style import handle_style
    handle_style(args)


def _handle_format(args: List[str]):
    """Handle /format command."""
    from episodic.commands.style import handle_format
    handle_format(args)


def _handle_set(args: List[str]):
    """Handle /set command."""
    from episodic.commands import set
    
    if not args:
        # No arguments - show current settings
        set(None, None)
    elif len(args) == 1:
        # Only parameter provided - show current value or handle 'all'
        set(args[0], None)
    else:
        # Parameter and value provided
        param = args[0]
        value = " ".join(args[1:])
        set(param, value)


def _handle_reset():
    """Handle /reset command."""
    from episodic.commands import reset
    reset()


def _handle_topics(args: List[str]):
    """Handle /topics command."""
    from episodic.commands.unified_topics import handle_topics_action
    
    if not args:
        handle_topics_action()
    else:
        action = args[0]
        action_args = args[1:]
        
        if action == "rename":
            handle_topics_action(action="rename")
        elif action == "compress":
            handle_topics_action(action="compress")
        elif action == "index":
            if action_args:
                handle_topics_action(action="index", window_size=int(action_args[0]))
            else:
                typer.secho("Usage: /topics index <number>", fg="red")
        elif action == "scores":
            handle_topics_action(action="scores")
        elif action == "stats":
            handle_topics_action(action="stats")
        elif action == "list":
            handle_topics_action(action="list")
        else:
            typer.secho(f"Unknown topics action: {action}", fg="red")
            typer.secho("Available actions: list, rename, compress, index, scores, stats", fg="yellow")


def _handle_compression(args: List[str]):
    """Handle /compression command."""
    from episodic.commands.unified_compression import compression_command
    
    if not args:
        compression_command()
    else:
        action = args[0]
        if action in ["stats", "queue", "compress", "api-stats", "reset-api"]:
            compression_command(action=action)
        else:
            typer.secho(f"Unknown compression action: {action}", fg="red")


def _handle_rag(args: List[str]):
    """Handle /rag command."""
    from episodic.commands.rag import rag_toggle, rag_stats
    
    if not args:
        # Show RAG stats by default
        rag_stats()
    elif args[0] == "on":
        rag_toggle(enable=True)
    elif args[0] == "off":
        rag_toggle(enable=False)
    else:
        typer.secho(f"Unknown rag action: {args[0]}", fg="red")


def _handle_search(args: List[str]):
    """Handle /search or /s command."""
    if not args:
        typer.secho("Usage: /search <query>", fg="red")
    else:
        from episodic.commands.rag import search
        query = " ".join(args)
        search(query)


def _handle_index(args: List[str]):
    """Handle /index or /i command."""
    from episodic.commands.rag import index_text, index_file
    
    if not args:
        typer.secho("Usage: /index <file_path> or /index --text \"<content>\"", fg="red")
    else:
        if args[0] == "--text" and len(args) > 1:
            # Index raw text
            text = " ".join(args[1:])
            index_text(content=text)
        else:
            # Index file
            file_path = " ".join(args)
            index_file(filepath=file_path)


def _handle_docs(args: List[str]):
    """Handle /docs command."""
    from episodic.commands.rag import docs_command
    
    # Pass all args to docs_command which handles the parsing
    docs_command(*args)



def _handle_web(args: List[str]):
    """Handle /web command."""
    from episodic.commands.web_provider import web_command
    
    if not args:
        # No args - show current provider
        web_command()
    else:
        # Pass subcommand and remaining args
        subcommand = args[0]
        remaining_args = args[1:] if len(args) > 1 else []
        if remaining_args:
            web_command(subcommand, ' '.join(remaining_args))
        else:
            web_command(subcommand, None)


def _handle_muse(args: List[str]):
    """Handle /muse command."""
    from episodic.commands.muse import muse
    
    if not args:
        muse()
    elif args[0] in ["on", "off"]:
        muse(action=args[0])
    else:
        typer.secho(f"Unknown muse action: {args[0]}", fg="red")


def _handle_prompt(args: List[str]):
    """Handle /prompt command."""
    from episodic.commands.prompt import prompt
    
    if not args:
        prompt()
    else:
        action = args[0]
        if action == "list":
            prompt(action="list")
        elif action == "set" and len(args) > 1:
            prompt(action="set", prompt_name=args[1])
        elif action == "show":
            prompt_name = args[1] if len(args) > 1 else None
            prompt(action="show", prompt_name=prompt_name)
        else:
            typer.secho(f"Unknown prompt action: {action}", fg="red")


def _handle_execute(args: List[str]):
    """Handle /execute command."""
    if not args:
        typer.secho("Usage: /execute <script_file>", fg="red")
    else:
        from episodic.cli_session import execute_script
        filename = " ".join(args)
        execute_script(filename)


def _handle_save(args: List[str]):
    """Handle /save command."""
    if not args:
        typer.secho("Usage: /save <filename>", fg="red")
    else:
        from episodic.cli_session import save_session_script
        filename = " ".join(args)
        save_session_script(filename)
        typer.secho(f"✅ Session saved to scripts/{filename}", fg="green")


def _handle_benchmark(args: List[str]):
    """Handle /benchmark command."""
    from episodic.commands import benchmark
    
    if not args:
        benchmark()
    else:
        action = args[0]
        if action == "on":
            benchmark(enable=True)
        elif action == "off":
            benchmark(enable=False)
        else:
            typer.secho(f"Unknown benchmark action: {action}", fg="red")


def _handle_reset_benchmarks():
    """Handle /reset-benchmarks command."""
    from episodic.benchmark import reset_benchmarks
    reset_benchmarks()
    typer.secho("✅ Benchmark data reset", fg="green")


def _handle_config_docs():
    """Handle /config-docs command."""
    from episodic.commands import config_docs
    config_docs()


def _handle_verify():
    """Handle /verify command."""
    from episodic.commands import verify
    verify()


def _handle_help(args: List[str]):
    """Handle /help command."""
    from episodic.commands import help
    
    if args:
        # Pass the argument as query
        query = " ".join(args)
        help(query=query)
    else:
        help()


def _handle_help_reindex():
    """Handle /help-reindex command."""
    from episodic.commands.help import help_reindex
    help_reindex()


def _handle_deprecated_commands(cmd: str, args: List[str]):
    """Handle deprecated commands with warnings."""
    # Deprecated commands mapping
    deprecated_commands = {
        "/rename-topics": ("topics", "rename"),
        "/compress-current-topic": ("topics", "compress"),
        "/api-stats": ("compression", "api-stats"),
        "/reset-api-stats": ("compression", "reset-api"),
        "/count-tokens": ("cost", None),
    }
    
    if cmd in deprecated_commands:
        new_cmd, action = deprecated_commands[cmd]
        if action:
            typer.secho(f"⚠️  '{cmd}' is deprecated. Use '/{new_cmd} {action}' instead.", 
                       fg="yellow")
            # Execute the new command
            if new_cmd == "topics":
                _handle_topics([action] if action else [])
            elif new_cmd == "compression":
                _handle_compression([action] if action else [])
        else:
            typer.secho(f"⚠️  '{cmd}' is deprecated. Use '/{new_cmd}' instead.", 
                       fg="yellow")
            if new_cmd == "cost":
                _handle_cost()
    else:
        typer.secho(f"Unknown command: {cmd}. Type /help for available commands.", 
                   fg=get_text_color())


def _handle_about():
    """Handle /about command."""
    typer.secho("\n📚 About Episodic", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    typer.secho("\nEpisodic is a conversational DAG-based memory agent that creates", fg=get_text_color())
    typer.secho("persistent, navigable conversations with language models.", fg=get_text_color())
    typer.secho("\nKey Features:", fg=get_heading_color(), bold=True)
    typer.secho("  • Conversation history stored as a directed acyclic graph (DAG)", fg=get_text_color())
    typer.secho("  • Automatic topic detection and organization", fg=get_text_color())
    typer.secho("  • Support for multiple LLM providers", fg=get_text_color())
    typer.secho("  • RAG (Retrieval Augmented Generation) capabilities", fg=get_text_color())
    typer.secho("  • Web search integration", fg=get_text_color())
    typer.secho("  • Conversation compression and summarization", fg=get_text_color())
    typer.secho("\nVersion: 0.1.0", fg=get_system_color())
    typer.secho("Repository: https://github.com/yourusername/episodic", fg=get_system_color())


def _handle_welcome():
    """Handle /welcome command."""
    from episodic.cli_display import display_welcome, display_model_info
    display_welcome()
    display_model_info()


def _handle_config():
    """Handle /config command - show current configuration."""
    from episodic.commands.settings_display import display_all_settings
    from episodic.config import config
    
    # Get values for context depth and semdepth
    context_depth = config.get("depth", 10)
    semdepth = config.get("semdepth", 0)
    
    typer.secho("\n⚙️  Current Configuration", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    display_all_settings(context_depth, semdepth)


def _handle_history(args: List[str]):
    """Handle /history command - show conversation history."""
    # This is essentially the same as /list command
    limit = int(args[0]) if args else 20
    
    from episodic.commands.navigation import list as list_command
    list_command(count=limit)


def _handle_tree(args: List[str]):
    """Handle /tree command - show conversation tree structure."""
    # For now, show ancestry of current node or specified node
    if args:
        from episodic.commands.navigation import ancestry
        ancestry(args[0])
    else:
        # Show ancestry of current node
        from episodic.conversation import conversation_manager
        from episodic.commands.navigation import ancestry
        
        current_id = conversation_manager.get_current_node_id()
        if current_id:
            from episodic.db import get_node
            node = get_node(current_id)
            if node:
                ancestry(node['short_id'])
        else:
            typer.secho("No current node. Use '/tree <node_id>' to show a specific node's tree.", 
                       fg=get_system_color())


def _handle_graph(args: List[str]):
    """Handle /graph command - show conversation graph visualization."""
    # This is the same as /visualize
    _handle_visualize(args)


def _handle_summary(args: List[str]):
    """Handle /summary command."""
    from episodic.commands.summary import summary
    
    if not args:
        summary()
    elif len(args) == 1:
        # Could be a number or "all"
        arg = args[0]
        if arg.lower() == "all":
            summary("all")
        else:
            try:
                count = int(arg)
                summary(count)
            except ValueError:
                typer.secho("Invalid argument. Use a number or 'all'", fg="red")
    else:
        typer.secho("Usage: /summary [count|all]", fg="red")


def _handle_list(args: List[str]):
    """Handle /list command."""
    from episodic.commands.navigation import list as list_command
    from episodic.configuration import DEFAULT_LIST_COUNT
    
    if not args:
        list_command()
    else:
        try:
            count = int(args[0])
            list_command(count=count)
        except ValueError:
            typer.secho("Usage: /list [count]", fg="red")


def _handle_last(args: List[str]):
    """Handle /last command - show the last conversation exchange."""
    from episodic.commands.navigation import list as list_command
    
    # Show just the last exchange (count=1)
    list_command(count=1)


def _handle_export(args: List[str]):
    """Handle /export command."""
    from episodic.commands.save import save_command
    
    # Join args back into a single string for parsing
    args_str = " ".join(args) if args else ""
    save_command(args_str)


def _handle_import(args: List[str]):
    """Handle /import command."""
    from episodic.commands.resume import resume_command
    
    if not args:
        typer.secho("Usage: /import <filename.md>", fg="red")
    else:
        filepath = " ".join(args)
        resume_command(filepath)


def _handle_ls(args: List[str]):
    """Handle /ls command."""
    from episodic.commands.ls import ls_command
    
    # Pass directory or None for current directory
    directory = " ".join(args) if args else None
    ls_command(directory)