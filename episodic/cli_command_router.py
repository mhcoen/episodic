"""
Command routing for Episodic CLI.

This module handles parsing and routing of commands to their respective handlers.
"""

import shlex
import typer
from typing import List, Tuple, Optional

from episodic.config import config
from episodic.configuration import EXIT_COMMANDS, get_text_color
from episodic.cli_helpers import _parse_flag_value, _has_flag
from episodic.benchmark import display_pending_benchmark


def parse_command(command_str: str) -> Tuple[str, List[str]]:
    """
    Parse a command string into command and arguments.
    
    Returns:
        Tuple of (command, arguments)
    """
    try:
        parts = shlex.split(command_str)
    except ValueError as e:
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
        elif cmd == "/models":
            _handle_models()
        elif cmd == "/mset":
            _handle_mset(args)
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
        elif cmd in ["/websearch", "/ws"]:
            _handle_websearch(args)
        elif cmd == "/muse":
            _handle_muse(args)
        elif cmd == "/prompt":
            _handle_prompt(args)
        elif cmd == "/execute":
            _handle_execute(args)
        elif cmd == "/save":
            _handle_save(args)
        elif cmd == "/benchmark":
            _handle_benchmark(args)
        elif cmd == "/reset-benchmarks":
            _handle_reset_benchmarks()
        elif cmd == "/config-docs":
            _handle_config_docs()
        elif cmd == "/verify":
            _handle_verify()
        elif cmd == "/help":
            _handle_help(args)
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
    from episodic.commands import print_conversation
    limit = int(args[0]) if args else 10
    print_conversation(limit)


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
    from episodic.commands import model
    
    if not args:
        model()
    elif args[0] == "list":
        model(action="list")
    else:
        # Check if it's a context-specific model setting
        if len(args) >= 2 and args[0] in ["chat", "detection", "compression", "synthesis"]:
            context = args[0]
            model_name = " ".join(args[1:])
            model(model_name=model_name, context=context)
        else:
            # Default to chat context for backward compatibility
            model_name = " ".join(args)
            model(model_name=model_name)


def _handle_models():
    """Handle /models command."""
    from episodic.commands import model
    model(action="list")


def _handle_mset(args: List[str]):
    """Handle /mset command."""
    from episodic.commands.mset import mset
    
    if not args:
        mset()
    elif len(args) == 1:
        # Show parameters for a specific context
        mset(context=args[0])
    elif args[0] == "embedding" and len(args) > 1:
        # Special handling for embedding configuration
        if args[1] == "list":
            from episodic.commands.mset import list_embedding_models
            list_embedding_models()
        else:
            # Parse as parameter.value
            param_str = " ".join(args)
            mset(param_str=param_str)
    else:
        # Parse as parameter.value or context.parameter.value
        param_str = " ".join(args)
        mset(param_str=param_str)


def _handle_set(args: List[str]):
    """Handle /set command."""
    from episodic.commands import set
    
    if len(args) < 2:
        typer.secho("Usage: /set <parameter> <value>", fg="red")
    else:
        param = args[0]
        value = " ".join(args[1:])
        set(param, value)


def _handle_reset():
    """Handle /reset command."""
    from episodic.commands import reset
    reset()


def _handle_topics(args: List[str]):
    """Handle /topics command."""
    from episodic.commands.unified_topics import topics
    
    if not args:
        topics()
    else:
        action = args[0]
        action_args = args[1:]
        
        if action == "rename":
            topics(action="rename")
        elif action == "compress":
            topics(action="compress")
        elif action == "index":
            if action_args:
                topics(action="index", messages=int(action_args[0]))
            else:
                typer.secho("Usage: /topics index <number>", fg="red")
        elif action == "scores":
            topics(action="scores")
        elif action == "stats":
            topics(action="stats")
        elif action == "list":
            topics(action="list")
        else:
            typer.secho(f"Unknown topics action: {action}", fg="red")


def _handle_compression(args: List[str]):
    """Handle /compression command."""
    from episodic.commands.unified_compression import compression
    
    if not args:
        compression()
    else:
        action = args[0]
        if action in ["stats", "queue", "compress", "api-stats", "reset-api"]:
            compression(action=action)
        else:
            typer.secho(f"Unknown compression action: {action}", fg="red")


def _handle_rag(args: List[str]):
    """Handle /rag command."""
    from episodic.commands.rag import rag
    
    if not args:
        rag()
    elif args[0] in ["on", "off"]:
        rag(action=args[0])
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
    from episodic.commands.rag import index
    
    if not args:
        typer.secho("Usage: /index <file_path> or /index --text \"<content>\"", fg="red")
    else:
        if args[0] == "--text" and len(args) > 1:
            # Index raw text
            text = " ".join(args[1:])
            index(content=text)
        else:
            # Index file
            file_path = " ".join(args)
            index(file_path=file_path)


def _handle_docs(args: List[str]):
    """Handle /docs command."""
    from episodic.commands.rag import docs
    
    if not args:
        docs()
    else:
        action = args[0]
        if action == "list":
            docs(action="list")
        elif action == "show" and len(args) > 1:
            docs(action="show", doc_id=args[1])
        elif action == "remove" and len(args) > 1:
            docs(action="remove", doc_id=args[1])
        elif action == "clear":
            source = args[1] if len(args) > 1 else None
            docs(action="clear", source=source)
        else:
            typer.secho(f"Unknown docs action: {action}", fg="red")


def _handle_websearch(args: List[str]):
    """Handle /websearch or /ws command."""
    from episodic.commands.web_search import websearch
    
    if not args:
        typer.secho("Usage: /websearch <query> or /websearch [on|off|config|stats|cache]", fg="red")
    else:
        if args[0] in ["on", "off"]:
            websearch(action=args[0])
        elif args[0] == "config":
            websearch(action="config")
        elif args[0] == "stats":
            websearch(action="stats")
        elif args[0] == "cache" and len(args) > 1 and args[1] == "clear":
            websearch(action="cache_clear")
        else:
            # It's a search query
            query = " ".join(args)
            websearch(query)


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
        help(topic=args[0])
    else:
        help()


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