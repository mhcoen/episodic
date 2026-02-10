"""
Navigation, display, and session command handlers for Episodic CLI.

Split from cli_command_handlers.py to stay under the 500-line limit.
These are thin wrappers that lazily import command implementations.
"""

from typing import List

import typer

from episodic.configuration import (
    get_text_color, get_heading_color, get_system_color,
    get_error_color, DEFAULT_LIST_COUNT,
)


def handle_about():
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
    typer.secho("Repository: https://github.com/mhcoen/episodic", fg=get_system_color())


def handle_welcome():
    """Handle /welcome command."""
    from episodic.cli_display import display_welcome, display_model_info
    display_welcome()
    display_model_info()


def handle_config():
    """Handle /config command - show current configuration."""
    from episodic.commands.settings_display import display_all_settings
    from episodic.config import config

    context_depth = config.get("depth", 10)
    semdepth = config.get("semdepth", 0)

    typer.secho("\n⚙️  Current Configuration", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())
    display_all_settings(context_depth, semdepth)


def handle_history(args: List[str]):
    """Handle /history command - show conversation history."""
    limit = int(args[0]) if args else 20

    from episodic.commands.navigation import list as list_command
    list_command(count=limit)


def handle_tree(args: List[str]):
    """Handle /tree command - show conversation tree structure."""
    if args:
        from episodic.commands.navigation import ancestry
        ancestry(args[0])
    else:
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


def handle_summary(args: List[str]):
    """Handle /summary command."""
    from episodic.commands.summary import summary

    valid_lengths = ["brief", "short", "standard", "detailed", "bulleted"]
    length = None
    count_arg = None

    for arg in args:
        if arg.lower() in valid_lengths:
            length = arg.lower()
        else:
            count_arg = arg

    if not count_arg:
        summary(length=length)
    elif count_arg.lower() == "all":
        summary("all", length=length)
    elif count_arg.lower() == "loaded":
        summary("loaded", length=length)
    else:
        try:
            count = int(count_arg)
            summary(count, length=length)
        except ValueError:
            typer.secho(f"Invalid argument '{count_arg}'. Use a number, 'all', or 'loaded'", fg=get_error_color())
            typer.secho(f"Valid length options: {', '.join(valid_lengths)}", fg=get_error_color())


def handle_list(args: List[str]):
    """Handle /list command."""
    from episodic.commands.navigation import list as list_command

    if not args:
        list_command()
    else:
        try:
            count = int(args[0])
            list_command(count=count)
        except ValueError:
            typer.secho("Usage: /list [count]", fg=get_error_color())


def handle_last(args: List[str]):
    """Handle /last command - show recent conversation nodes."""
    from episodic.commands.navigation import list as list_command

    count = DEFAULT_LIST_COUNT
    if args:
        try:
            count = int(args[0])
        except ValueError:
            typer.secho(f"Invalid count: {args[0]}. Using default.", fg="yellow")

    list_command(count=count)


def handle_out(args: List[str]):
    """Handle /out command."""
    from episodic.commands.save import save_command

    args_str = " ".join(args) if args else ""
    save_command(args_str)


def handle_in(args: List[str]):
    """Handle /in command."""
    from episodic.commands.resume import resume_command

    if not args:
        typer.secho("Usage: /in <filename.md>", fg=get_error_color())
    else:
        filepath = " ".join(args)
        resume_command(filepath)


def handle_ls(args: List[str]):
    """Handle /ls or /files command."""
    from episodic.commands.ls import ls_command

    directory = " ".join(args) if args else None
    ls_command(directory)


def handle_scripts(args: List[str]):
    """Handle /scripts command."""
    from episodic.commands.scripts import scripts_command

    if not args:
        scripts_command()
    else:
        subcommand = args[0]
        remaining_args = args[1:]
        if subcommand in ["save", "run", "list"]:
            if subcommand == "save" and remaining_args:
                scripts_command(subcommand, " ".join(remaining_args))
            elif subcommand == "run" and remaining_args:
                scripts_command(subcommand, " ".join(remaining_args))
            else:
                scripts_command(subcommand)
        else:
            scripts_command("run", subcommand)


def handle_pause():
    """Handle /pause command - wait for user to press Enter."""
    input("Press Enter to continue...")


def handle_save_new(args: List[str]):
    """Handle new /save command (for conversations)."""
    from episodic.commands.save_load import save_command

    filename = " ".join(args) if args else None
    save_command(filename)


def handle_load(args: List[str]):
    """Handle /load command."""
    from episodic.commands.save_load import load_command

    if not args:
        typer.secho("Usage: /load <filename>", fg=get_error_color())
    else:
        filename = " ".join(args)
        load_command(filename)


def handle_simple():
    """Handle /simple command."""
    from episodic.commands.interface_mode import simple_mode_command
    simple_mode_command()


def handle_advanced():
    """Handle /advanced command."""
    from episodic.commands.interface_mode import advanced_mode_command
    advanced_mode_command()


def handle_new(args: List[str]):
    """Handle /new command."""
    from episodic.commands.new_topic import new_command

    if args:
        topic_name = " ".join(args)
        new_command(topic_name)
    else:
        new_command()


def handle_clear():
    """Handle /clear command."""
    from episodic.commands.new_topic import clear_command
    clear_command()


def handle_debug(args: List[str]):
    """Handle /debug command."""
    from episodic.commands.debug import debug_on, debug_off, debug_only, debug_status, debug_toggle

    if not args:
        debug_status()
    else:
        subcommand = args[0]
        sub_args = args[1:] if len(args) > 1 else []

        if subcommand == "on":
            debug_on(sub_args if sub_args else None)
        elif subcommand == "off":
            debug_off(sub_args if sub_args else None)
        elif subcommand == "only":
            if sub_args:
                debug_only(sub_args)
            else:
                typer.secho("Usage: /debug only <categories...>", fg=get_error_color())
        elif subcommand == "status":
            debug_status()
        elif subcommand == "toggle":
            if sub_args:
                debug_toggle(sub_args[0])
            else:
                typer.secho("Usage: /debug toggle <category>", fg=get_error_color())
        else:
            typer.secho(f"Unknown debug subcommand: {subcommand}", fg=get_error_color())
            typer.secho("Available: on, off, only, status, toggle", fg=get_text_color())


def handle_dev(args: List[str]):
    """Handle /dev command."""
    from episodic.commands.dev import dev
    if args:
        dev(args[0], *args[1:])
    else:
        dev()


def handle_memory(args: List[str]):
    """Handle /memory command."""
    from episodic.commands.memory import memory_command
    if args:
        memory_command(args[0], *args[1:])
    else:
        memory_command()


def handle_forget(args: List[str]):
    """Handle /forget command."""
    from episodic.commands.memory import forget_command
    if args:
        forget_command(args[0], *args[1:])
    else:
        forget_command()


def handle_memory_stats():
    """Handle /memory-stats command."""
    from episodic.commands.memory import memory_stats_command
    memory_stats_command()


def handle_migrate(args: List[str]):
    """Handle /migrate command."""
    from episodic.commands.migrate import migrate_command
    migrate_command(*args)


def handle_critique(args: List[str]):
    """Handle /critique command."""
    from episodic.commands.critique import critique_command
    target = args[0] if args else None
    critique_command(target)


def handle_copy(args: List[str]):
    """Handle /copy command."""
    from episodic.commands.clipboard import copy_command
    node_id = args[0] if args else None
    copy_command(node_id)


def handle_recall(args: List[str]):
    """Handle /recall command - search conversation history."""
    from episodic.commands.recall import recall_command
    recall_command(args)


def handle_test(args: List[str]):
    """Handle /test command - manage test mode and fixtures."""
    from episodic.commands.test_mode import test_command
    subcommand = args[0] if args else None
    test_command(subcommand)


def handle_doctor(args: List[str]):
    """Handle /doctor command - installation health check."""
    from episodic.commands.doctor import doctor
    verbose = args[0] if args else None
    doctor(verbose)


def handle_mcp(args: List[str]):
    """Handle /mcp command - manage MCP server."""
    from episodic.commands.mcp_cmd import mcp_command
    if not args:
        mcp_command()
    else:
        mcp_command(args[0], *args[1:])


def handle_kg(args: List[str]):
    """Handle /kg command - knowledge graph visualization and queries."""
    from episodic.commands.kg import kg_command
    if not args:
        kg_command()
    else:
        kg_command(args[0], *args[1:])


def handle_rollback(args: List[str]):
    """Handle /rollback command."""
    from episodic.commands.rollback import rollback_command
    rollback_command(args[0] if args else None)


def handle_files():
    """Handle /files command."""
    from episodic.commands.save_load import files_command
    files_command()
