"""
Command handler functions for Episodic CLI.

Each handler is a thin wrapper that lazily imports the actual command
implementation and delegates to it.  Extracted from cli_command_router.py
to keep that module under the 500-line limit.

Navigation/display/session handlers live in cli_command_handlers_nav.py.
"""

import shlex
from typing import List

import typer

from episodic.config import config
from episodic.configuration import (
    get_text_color, get_error_color, get_warning_color, get_success_color,
)
from episodic.cli_helpers import _has_flag
from episodic.constants import TOPIC_ACTIONS, COMPRESSION_ACTIONS

# Import nav/display/session handlers so COMMAND_MAP can reference them
from episodic.cli_command_handlers_nav import (
    handle_about, handle_welcome, handle_config,
    handle_history, handle_tree, handle_summary, handle_list, handle_last,
    handle_out, handle_in, handle_ls, handle_scripts, handle_pause,
    handle_save_new, handle_load, handle_simple, handle_advanced,
    handle_new, handle_clear, handle_debug, handle_dev, handle_memory,
    handle_forget, handle_memory_stats, handle_migrate, handle_critique,
    handle_copy, handle_recall, handle_test, handle_doctor, handle_mcp,
    handle_kg, handle_rollback, handle_files,
)


# ---------------------------------------------------------------------------
# Core command handlers (init through help)
# ---------------------------------------------------------------------------

def handle_init(args: List[str]):
    """Handle /init command."""
    from episodic.commands import init
    erase = _has_flag(args, ["--erase", "-e"])
    init(erase=erase)


def handle_add(args: List[str]):
    """Handle /add command."""
    if not args:
        typer.secho("Usage: /add <content>", fg=get_error_color())
    else:
        from episodic.commands import add
        content = " ".join(args)
        add(content)


def handle_show(args: List[str]):
    """Handle /show command."""
    if not args:
        typer.secho("Usage: /show <node_id>", fg=get_error_color())
    else:
        from episodic.commands import show
        show(args[0])


def handle_print(args: List[str]):
    """Handle /print command."""
    from episodic.commands import print_node

    if args:
        print_node(args[0])
    else:
        print_node()


def handle_cost():
    """Handle /cost command."""
    from episodic.commands import cost
    cost()


def handle_model(args: List[str]):
    """Handle /model command."""
    try:
        from episodic.commands.unified_model import model_command

        if not args:
            model_command(None, None)
        elif len(args) == 1:
            model_command(args[0], None)
        else:
            context = args[0]
            model_name = " ".join(args[1:])
            model_command(context, model_name)
    except ImportError:
        from episodic.commands import handle_model as _handle_model_legacy

        if not args:
            _handle_model_legacy()
        else:
            model_name = " ".join(args)
            _handle_model_legacy(model_name)


def handle_mset(args: List[str]):
    """Handle /mset command."""
    from episodic.commands.mset import mset_command

    if not args:
        mset_command(None, None)
    elif len(args) == 1:
        mset_command(args[0], None)
    elif args[0] == "embedding" and args[1] == "list" and len(args) == 2:
        from episodic.commands.mset import list_embedding_models
        list_embedding_models()
    elif len(args) == 2:
        mset_command(args[0], args[1])
    else:
        mset_command(args[0], " ".join(args[1:]))


def handle_style(args: List[str]):
    """Handle /style command."""
    from episodic.commands.style import handle_style
    handle_style(args)


def handle_format(args: List[str]):
    """Handle /format command."""
    from episodic.commands.style import handle_format
    handle_format(args)


def handle_detail(args: List[str]):
    """Handle /detail command."""
    from episodic.commands.detail import detail
    detail(args[0] if args else None)


def handle_theme(args: List[str]):
    """Handle /theme command."""
    from episodic.commands.theme import theme_command
    if args:
        theme_command(args[0])
    else:
        theme_command()


def handle_reflect(args: List[str]):
    """Handle /reflect command."""
    from episodic.commands.reflection import reflection_command
    if args:
        steps = 3
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--steps" and i + 1 < len(args):
                try:
                    steps = int(args[i + 1])
                    i += 2
                    continue
                except ValueError:
                    typer.secho("Error: --steps must be followed by a number", fg=get_error_color())
                    return
            query_parts.append(args[i])
            i += 1

        query = " ".join(query_parts) if query_parts else None
        reflection_command(query=query, steps=steps)
    else:
        reflection_command()


def handle_set(args: List[str]):
    """Handle /set command."""
    from episodic.commands import set

    if not args:
        set(None, None)
    elif len(args) == 1:
        set(args[0], None)
    else:
        param = args[0]
        value = " ".join(args[1:])
        set(param, value)


def handle_reset():
    """Handle /reset command."""
    from episodic.commands import reset
    reset()


def handle_topics(args: List[str]):
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
                typer.secho("Usage: /topics index <number>", fg=get_error_color())
        elif action == "scores":
            handle_topics_action(action="scores")
        elif action == "stats":
            handle_topics_action(action="stats")
        elif action == "list":
            handle_topics_action(action="list")
        elif action == "reanalyze":
            apply = "apply" in action_args
            verbose = "verbose" in action_args
            min_similarity = None
            for arg in action_args:
                if arg not in ("apply", "verbose"):
                    try:
                        min_similarity = float(arg)
                    except ValueError:
                        pass
            handle_topics_action(action="reanalyze", apply=apply, verbose=verbose, min_similarity=min_similarity)
        elif action == "delete":
            args_str = shlex.join(action_args)
            handle_topics_action(action="delete", args_str=args_str)
        else:
            typer.secho(f"Unknown topics action: {action}", fg=get_error_color())
            typer.secho(f"Available actions: {', '.join(TOPIC_ACTIONS)}", fg=get_warning_color())


def handle_compression(args: List[str]):
    """Handle /compression command."""
    from episodic.commands.unified_compression import compression_command

    if not args:
        compression_command()
    else:
        action = args[0]
        if action in COMPRESSION_ACTIONS:
            compression_command(action=action)
        else:
            typer.secho(f"Unknown compression action: {action}", fg=get_error_color())
            typer.secho(f"Available actions: {', '.join(COMPRESSION_ACTIONS)}", fg=get_warning_color())


def handle_rag(args: List[str]):
    """Handle /rag command."""
    from episodic.commands.rag import rag_toggle, rag_stats

    if not args:
        rag_stats()
    elif args[0] == "on":
        rag_toggle(enable=True)
    elif args[0] == "off":
        rag_toggle(enable=False)
    else:
        typer.secho(f"Unknown rag action: {args[0]}", fg=get_error_color())


def handle_search(args: List[str]):
    """Handle /search or /s command."""
    if not args:
        typer.secho("Usage: /search <query>", fg=get_error_color())
    else:
        from episodic.commands.rag import search
        query = " ".join(args)
        search(query)


def handle_index(args: List[str]):
    """Handle /index or /i command."""
    from episodic.commands.rag import index_text, index_file

    if not args:
        typer.secho('Usage: /index <file_path> or /index --text "<content>"', fg=get_error_color())
    else:
        if args[0] == "--text" and len(args) > 1:
            text = " ".join(args[1:])
            index_text(content=text)
        else:
            file_path = " ".join(args)
            index_file(filepath=file_path)


def handle_docs(args: List[str]):
    """Handle /docs command."""
    from episodic.commands.rag import docs_command
    docs_command(*args)


def handle_web(args: List[str]):
    """Handle /web command."""
    from episodic.commands.web_provider import web_command

    if not args:
        web_command()
    else:
        subcommand = args[0]
        remaining_args = args[1:] if len(args) > 1 else []
        if remaining_args:
            web_command(subcommand, ' '.join(remaining_args))
        else:
            web_command(subcommand, None)


def handle_muse(args: List[str]):
    """Handle /muse command."""
    from episodic.commands.muse import muse

    if not args:
        muse()
    elif args[0] in ["on", "off"]:
        muse(action=args[0])
    else:
        typer.secho(f"Unknown muse action: {args[0]}", fg=get_error_color())


def handle_chat(args: List[str]):
    """Handle /chat command."""
    from episodic.commands.mode import chat_command
    chat_command()


def handle_voice(args: List[str]):
    """Handle /voice command."""
    from episodic.commands.voice import voice

    if not args:
        voice()
    elif len(args) == 1:
        voice(action=args[0])
    else:
        voice(action=args[0], arg=args[1])


def handle_mode(args: List[str]):
    """Handle /mode command."""
    from episodic.commands.mode import mode_command

    if not args:
        mode_command()
    else:
        mode_command(args[0])


def handle_prompt(args: List[str]):
    """Handle /prompt command."""
    from episodic.commands.prompts import prompts

    if not args:
        prompts()
    else:
        action = args[0]
        if action == "list":
            prompts(action="list")
        elif action == "use" and len(args) > 1:
            prompts(action="use", name=args[1])
        elif action == "show" and len(args) > 1:
            prompts(action="show", name=args[1])
        else:
            typer.secho(f"Unknown prompt action: {action}", fg=get_error_color())


def handle_benchmark(args: List[str]):
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
            typer.secho(f"Unknown benchmark action: {action}", fg=get_error_color())


def handle_reset_benchmarks():
    """Handle /reset-benchmarks command."""
    from episodic.benchmark import reset_benchmarks
    reset_benchmarks()
    typer.secho("✅ Benchmark data reset", fg=get_success_color())


def handle_config_docs():
    """Handle /config-docs command."""
    from episodic.commands import config_docs
    config_docs()


def handle_verify():
    """Handle /verify command."""
    from episodic.commands import verify
    verify()


def handle_help(args: List[str]):
    """Handle /help command."""
    from episodic.commands import help

    if args:
        query = " ".join(args)
        help(query=query)
    else:
        help()


def handle_deprecated_commands(cmd: str, args: List[str]):
    """Handle deprecated commands with warnings."""
    deprecated_commands = {
        "/rename-topics": ("topics", "rename"),
        "/compress-current-topic": ("topics", "compress"),
        "/api-stats": ("compression", "api-stats"),
        "/reset-api-stats": ("compression", "reset-api"),
        "/count-tokens": ("cost", None),
        "/model-params": ("mset", None),
        "/mp": ("mset", None),
    }

    if cmd in deprecated_commands:
        new_cmd, action = deprecated_commands[cmd]
        if action:
            typer.secho(f"⚠️  '{cmd}' is deprecated. Use '/{new_cmd} {action}' instead.",
                       fg="yellow")
            if new_cmd == "topics":
                handle_topics([action] if action else [])
            elif new_cmd == "compression":
                handle_compression([action] if action else [])
        else:
            typer.secho(f"⚠️  '{cmd}' is deprecated. Use '/{new_cmd}' instead.",
                       fg="yellow")
            if new_cmd == "cost":
                handle_cost()
            elif new_cmd == "mset":
                handle_mset(args)
    else:
        typer.secho(f"Unknown command: {cmd}. Type /help for available commands.",
                   fg=get_text_color())


# ---------------------------------------------------------------------------
# Command dispatch map
# ---------------------------------------------------------------------------

def _no_args(fn):
    """Wrap a no-args handler so the dispatch loop can call it uniformly."""
    def wrapper(args):
        fn()
    return wrapper


COMMAND_MAP = {
    "/init": handle_init,
    "/add": handle_add,
    "/show": handle_show,
    "/print": handle_print,
    "/cost": _no_args(handle_cost),
    "/model": handle_model,
    "/mset": handle_mset,
    "/style": handle_style,
    "/format": handle_format,
    "/detail": handle_detail,
    "/theme": handle_theme,
    "/reflect": handle_reflect,
    "/set": handle_set,
    "/reset": _no_args(handle_reset),
    "/topics": handle_topics,
    "/compression": handle_compression,
    "/rag": handle_rag,
    "/search": handle_search,
    "/s": handle_search,
    "/index": handle_index,
    "/i": handle_index,
    "/docs": handle_docs,
    "/web": handle_web,
    "/muse": handle_muse,
    "/chat": handle_chat,
    "/voice": handle_voice,
    "/mode": handle_mode,
    "/prompt": handle_prompt,
    "/script": handle_scripts,
    "/scripts": handle_scripts,
    "/pause": _no_args(handle_pause),
    "/save": handle_save_new,
    "/load": handle_load,
    "/new": handle_new,
    "/clear": _no_args(handle_clear),
    "/simple": _no_args(handle_simple),
    "/advanced": _no_args(handle_advanced),
    "/out": handle_out,
    "/in": handle_in,
    "/ls": handle_ls,
    "/files": _no_args(handle_files),
    "/benchmark": handle_benchmark,
    "/reset-benchmarks": _no_args(handle_reset_benchmarks),
    "/config-docs": _no_args(handle_config_docs),
    "/verify": _no_args(handle_verify),
    "/help": handle_help,
    "/h": handle_help,
    "/about": _no_args(handle_about),
    "/welcome": _no_args(handle_welcome),
    "/config": _no_args(handle_config),
    "/history": handle_history,
    "/tree": handle_tree,
    "/summary": handle_summary,
    "/list": handle_list,
    "/last": handle_last,
    "/debug": handle_debug,
    "/dev": handle_dev,
    "/memory": handle_memory,
    "/forget": handle_forget,
    "/memory-stats": _no_args(handle_memory_stats),
    "/migrate": handle_migrate,
    "/critique": handle_critique,
    "/copy": handle_copy,
    "/recall": handle_recall,
    "/test": handle_test,
    "/doctor": handle_doctor,
    "/mcp": handle_mcp,
    "/kg": handle_kg,
    "/rollback": handle_rollback,
}
