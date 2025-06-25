"""
Streamlined CLI module for Episodic.

This is the main entry point that coordinates all the command modules.
"""

import typer
import sys
import os
import shlex
import time
from typing import Optional, List
from datetime import datetime

from episodic.config import config
from episodic.configuration import (
    EXIT_COMMANDS, DEFAULT_HISTORY_FILE, MAIN_LOOP_SLEEP_INTERVAL,
    get_prompt_color, get_system_color, get_text_color, get_heading_color
)
from episodic.db import initialize_db as init_db
from episodic.conversation import conversation_manager, handle_chat_message as _handle_chat_message_impl
from episodic.prompt_manager import load_prompt
from episodic.benchmark import display_pending_benchmark, reset_benchmarks
from episodic.compression import start_auto_compression

# Import command modules
from episodic.commands import (
    # Navigation
    init, add, show, print_node, head, ancestry, list_nodes,
    # Settings
    set, verify, cost,
    # Topics
    topics, compress_current_topic,
    # Compression
    compress, compression_stats, compression_queue,
    # Other
    visualize, prompts, summary, benchmark, help,
    handle_model
)

# Import helper functions
from episodic.cli_helpers import _parse_flag_value, _remove_flag_and_value, _has_flag

# Initialize the Typer app
app = typer.Typer()

# Global variables
chat_history_file = DEFAULT_HISTORY_FILE
current_node_id = None


def handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input)."""
    global current_node_id
    
    try:
        # Get the current model from config
        model = config.get("model", "gpt-3.5-turbo")
        
        # Use the conversation manager's system prompt
        system_message = conversation_manager.system_prompt
        
        # Get context depth from config (if available)
        context_depth = config.get("context_depth", 5)
        
        # Use the conversation manager to handle the message
        assistant_node_id, display_response = _handle_chat_message_impl(
            user_input,
            model=model,
            system_message=system_message,
            context_depth=context_depth
        )
        
        # Update the current node
        current_node_id = assistant_node_id
        conversation_manager.set_current_node_id(assistant_node_id)
        
    except Exception as e:
        typer.secho(f"Error querying LLM: {str(e)}", fg="red")
        if config.get("debug", False):
            import traceback
            traceback.print_exc()


def handle_command(command_str: str) -> bool:
    """
    Handle a command string.
    
    Returns:
        bool: True if should exit, False otherwise
    """
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
    
    # Check for exit commands (remove leading slash for comparison)
    cmd_without_slash = cmd[1:] if cmd.startswith('/') else cmd
    if cmd_without_slash in EXIT_COMMANDS or cmd_without_slash == "q":
        return True
    
    try:
        # Navigation commands
        if cmd == "/init":
            erase = _has_flag(args, ["--erase", "-e"])
            init(erase=erase)
        
        elif cmd == "/add":
            if not args:
                typer.secho("Usage: /add <content>", fg="red")
            else:
                content = " ".join(args)
                add(content)
        
        elif cmd == "/show":
            if not args:
                typer.secho("Usage: /show <node_id>", fg="red")
            else:
                show(args[0])
        
        elif cmd == "/print":
            node_id = args[0] if args else None
            print_node(node_id)
        
        elif cmd == "/head":
            node_id = args[0] if args else None
            head(node_id)
        
        elif cmd == "/ancestry":
            if not args:
                typer.secho("Usage: /ancestry <node_id>", fg="red")
            else:
                ancestry(args[0])
        
        elif cmd == "/list":
            count = None
            if args:
                try:
                    count = int(_parse_flag_value(args, ["--count", "-c"]) or args[0])
                except (ValueError, IndexError):
                    pass
            list_nodes(count=count or 10)
        
        # Model command
        elif cmd == "/model":
            model_name = args[0] if args else None
            handle_model(model_name)
        
        # Settings commands
        elif cmd == "/set":
            if len(args) >= 2:
                set(args[0], args[1])
            elif len(args) == 1:
                set(args[0])
            else:
                set()
        
        elif cmd == "/verify":
            verify()
        
        elif cmd == "/cost":
            cost()
        
        # Topic commands
        elif cmd == "/topics":
            limit = 10
            all_topics = False
            verbose = False
            
            if "--all" in args or "-a" in args:
                all_topics = True
            if "--verbose" in args or "-v" in args:
                verbose = True
            
            limit_str = _parse_flag_value(args, ["--limit", "-l"])
            if limit_str:
                try:
                    limit = int(limit_str)
                except ValueError:
                    pass
            
            topics(limit=limit, all=all_topics, verbose=verbose)
        
        elif cmd == "/compress-current-topic":
            compress_current_topic()
        
        # Compression commands
        elif cmd == "/compress":
            strategy = _parse_flag_value(args, ["--strategy", "-s"]) or "simple"
            node_id = _parse_flag_value(args, ["--node", "-n"])
            dry_run = _has_flag(args, ["--dry-run", "-d"])
            compress(strategy=strategy, node_id=node_id, dry_run=dry_run)
        
        elif cmd == "/compression-stats":
            compression_stats()
        
        elif cmd == "/compression-queue":
            compression_queue()
        
        # Other commands
        elif cmd == "/visualize":
            output = _parse_flag_value(args, ["--output", "-o"])
            no_browser = _has_flag(args, ["--no-browser"])
            port = 8080
            port_str = _parse_flag_value(args, ["--port", "-p"])
            if port_str:
                try:
                    port = int(port_str)
                except ValueError:
                    pass
            visualize(output=output, no_browser=no_browser, port=port)
        
        elif cmd == "/prompts":
            if args:
                if len(args) >= 2:
                    prompts(args[0], args[1])
                else:
                    prompts(args[0])
            else:
                prompts()
        
        elif cmd == "/summary":
            count = args[0] if args else None
            summary(count)
        
        elif cmd == "/script":
            if not args:
                typer.secho("Usage: /script <filename>", fg="red")
            else:
                execute_script(args[0])
        
        elif cmd == "/benchmark":
            benchmark()
        
        elif cmd == "/help":
            help()
        
        else:
            typer.secho(f"Unknown command: {cmd}", fg="red")
            typer.secho("Type '/help' for available commands", fg=get_text_color())
    
    except Exception as e:
        typer.secho(f"Error executing command: {str(e)}", fg="red")
        if config.get("debug", False):
            import traceback
            traceback.print_exc()
    
    # Display any pending benchmarks after command execution
    display_pending_benchmark()
    
    return False


def execute_script(filename: str):
    """Execute commands from a script file."""
    import os
    from episodic.benchmark import benchmark_operation, display_pending_benchmark
    
    # Check if file exists
    if not os.path.exists(filename):
        typer.secho(f"Error: Script file '{filename}' not found", fg="red", err=True)
        return
    
    with benchmark_operation(f"Script execution: {filename}"):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            typer.secho(f"\n📜 Executing script: {filename}", fg=get_heading_color())
            typer.secho("─" * 50, fg=get_heading_color())
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Show what we're executing
                typer.secho(f"\n[{i}] ", fg=get_text_color(), dim=True, nl=False)
                typer.secho(f"> {line}", fg=get_system_color())
                
                # Execute the command/message
                try:
                    if line.startswith('/'):
                        handle_command(line)
                    else:
                        handle_chat_message(line)
                    
                    # Display any pending benchmarks after each command
                    display_pending_benchmark()
                    
                except Exception as e:
                    typer.secho(f"Error on line {i}: {str(e)}", fg="red")
                    if typer.confirm("Continue with script?"):
                        continue
                    else:
                        break
            
            typer.secho("\n─" * 50, fg=get_heading_color())
            typer.secho("✅ Script execution completed", fg=get_system_color())
            
        except Exception as e:
            typer.secho(f"Error reading script: {str(e)}", fg="red", err=True)


def save_to_history(message: str):
    """Save a message to the history file."""
    global chat_history_file
    
    if not chat_history_file:
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(chat_history_file), exist_ok=True)
        
        # Append to history file
        with open(chat_history_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        if config.get("debug", False):
            typer.secho(f"Failed to save to history: {e}", fg="red", err=True)


def setup_environment():
    """Set up the environment for the CLI."""
    # Initialize database
    init_db()
    
    # Load active prompt
    active_prompt = config.get("active_prompt", "default")
    prompt_data = load_prompt(active_prompt)
    if prompt_data and 'content' in prompt_data:
        conversation_manager.system_prompt = prompt_data['content']
    
    # Start auto-compression if enabled
    if config.get("auto_compress_topics", True):
        start_auto_compression()
    
    # Reset benchmarks for new session
    reset_benchmarks()


def get_prompt() -> str:
    """Get the appropriate prompt based on color settings."""
    color_mode = config.get("color_mode", "full")
    
    if color_mode == "none":
        return "> "
    else:
        # Get the actual color name, not ANSI string
        from episodic.configuration import get_color_scheme
        color_name = get_color_scheme()["prompt"].lower()
        return typer.style("> ", fg=color_name, bold=True)


def talk_loop() -> None:
    """Main conversation loop."""
    setup_environment()
    
    typer.secho("Welcome to Episodic! Type '/help' for commands or start chatting.", 
               fg=get_system_color())
    
    while True:
        try:
            # Get user input with styled prompt
            user_input = typer.prompt("", prompt_suffix=get_prompt()).strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Save to history
            save_to_history(user_input)
            
            # Check if it's a command
            if user_input.startswith('/'):
                should_exit = handle_command(user_input)
                if should_exit:
                    typer.secho("Goodbye! 👋", fg=get_system_color())
                    break
            else:
                # Handle as chat message
                handle_chat_message(user_input)
            
            # Small delay to prevent CPU spinning
            time.sleep(MAIN_LOOP_SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            typer.secho("\n\nUse '/exit' to quit properly", fg=get_system_color())
            continue
        except EOFError:
            # Handle Ctrl+D
            typer.secho("\nGoodbye! 👋", fg=get_system_color())
            break
        except Exception as e:
            typer.secho(f"Error: {str(e)}", fg="red")
            if config.get("debug", False):
                import traceback
                traceback.print_exc()


@app.command()
def main():
    """Start the Episodic conversation interface."""
    talk_loop()


if __name__ == "__main__":
    app()