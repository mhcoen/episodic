"""
Streamlined CLI module for Episodic.

This is the main entry point that coordinates all the command modules.
"""

import typer
import os
import shlex
import time
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from typing import Optional

from episodic.config import config
from episodic.configuration import (
    EXIT_COMMANDS, DEFAULT_HISTORY_FILE, MAIN_LOOP_SLEEP_INTERVAL,
    get_system_color, get_text_color, get_heading_color, get_llm_color
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
    set, verify, cost, model_params, config_docs,
    # Topics
    topics, compress_current_topic, rename_ongoing_topics,
    # Compression
    compress, compression_stats, compression_queue, api_call_stats, reset_api_stats,
    # Other
    visualize, prompts, summary, benchmark, help,
    handle_model
)
from episodic.commands.debug_topics import topic_scores
from episodic.commands.index_topics import index_topics

# Import helper functions
from episodic.cli_helpers import _parse_flag_value, _has_flag

# Initialize the Typer app
app = typer.Typer()

# Global variables
chat_history_file = DEFAULT_HISTORY_FILE
current_node_id = None
session_commands = []  # Track commands entered in this session


def handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input)."""
    global current_node_id
    
    try:
        # Get the current model from config
        model = config.get("model", "gpt-3.5-turbo")
        
        # Get the system prompt using the prompt manager
        from episodic.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        system_message = prompt_manager.get_active_prompt_content(config.get)
        
        # Get context depth from config (if available)
        context_depth = config.get("context_depth", 5)
        
        # Enhance message with document context if enabled
        try:
            from episodic.commands.documents import doc_commands
            enhanced_input = doc_commands.enhance_message_if_enabled(user_input)
        except ImportError:
            # Document features not available
            enhanced_input = user_input
        
        # Use the conversation manager to handle the message
        assistant_node_id, display_response = _handle_chat_message_impl(
            enhanced_input,
            model=model,
            system_message=system_message,
            context_depth=context_depth
        )
        
        # Update the current node
        current_node_id = assistant_node_id
        conversation_manager.set_current_node_id(assistant_node_id)
        
    except Exception as e:
        typer.secho(f"Error querying LLM: {str(e)}", fg="red")
        if config.get("debug"):
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
        
        elif cmd == "/model-params" or cmd == "/mp":
            param_set = args[0] if args else None
            model_params(param_set)
        
        elif cmd == "/config-docs":
            config_docs()
        
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
        
        elif cmd == "/rename-topics":
            rename_ongoing_topics()
        
        elif cmd == "/topic-scores":
            node_id = args[0] if args else None
            limit = int(_parse_flag_value(args, ["--limit", "-l"]) or 20)
            verbose = _has_flag(args, ["--verbose", "-v"])
            topic_scores(node_id=node_id, limit=limit, verbose=verbose)
        
        elif cmd == "/index":
            window_size = int(args[0]) if args else 5
            apply = _has_flag(args, ["--apply", "-a"])
            verbose = _has_flag(args, ["--verbose", "-v"])
            index_topics(window_size=window_size, apply=apply, verbose=verbose)
        
        # Compression commands
        elif cmd == "/compress":
            strategy = _parse_flag_value(args, ["--strategy", "-s"]) or "simple"
            node_id = _parse_flag_value(args, ["--node", "-n"])
            dry_run = _has_flag(args, ["--dry-run", "-d"])
            compress(strategy=strategy, node_id=node_id, dry_run=dry_run)
        
        elif cmd == "/compression-stats":
            compression_stats()
            
        elif cmd == "/api-stats":
            api_call_stats()
            
        elif cmd == "/reset-api-stats":
            reset_api_stats()
        
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
        
        # Document commands
        elif cmd == "/load":
            if not args:
                typer.secho("Usage: /load <pdf_file>", fg="red")
            else:
                from episodic.commands.documents import handle_load_command
                handle_load_command(' '.join(args))
        
        elif cmd == "/docs":
            from episodic.commands.documents import handle_docs_command
            action = args[0] if args else None
            handle_docs_command(action)
        
        elif cmd == "/search":
            if not args:
                typer.secho("Usage: /search <query>", fg="red")
            else:
                from episodic.commands.documents import handle_search_command
                handle_search_command(' '.join(args))
        
        elif cmd == "/script":
            if not args:
                typer.secho("Usage: /script <filename>", fg="red")
            else:
                execute_script(args[0])
        
        elif cmd == "/save":
            if not args:
                typer.secho("Usage: /save <filename>", fg="red")
            else:
                save_session_script(args[0])
        
        elif cmd == "/benchmark":
            benchmark()
        
        elif cmd == "/help":
            help()
        
        else:
            typer.secho(f"Unknown command: {cmd}", fg="red")
            typer.secho("Type '/help' for available commands", fg=get_text_color())
    
    except Exception as e:
        typer.secho(f"Error executing command: {str(e)}", fg="red")
        if config.get("debug"):
            import traceback
            traceback.print_exc()
    
    # Display any pending benchmarks after command execution
    display_pending_benchmark()
    
    return False


def save_session_script(filename: str):
    """Save the current session's commands to a script file."""
    import os
    
    # Ensure scripts directory exists
    scripts_dir = "scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # Full path to script file
    script_path = os.path.join(scripts_dir, filename)
    
    # Filter out the /save command itself from the session
    commands_to_save = [cmd for cmd in session_commands if not cmd.startswith('/save')]
    
    if not commands_to_save:
        typer.secho("No commands to save in this session.", fg=get_system_color())
        return
    
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            # Write each command on a new line
            for cmd in commands_to_save:
                f.write(cmd + '\n')
        
        typer.secho(f"✅ Saved {len(commands_to_save)} commands to: {script_path}", fg=get_system_color())
        typer.secho(f"   You can reload with: /script {filename}", fg=get_text_color())
    except Exception as e:
        typer.secho(f"Error saving script: {str(e)}", fg="red")


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
                typer.secho(f"\n[{i}]", fg=get_text_color(), bold=True, nl=False)
                typer.secho(f" ", nl=False)
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
            
            # Finalize the current topic before ending
            conversation_manager.finalize_current_topic()
            
            typer.secho("\n" + "─" * 50, fg=get_heading_color())
            typer.secho("✅ Script execution completed", fg=get_system_color())
            
        except Exception as e:
            typer.secho(f"Error reading script: {str(e)}", fg="red", err=True)


def setup_readline():
    """Set up readline for command history with arrow keys."""
    # No longer needed - prompt_toolkit handles this


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
        if config.get("debug"):
            typer.secho(f"Failed to save to history: {e}", fg="red", err=True)


def setup_environment():
    """Set up the environment for the CLI."""
    # Initialize database
    init_db()
    
    # Set up readline for command history
    setup_readline()
    
    # Load active prompt (no longer needed - prompt manager handles this)
    # The prompt manager will get the active prompt when needed
    
    # Start auto-compression if enabled
    if config.get("auto_compress_topics"):
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
    import sys
    
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        typer.secho("Error: Episodic requires an interactive terminal. Use /script to execute scripts.", fg="red")
        sys.exit(1)
    
    setup_environment()
    
    typer.secho("Welcome to Episodic! Type '/help' for commands or start chatting.", 
               fg=get_system_color())
    
    # Display current model and pricing information
    from episodic.llm_config import get_default_model, get_current_provider
    from litellm import cost_per_token
    
    current_model = get_default_model()
    provider = get_current_provider()
    
    # Check if it's a local provider
    LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]
    
    if provider in LOCAL_PROVIDERS:
        typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
        typer.secho("Pricing: Local model", fg=get_system_color())
    else:
        try:
            input_cost, output_cost = cost_per_token(model=current_model, prompt_tokens=1000, completion_tokens=1000)
            typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
            typer.secho(f"Pricing: ${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output", fg=get_system_color())
        except Exception:
            typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
            typer.secho("Pricing: Not available", fg=get_system_color())
    
    typer.echo()  # Blank line for spacing
    
    # Initialize the conversation manager with the current head node
    conversation_manager.initialize_conversation()
    
    # Set up prompt_toolkit session with history
    history_file = os.path.expanduser("~/.episodic_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True
    )
    
    while True:
        try:
            # Get the prompt with color
            color_mode = config.get("color_mode", "full")
            if color_mode == "none":
                prompt_html = "> "
            else:
                # Get color based on color scheme
                from episodic.configuration import get_color_scheme
                color_map = {
                    'green': '#00ff00',
                    'blue': '#0000ff',
                    'cyan': '#00ffff',
                    'magenta': '#ff00ff',
                    'yellow': '#ffff00',
                    'red': '#ff0000',
                    'white': '#ffffff'
                }
                color_name = get_color_scheme()["prompt"].lower()
                hex_color = color_map.get(color_name, '#00ff00')
                prompt_html = HTML(f'<ansigreen><b>&gt; </b></ansigreen>')
            
            # Get user input with full readline support
            user_input = session.prompt(prompt_html).strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Save to history (for our custom history file too)
            save_to_history(user_input)
            
            # Track commands in this session
            session_commands.append(user_input)
            
            # Check if it's a command
            if user_input.startswith('/'):
                should_exit = handle_command(user_input)
                if should_exit:
                    # Finalize current topic before exiting
                    conversation_manager.finalize_current_topic()
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
            conversation_manager.finalize_current_topic()
            typer.secho("\nGoodbye! 👋", fg=get_system_color())
            break
        except Exception as e:
            typer.secho(f"Error: {str(e)}", fg="red")
            if config.get("debug"):
                import traceback
                traceback.print_exc()


@app.command()
def main(
    execute: Optional[str] = typer.Option(None, "--execute", "-e", help="Execute a script file and exit"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specify the model to use"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output")
):
    """Start the Episodic conversation interface."""
    # Handle command-line options
    if model and isinstance(model, str):
        config.set("model", model)
        
        # If model specified, ensure provider matches
        from episodic.llm_config import set_current_provider, save_config, load_config
        if "/" in model:
            # Provider specified in model string (e.g. "ollama/llama3")
            provider = model.split("/")[0]
            set_current_provider(provider)
        elif model.startswith("gpt"):
            # OpenAI model
            set_current_provider("openai")
        elif model.startswith("claude"):
            # Anthropic model
            set_current_provider("anthropic")
        # For other models, keep current provider
    
    if no_stream and isinstance(no_stream, bool):
        config.set("streaming", False)
    
    if execute and isinstance(execute, str):
        # Execute script mode
        import sys
        setup_environment()
        
        # Display model info
        from episodic.llm_config import get_default_model, get_current_provider
        current_model = get_default_model()
        provider = get_current_provider()
        LOCAL_PROVIDERS = ["ollama", "lmstudio", "local"]
        
        typer.secho("Welcome to Episodic! Type '/help' for commands or start chatting.", 
                   fg=get_system_color())
        
        if provider in LOCAL_PROVIDERS:
            typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
            typer.secho("Pricing: Local model", fg=get_system_color())
        else:
            try:
                from litellm import cost_per_token
                input_cost, output_cost = cost_per_token(model=current_model, prompt_tokens=1000, completion_tokens=1000)
                typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
                typer.secho(f"Pricing: ${input_cost:.6f}/1K input, ${output_cost:.6f}/1K output", fg=get_system_color())
            except Exception:
                typer.secho(f"Using model: {current_model} (Provider: {provider})", fg=get_llm_color())
                typer.secho("Pricing: Not available", fg=get_system_color())
        
        typer.echo()  # Blank line
        
        # Initialize conversation
        conversation_manager.initialize_conversation()
        
        # Execute the script
        execute_script(execute)
        
        # Finalize and exit
        conversation_manager.finalize_current_topic()
        sys.exit(0)
    else:
        # Interactive mode
        talk_loop()


if __name__ == "__main__":
    app()