"""
Main loop and entry point for Episodic CLI.

This module contains the main talk loop and application entry point.
"""

import time
import typer
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from episodic.config import config
from episodic.configuration import (
    DEFAULT_HISTORY_FILE, MAIN_LOOP_SLEEP_INTERVAL,
    get_system_color
)
from episodic.db import initialize_db as init_db
from episodic.conversation import ConversationManager
from episodic.benchmark import display_pending_benchmark, reset_benchmarks
from episodic.cli_command_router import handle_command
from episodic.cli_session import (
    add_to_session_commands, save_to_history,
    execute_script
)
from episodic.cli_display import (
    setup_environment, display_welcome, display_model_info,
    get_prompt
)

# Initialize the Typer app
app = typer.Typer()

# Global variables
chat_history_file = DEFAULT_HISTORY_FILE
conversation_manager = None


def handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input)."""
    global conversation_manager
    
    if not conversation_manager:
        typer.secho("⚠️  Please run /init first to initialize the database.", fg="yellow")
        return
    
    try:
        # Check if we're in muse mode
        if config.get("muse_mode", False):
            # In muse mode, treat all input as web search
            from episodic.commands.web_search import websearch
            # Check if web search is enabled
            if not config.get('web_search_enabled', False):
                typer.secho("⚠️  Web search is not enabled. Use '/websearch on' to enable.", fg="yellow")
                return
            
            # Perform web search with synthesis
            websearch(user_input, synthesize=True)
            return
        
        # Normal chat mode - continue with LLM
        # Get the current model from config
        model = config.get("model", "gpt-3.5-turbo")
        
        # Get the system prompt using the prompt manager
        from episodic.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        system_message = prompt_manager.get_active_prompt_content(config.get)
        
        # Get context depth from config (if available)
        context_depth = config.get("context_depth", 5)
        
        # Call the conversation handler
        from episodic.conversation import handle_chat_message as _handle_chat_message_impl
        assistant_node_id, response_text = _handle_chat_message_impl(
            user_input,
            model,
            system_message,
            context_depth,
            conversation_manager
        )
        
        # Display costs if enabled
        if config.get("show_costs"):
            from episodic.commands import cost
            cost()
        
        # Add blank line after LLM output before next prompt
        typer.echo()
            
    except Exception as e:
        typer.secho(f"Error: {e}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")


def talk_loop() -> None:
    """Main conversation loop."""
    global conversation_manager
    
    # Initialize conversation manager
    conversation_manager = ConversationManager()
    
    # Initialize the database state
    conversation_manager.initialize_conversation()
    
    # Display model info
    display_model_info()
    
    # Create prompt session with history
    session = PromptSession(
        history=FileHistory(chat_history_file),
        auto_suggest=AutoSuggestFromHistory(),
        message=get_prompt,
        vi_mode=config.get("vi_mode", False),  # Enable vi mode if configured
    )
    
    # Main loop
    while True:
        try:
            # Get user input using the enhanced prompt
            user_input = session.prompt()
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Save to history (both internal and file)
            add_to_session_commands(user_input)
            save_to_history(user_input)
            
            # Check if it's a command
            if user_input.startswith('/'):
                should_exit = handle_command(user_input)
                if should_exit:
                    # Finalize any ongoing topics before exit
                    conversation_manager.finalize_current_topic()
                    typer.secho("\nGoodbye! 👋", fg=get_system_color())
                    break
            else:
                # It's a chat message
                handle_chat_message(user_input)
                
            # Small sleep to prevent CPU spinning
            time.sleep(MAIN_LOOP_SLEEP_INTERVAL)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            typer.echo()  # New line after ^C
            # Finalize any ongoing topics before exit
            if conversation_manager:
                conversation_manager.finalize_current_topic()
            typer.secho("\nGoodbye! 👋", fg=get_system_color())
            break
        except EOFError:
            # Handle Ctrl+D
            typer.echo()  # New line
            # Finalize any ongoing topics before exit
            if conversation_manager:
                conversation_manager.finalize_current_topic()
            typer.secho("\nGoodbye! 👋", fg=get_system_color())
            break
        except Exception as e:
            typer.secho(f"Error: {e}", fg="red")
            if config.get("debug"):
                import traceback
                typer.secho(traceback.format_exc(), fg="red")


@app.command()
def main(
    execute: Optional[str] = typer.Option(
        None,
        "--execute", "-e",
        help="Execute a script file and exit"
    ),
    init: bool = typer.Option(
        False,
        "--init",
        help="Initialize the database and exit"
    ),
    erase: bool = typer.Option(
        False,
        "--erase",
        help="Erase existing database when initializing"
    ),
    cost: bool = typer.Option(
        False,
        "--cost",
        help="Show cost summary and exit"
    ),
):
    """
    Episodic CLI - A conversational memory agent.
    
    Start an interactive session or execute commands.
    """
    # Set up environment
    setup_environment()
    
    # Reset benchmarks at start
    reset_benchmarks()
    
    # Handle init flag
    if init:
        init_db(erase=erase)
        typer.secho("✅ Database initialized", fg="green")
        return
    
    # Handle cost flag
    if cost:
        # Need to initialize conversation manager to get costs
        global conversation_manager
        conversation_manager = ConversationManager()
        conversation_manager.initialize_conversation()
        
        from episodic.commands import cost as show_cost
        show_cost()
        return
    
    # Handle execute flag
    if execute:
        # Initialize database if needed
        from episodic.db import database_exists
        if not database_exists():
            init_db()
        
        # Initialize conversation manager (reuse global)
        if not conversation_manager:
            conversation_manager = ConversationManager()
        conversation_manager.initialize_conversation()
        
        # Execute the script
        execute_script(execute)
        
        # Show costs if configured
        if config.get("show_costs_on_exit", True):
            from episodic.commands import cost as show_cost
            show_cost()
        
        # Finalize any ongoing topics
        conversation_manager.finalize_current_topic()
        return
    
    # Normal interactive mode
    # Check if database exists
    from episodic.db import database_exists
    if not database_exists():
        typer.secho("Database not found. Initializing...", fg="yellow")
        init_db()
        typer.secho("✅ Database initialized", fg="green")
        typer.echo()
    
    # Display welcome
    display_welcome()
    
    # Start the main talk loop
    talk_loop()
    
    # Display final benchmark if any
    display_pending_benchmark()
    
    # Show costs on exit if configured
    if config.get("show_costs_on_exit", True) and conversation_manager:
        typer.echo()
        from episodic.commands import cost as show_cost
        show_cost()
    
    # Clean up database connections on exit
    from episodic.db_connection import close_pool
    close_pool()


if __name__ == "__main__":
    try:
        app()
    finally:
        # Ensure cleanup happens even on unexpected exit
        from episodic.db_connection import close_pool
        close_pool()