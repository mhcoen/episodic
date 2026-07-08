"""
Main loop and entry point for Episodic CLI.

This module contains the main talk loop and application entry point.
"""

import asyncio
import os
import time
import typer
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from episodic.config import config
from episodic.configuration import (
    MAIN_LOOP_SLEEP_INTERVAL,
    get_system_color
)
from episodic.db import initialize_db as init_db
from episodic.conversation import ConversationManager
from episodic.benchmark import display_pending_benchmark, reset_benchmarks
from episodic.cli_command_router import handle_command
from episodic.cli_session import (
    add_to_session_commands,
    execute_script
)
from episodic.cli_display import (
    setup_environment, display_welcome, display_model_info,
    get_prompt
)

# Initialize the Typer app
app = typer.Typer()

# Global variables
conversation_manager = None


def handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input)."""
    global conversation_manager

    if not conversation_manager:
        typer.secho("⚠️  Please run /init first to initialize the database.", fg="yellow")
        return

    # Check if this is a memory query that should be routed to retrieval
    from episodic.cli_memory_query import handle_memory_query
    if handle_memory_query(user_input):
        return  # Query was handled, don't send to LLM

    try:
        # Normal chat mode - continue with LLM (muse mode is handled within)
        # Get the current model from config
        model = config.get("model", "gpt-3.5-turbo")
        
        # Get the system prompt using the prompt manager
        from episodic.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        system_message = prompt_manager.get_active_prompt_content(config.get)
        
        # Get context depth from config - use muse_context_depth if in muse mode
        if config.get("muse_mode", False):
            context_depth = config.get("muse_context_depth", 5)
        else:
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
        if config.get("show_cost"):
            from episodic.commands import cost
            cost()
        
        # Add blank line after LLM output before next prompt
        typer.echo()
            
    except Exception as e:
        typer.secho(f"Error: {e}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")
        # Resync the in-memory conversation tip to the DB head. A mid-turn
        # failure can leave the DB head advanced (a node was inserted) while
        # manager.current_node_id still points at the previous node; the next
        # message would then insert against a stale parent and fork the DAG.
        try:
            from episodic.db_nodes import get_head
            head = get_head()
            if head and conversation_manager:
                conversation_manager.set_current_node_id(head)
        except Exception:
            pass


async def _async_talk_loop() -> None:
    """Async main conversation loop with native prompt_toolkit support."""
    global conversation_manager

    # Initialize conversation manager
    conversation_manager = ConversationManager()

    # Track Ctrl-C for double-press exit
    last_interrupt_time = 0

    # Update the module-level instance in conversation.py to use the same instance
    import episodic.conversation
    episodic.conversation.conversation_manager = conversation_manager

    # Initialize the database state
    conversation_manager.initialize_conversation()

    # Display model info with spacing
    typer.echo()  # Add blank line for visual separation
    display_model_info()

    # Create prompt session with history file in ~/.episodic directory
    history_file = os.path.expanduser("~/.episodic/command_history")

    # Set up completer if tab completion is enabled
    completer = None
    if config.get("enable_tab_completion", False):
        from episodic.cli_completer import EpisodicCompleter
        completer = EpisodicCompleter()

    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        message=get_prompt,
        vi_mode=config.get("vi_mode", False),  # Enable vi mode if configured
        completer=completer,
    )

    # Voice loop manager (handles voice listener lifecycle and input racing)
    from episodic.cli_voice_loop import VoiceLoopManager, VOICE_OFF_SIGNAL
    voice_mgr = VoiceLoopManager()

    # Main loop
    while True:
        try:
            # Get input (races voice and keyboard when voice mode is on)
            user_input = await voice_mgr.get_input(session)
            if user_input is None:
                continue
            if user_input == VOICE_OFF_SIGNAL:
                continue

            # Skip empty input and strip whitespace
            user_input = user_input.strip()
            if not user_input:
                continue

            # Save to session commands (internal only, no file history)
            add_to_session_commands(user_input)

            # Display user input in a box if enabled
            if config.get("show_input_box", True):
                from episodic.box_utils import draw_input_box, draw_simple_input_box
                import shutil

                # Calculate how many lines the input took up in the terminal
                terminal_width = shutil.get_terminal_size().columns

                # The prompt is either "> " or "» " (both 2 chars)
                prompt_length = 2

                # Calculate how many lines the input occupied
                cursor_position = prompt_length
                lines_used = 1

                # Simple character counting to handle wrapped lines
                for char in user_input:
                    if char == '\n':
                        lines_used += 1
                        cursor_position = 0
                    else:
                        cursor_position += 1
                        if cursor_position >= terminal_width:
                            lines_used += 1
                            cursor_position = 0

                # Move cursor up and clear all the lines the input occupied
                for _ in range(lines_used):
                    print("\033[1A\033[2K", end="")  # Move up 1 line and clear it

                # Use Unicode box if supported, otherwise ASCII
                if config.get("use_unicode_boxes", True):
                    draw_input_box(user_input)
                else:
                    draw_simple_input_box(user_input)

            # Pause idle timer during processing (prevents idle message during LLM streaming)
            voice_mgr.pause_idle_timer()

            try:
                # Check if it's a command
                if user_input.startswith('/'):
                    should_exit = handle_command(user_input)
                    if should_exit:
                        # Finalize any ongoing topics before exit
                        from episodic.db import database_exists
                        if database_exists() and conversation_manager:
                            conversation_manager.finalize_current_topic()
                        voice_mgr.stop_listener()
                        typer.secho("\nGoodbye! 👋", fg=get_system_color())
                        break
                else:
                    # Try voice grammar for utility commands first
                    from episodic.utility.cli_integration import (
                        handle_voice_utterance,
                        display_utility_result,
                    )
                    utility_result = handle_voice_utterance(user_input)
                    if utility_result is not None:
                        display_utility_result(utility_result)
                    else:
                        # Fall through to chat message handler
                        handle_chat_message(user_input)
            finally:
                # Resume idle timer after processing
                voice_mgr.resume_idle_timer()

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(MAIN_LOOP_SLEEP_INTERVAL)

        except KeyboardInterrupt:
            # Handle Ctrl+C
            current_time = time.time()
            if current_time - last_interrupt_time < 1.0:
                typer.echo()
                from episodic.db import database_exists
                if database_exists() and conversation_manager:
                    conversation_manager.finalize_current_topic()
                voice_mgr.stop_listener()
                typer.secho("\nGoodbye! 👋", fg=get_system_color())
                break
            else:
                typer.echo()
                last_interrupt_time = current_time
                continue
        except EOFError:
            typer.echo()
            from episodic.db import database_exists
            if database_exists() and conversation_manager:
                conversation_manager.finalize_current_topic()
            voice_mgr.stop_listener()
            typer.secho("\nGoodbye! 👋", fg=get_system_color())
            break
        except Exception as e:
            typer.secho(f"Error: {e}", fg="red")
            if config.get("debug"):
                import traceback
                typer.secho(traceback.format_exc(), fg="red")


def talk_loop() -> None:
    """Main conversation loop - runs async loop internally."""
    asyncio.run(_async_talk_loop())


@app.command()
def main(
    script: Optional[str] = typer.Option(
        None,
        "--script",
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
        
        # Update the module-level instance in conversation.py to use the same instance
        import episodic.conversation
        episodic.conversation.conversation_manager = conversation_manager
        
        conversation_manager.initialize_conversation()
        
        from episodic.commands import cost as show_cost
        show_cost()
        return
    
    # Handle --script flag
    if script:
        # Initialize database if needed
        from episodic.db import database_exists
        if not database_exists():
            init_db()
        
        # Initialize conversation manager (reuse global)
        if not conversation_manager:
            conversation_manager = ConversationManager()
            
            # Update the module-level instance in conversation.py to use the same instance
            import episodic.conversation
            episodic.conversation.conversation_manager = conversation_manager
            
        conversation_manager.initialize_conversation()
        
        # Execute the script
        execute_script(script)
        
        # Show costs if configured
        if config.get("show_costs_on_exit", True):
            from episodic.commands import cost as show_cost
            show_cost()
        
        # Finalize any ongoing topics (only if database still exists)
        from episodic.db import database_exists
        if database_exists():
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

    # Start the data refresh scheduler for background provider updates
    from episodic.utility.cli_integration import start_data_refresh_scheduler
    start_data_refresh_scheduler()

    # Start the task scheduler eagerly so alarms/timers persisted by a
    # previous session are restored and can actually fire.
    from episodic.utility.cli_integration import get_scheduler
    get_scheduler()

    # Start the main talk loop
    talk_loop()
    
    # Display final benchmark if any
    display_pending_benchmark()
    
    # Show costs on exit if configured
    if config.get("show_costs_on_exit", True) and conversation_manager:
        typer.echo()
        from episodic.commands import cost as show_cost
        show_cost()
    
    # Shutdown utility services (scheduler, adapters)
    from episodic.utility.cli_integration import shutdown_utility_services
    shutdown_utility_services()

    # Clean up database connections on exit
    from episodic.db_connection import close_pool
    close_pool()


if __name__ == "__main__":
    try:
        app()
    finally:
        # Ensure cleanup happens even on unexpected exit
        from episodic.utility.cli_integration import shutdown_utility_services
        shutdown_utility_services()
        from episodic.db_connection import close_pool
        close_pool()
