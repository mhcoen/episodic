"""
Main loop and entry point for Episodic CLI.

This module contains the main talk loop and application entry point.
"""

import asyncio
import os
import re
import time
import typer
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from episodic.config import config

# Resume cue patterns for detecting topic-resume intent vs recall intent
RESUME_CUE_PATTERNS = [
    r"\bback to\b",
    r"\bcontinuing\b",
    r"\breturning to\b",
    r"\bthat\s+.{1,30}\s+thing\b",  # "that Python thing"
    r"\bas we were\b",
    r"\banyway\b",
    r"\bwhere were we\b",
    r"\blet'?s get back\b",
    r"\bresume\b",
]


def _has_resume_cues(text: str) -> bool:
    """Check if text contains resume cues suggesting topic continuation, not recall."""
    text_lower = text.lower()
    # Check explicit patterns
    for pattern in RESUME_CUE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    # Anaphoric reference + forward question (e.g., "that thing - should I...?")
    if "that" in text_lower and "?" in text:
        return True
    return False
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


def _is_memory_query(ast):
    """
    Determine if an AST represents a memory query that should be routed to retrieval.

    Returns:
        True: Definitely a memory query (route to retrieval)
        False: Definitely not a memory query (route to LLM)
        "needs_classifier": Ambiguous, needs LLM classifier to decide

    Memory queries are:
    1. DiscussionQuery - Always (explicit "when we discussed X" patterns)
    2. MQLCommand with explicit mode keyword (browse, summarize, answer, etc.)
    3. MQLCommand with explicit segment ("in topic: X")
    4. MQLCommand with speaker or temporal constraints
    5. MQLCommand with deictic reference ("earlier", "last time")

    Ambiguous (needs classifier):
    - MQLCommand with just target text and no explicit markers
      e.g., "anything about coffee in our past chats" - CFG parses it
      but the freetext may contain memory intent phrases
    """
    from episodic.query.types import DiscussionQuery, MQLCommand

    # DiscussionQuery always routes to memory
    if isinstance(ast, DiscussionQuery):
        return True

    if isinstance(ast, MQLCommand):
        # Check audit.rule_path for explicit mode
        has_explicit_mode = any(
            "explicit_mode:" in rule for rule in ast.audit.rule_path
        )

        # Check for explicit segment
        has_explicit_segment = ast.segment.explicit

        # Check for speaker constraint
        has_speaker = ast.speaker is not None

        # Check for temporal constraint
        has_temporal = ast.temporal is not None

        # Check for deictic reference
        has_deictic = ast.deictic is not None

        # If any memory-related marker is present, it's a memory query
        if has_explicit_mode or has_explicit_segment or has_speaker or has_temporal or has_deictic:
            return True

        # MQLCommand with no explicit markers is ambiguous - needs classifier
        # The freetext may contain memory intent phrases like "in our past chats"
        return "needs_classifier"

    return False


def _handle_memory_query(user_input: str) -> bool:
    """
    Check if input is a memory query (DiscussionQuery or explicit MQLCommand).

    If so, route to retrieval system instead of LLM.
    Returns True if handled, False to continue to LLM.
    """
    import sqlite3
    from datetime import datetime, timezone

    from episodic.query import parse_to_ast, parse_query
    from episodic.query.types import DiscussionQuery, MQLCommand, FreeText
    from episodic.db_connection import get_connection, get_db_path
    from episodic.retrieval import retrieve, migrate_fts5
    from episodic.retrieval.display import get_exchange_for_display
    from episodic.retrieval.modes import format_answer_response
    from episodic.configuration import get_text_color, get_system_color, get_heading_color

    # Parse to AST first to check type
    ast = parse_to_ast(user_input)

    # Debug output
    if config.get("debug"):
        ast_type = type(ast).__name__
        typer.secho(f"[DEBUG] [MQL] Input: {user_input[:60]}...", fg="cyan", dim=True)
        typer.secho(f"[DEBUG] [MQL] AST type: {ast_type}", fg="cyan", dim=True)
        if hasattr(ast, 'parse_error') and ast.parse_error:
            typer.secho(f"[DEBUG] [MQL] Parse error: {ast.parse_error}", fg="cyan", dim=True)

    # NEW RECALL SYSTEM: Use for DiscussionQuery when enabled
    if isinstance(ast, DiscussionQuery) and config.get("enable_recall_system", True):
        if config.get("debug"):
            typer.secho(f"[DEBUG] [MQL] DiscussionQuery → routing to recall system", fg="cyan", dim=True)
        
        now_utc = datetime.now(timezone.utc)
        user_tz = config.get("timezone", "America/Chicago")
        
        with get_connection() as conn:
            from episodic.recall import handle_recall_query
            handled, context = handle_recall_query(user_input, conn, now_utc, user_tz)
            if handled:
                # Store meta-query in database for audit trail
                try:
                    from episodic.db_nodes import insert_node
                    from episodic.db import get_head
                    current_head = get_head()
                    parent_id = current_head if current_head else None
                    insert_node(
                        content=user_input,
                        parent_id=parent_id,
                        role="user",
                        is_meta_query=True
                    )
                    if config.get("debug"):
                        typer.secho("[DEBUG] [RECALL] Stored meta-query with is_meta_query=True", fg="cyan", dim=True)
                except Exception as e:
                    if config.get("debug"):
                        typer.secho(f"[DEBUG] [RECALL] Failed to store meta-query: {e}", fg="yellow", dim=True)
                return True
        # Fall through to legacy system if recall didn't handle it

    # Check if this should route to memory
    memory_result = _is_memory_query(ast)
    classification = None  # Store classifier result for later use

    # FreeText or ambiguous MQLCommand: use LLM classifier
    if isinstance(ast, FreeText) or memory_result == "needs_classifier":
        # Skip classifier if explicitly disabled
        if not config.get("enable_memory_classifier", True):
            if config.get("debug"):
                reason = "FreeText" if isinstance(ast, FreeText) else "ambiguous MQLCommand"
                typer.secho(f"[DEBUG] [MQL] {reason} + classifier disabled → routing to LLM", fg="cyan", dim=True)
            return False

        # Use single LLM call to classify AND extract intent
        from episodic.query.classifier import classify_and_extract_intent
        if config.get("debug"):
            reason = "FreeText" if isinstance(ast, FreeText) else "ambiguous MQLCommand (no explicit markers)"
            typer.secho(f"[DEBUG] [MQL] {reason} → invoking LLM classifier...", fg="cyan", dim=True)

        classification = classify_and_extract_intent(user_input)

        if config.get("debug"):
            typer.secho(f"[DEBUG] [MQL] Classifier result: is_memory={classification.is_memory_query}, "
                       f"confidence={classification.confidence}", fg="cyan", dim=True)
            if classification.is_memory_query:
                typer.secho(f"[DEBUG] [MQL] Intent: target={classification.target}, mode={classification.mode}, "
                           f"temporal={classification.temporal_hint}, speaker={classification.speaker_hint}", fg="cyan", dim=True)

        if not classification.is_memory_query:
            if config.get("debug"):
                typer.secho("[DEBUG] [MQL] Classifier: GENERAL → routing to LLM", fg="cyan", dim=True)
            return False

        # Classifier identified as memory query - route to NEW recall system
        if config.get("enable_recall_system", True):
            # Check for resume cues FIRST - if present AND reactivation enabled, fall through to chat
            # Resume cues (e.g., "back to that X thing") indicate conversation continuation, not recall
            if _has_resume_cues(user_input) and config.get("enable_topic_reactivation", False):
                if config.get("debug"):
                    typer.secho("[DEBUG] [MQL] Resume cues detected + reactivation enabled → falling through to chat", fg="cyan", dim=True)
                return False  # Fall through to chat flow where reactivation will run

            if config.get("debug"):
                typer.secho("[DEBUG] [MQL] Classifier: MEMORY → routing to recall system", fg="cyan", dim=True)

            now_utc = datetime.now(timezone.utc)
            user_tz = config.get("timezone", "America/Chicago")

            # Build ResolvedQuery from classifier result
            from episodic.query.types import ResolvedQuery
            resolved = ResolvedQuery(
                mode=classification.mode or 'answer',
                target=classification.target,
                segment_explicit=False,
                segment_query=None,
                segment_resolved_ids=None,
                segment_ambiguous=False,
                segment_candidates=None,
                temporal=None,  # TODO: resolve temporal_hint if present
                speaker=classification.speaker_hint if classification.speaker_hint != 'both' else None,
                deictic=None,
                has_broadness_cue=False,
                audit_trace='{"source": "classifier"}'
            )

            with get_connection() as conn:
                from episodic.recall import recall
                from episodic.recall.cli_integration import _display_recall_results, _display_no_results

                result = recall(conn=conn, query=resolved, query_form=None)

                if result.is_empty():
                    _display_no_results(classification.target)
                else:
                    _display_recall_results(result)

                # Store meta-query
                try:
                    from episodic.db_nodes import insert_node
                    from episodic.db import get_head
                    current_head = get_head()
                    parent_id = current_head if current_head else None
                    insert_node(
                        content=user_input,
                        parent_id=parent_id,
                        role="user",
                        is_meta_query=True
                    )
                except Exception:
                    pass

                return True

        if config.get("debug"):
            typer.secho("[DEBUG] [MQL] Classifier: MEMORY → routing to retrieval (legacy)", fg="cyan", dim=True)

    # Definite non-memory query
    elif not memory_result:
        if config.get("debug"):
            typer.secho("[DEBUG] [MQL] Not a memory query → routing to LLM", fg="cyan", dim=True)
        return False

    if config.get("debug"):
        typer.secho("[DEBUG] [MQL] Memory query detected → routing to retrieval (legacy)", fg="cyan", dim=True)

    # Get current time and timezone
    now_utc = datetime.now(timezone.utc)
    user_tz = config.get("timezone", "America/Chicago")

    # Run FTS migration (idempotent)
    db_path = get_db_path()
    migration_conn = sqlite3.connect(db_path, isolation_level=None)
    migration_conn.row_factory = sqlite3.Row
    try:
        migrate_fts5(migration_conn)
    except Exception as e:
        if config.get("debug"):
            typer.secho(f"FTS5 migration notice: {e}", fg="yellow", dim=True)
    finally:
        migration_conn.close()

    # Get main connection and resolve query
    with get_connection() as conn:
        if classification is not None and (isinstance(ast, FreeText) or memory_result == "needs_classifier"):
            # FreeText or ambiguous MQLCommand + classified as MEMORY: use intent from classifier
            # Re-parsing with parse_query() would lose the classifier's extracted target
            from episodic.query import resolve_temporal
            from episodic.query.types import ResolvedQuery, TemporalSpec

            if config.get("debug"):
                src = "FreeText" if isinstance(ast, FreeText) else "ambiguous MQLCommand"
                typer.secho(f"[DEBUG] [MQL] {src} → using classification intent", fg="cyan", dim=True)

            # Resolve temporal hint if present
            temporal = None
            if classification.temporal_hint:
                hint = classification.temporal_hint.lower().replace(' ', '_')
                # Map common hints to TemporalSpec kinds
                hint_map = {
                    'yesterday': 'yesterday',
                    'today': 'today',
                    'last_week': 'last_week',
                    'this_week': 'this_week',
                    'last_month': 'last_month',
                    'this_month': 'this_month',
                }
                if hint in hint_map:
                    spec = TemporalSpec(kind=hint_map[hint], raw=classification.temporal_hint)
                    temporal = resolve_temporal(spec, now_utc, user_tz)

            # Build ResolvedQuery from classification result
            resolved = ResolvedQuery(
                mode=classification.mode or 'answer',
                target=classification.target or user_input,
                segment_explicit=False,
                segment_query=None,
                segment_resolved_ids=None,
                segment_ambiguous=False,
                segment_candidates=None,
                temporal=temporal,
                speaker=classification.speaker_hint if classification.speaker_hint != 'both' else None,
                deictic=None,
                has_broadness_cue=False,
                audit_trace='{"source": "classify_and_extract_intent"}'
            )
        else:
            # MQLCommand or DiscussionQuery: use normal parse_query pipeline
            resolved = parse_query(user_input, conn=conn, now_utc=now_utc, user_tz=user_tz)

        # Debug output if enabled
        if config.get("debug"):
            typer.secho(f"\n[Query Understanding]", fg="cyan", dim=True)
            typer.secho(f"  AST type: {type(ast).__name__}", fg="cyan", dim=True)
            typer.secho(f"  Mode: {resolved.mode}", fg="cyan", dim=True)
            typer.secho(f"  Target: {resolved.target}", fg="cyan", dim=True)
            if resolved.temporal:
                typer.secho(f"  Temporal: {resolved.temporal}", fg="cyan", dim=True)
            if resolved.speaker:
                typer.secho(f"  Speaker: {resolved.speaker}", fg="cyan", dim=True)
            if resolved.segment_explicit:
                typer.secho(f"  Segment: {resolved.segment_query}", fg="cyan", dim=True)

        # Build segment scope
        segment_scope = None
        if resolved.segment_resolved_ids:
            segment_scope = resolved.segment_resolved_ids

        # Build temporal tuple
        temporal_tuple = None
        if resolved.temporal:
            temporal_tuple = (
                resolved.temporal[0].isoformat(),
                resolved.temporal[1].isoformat()
            )

        # Stub Chroma (lexical only for now)
        class NoChroma:
            def query(self, **kw):
                return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        retrieval_config = {
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
            "over_fetch_multiplier": 3,
            "segment_filter_in_clause_max": 100,
            "sqlite_max_variable_number": 999,
        }

        results = retrieve(
            conn=conn,
            chroma=NoChroma(),
            target=resolved.target or "",
            segment_scope=segment_scope,
            temporal=temporal_tuple,
            speaker=resolved.speaker,
            mode=resolved.mode,
            max_results=5,
            config=retrieval_config
        )

        # Display results
        if not results:
            msg = format_answer_response(results)
            typer.secho(msg, fg="yellow")
            typer.echo()
            return True

        typer.secho(f"\nFound {len(results)} results", fg=get_system_color(), bold=True)
        typer.secho("─" * 60, fg=get_heading_color())

        for idx, r in enumerate(results, 1):
            ex = get_exchange_for_display(conn, r['exchange_id'], r.get('metadata'))

            typer.secho(f"\n[{idx}]", fg=get_heading_color(), bold=True, nl=False)

            # Show timestamp if available
            if ex.get('created_at'):
                try:
                    ts = datetime.fromisoformat(ex['created_at'].replace('Z', '+00:00'))
                    typer.secho(f" {ts.strftime('%Y-%m-%d %H:%M')}", fg=get_text_color(), dim=True)
                except Exception:
                    typer.echo()
            else:
                typer.echo()

            # User content
            user_content = ex.get('user_content', '')
            if user_content:
                preview = user_content[:200]
                if len(user_content) > 200:
                    preview += "..."
                typer.secho("You: ", fg=get_system_color(), bold=True, nl=False)
                typer.secho(preview, fg=get_text_color())

            # Assistant content
            assistant_content = ex.get('assistant_content', '')
            if assistant_content:
                preview = assistant_content[:200]
                if len(assistant_content) > 200:
                    preview += "..."
                typer.secho("AI:  ", fg=get_system_color(), bold=True, nl=False)
                typer.secho(preview, fg=get_text_color())

        typer.secho("\n" + "─" * 60, fg=get_heading_color())
        typer.echo()

    # Store the meta-query in the database for audit trail
    # Use is_meta_query=True so it's excluded from future retrieval
    try:
        from episodic.db_nodes import insert_node
        from episodic.db import get_head

        # Get current head to maintain conversation continuity
        current_head = get_head()
        parent_id = current_head if current_head else None

        # Insert the meta-query node (no response node for meta-queries)
        insert_node(
            content=user_input,
            parent_id=parent_id,
            role="user",
            is_meta_query=True
        )

        if config.get("debug"):
            typer.secho("[DEBUG] [MQL] Stored meta-query with is_meta_query=True", fg="cyan", dim=True)

    except Exception as e:
        # Don't fail the query handling if storage fails
        if config.get("debug"):
            typer.secho(f"[DEBUG] [MQL] Failed to store meta-query: {e}", fg="red", dim=True)

    return True


def handle_chat_message(user_input: str) -> None:
    """Handle a chat message (non-command input)."""
    global conversation_manager

    if not conversation_manager:
        typer.secho("⚠️  Please run /init first to initialize the database.", fg="yellow")
        return

    # Check if this is a memory query that should be routed to retrieval
    if _handle_memory_query(user_input):
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


async def _voice_listener_task(voice_queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """
    Async task that listens for voice input and puts results in a queue.

    Runs the blocking voice listen() in an executor to avoid blocking the event loop.
    """
    try:
        from episodic.voice import get_voice_manager

        manager = get_voice_manager()
        if not manager.is_active:
            manager.start()

        loop = asyncio.get_event_loop()

        while not stop_event.is_set():
            # Run blocking listen() in executor
            text = await loop.run_in_executor(None, lambda: manager.listen(timeout=2.0))

            if text:
                # Check for sleep commands
                if manager.is_sleep_command(text):
                    manager.force_idle()
                    continue  # Don't put sleep command in queue

                await voice_queue.put(text)
                return  # Got input, exit task

            # Small yield to allow other tasks to run
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        typer.secho(f"Voice input error: {e}", fg="red")


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

    # Voice mode state
    voice_task: Optional[asyncio.Task] = None
    voice_queue: asyncio.Queue = asyncio.Queue()
    voice_stop_event = asyncio.Event()

    def start_voice_listener():
        """Start the voice listener task if voice mode is enabled."""
        nonlocal voice_task
        if voice_task is None or voice_task.done():
            voice_stop_event.clear()
            voice_task = asyncio.create_task(
                _voice_listener_task(voice_queue, voice_stop_event)
            )

    def stop_voice_listener():
        """Stop the voice listener task."""
        nonlocal voice_task
        voice_stop_event.set()
        if voice_task and not voice_task.done():
            voice_task.cancel()
        voice_task = None

    # Main loop
    while True:
        try:
            user_input = None

            # Check for voice mode
            if config.get("voice_mode", False):
                # Start voice listener if not running
                start_voice_listener()

                # Race between voice input and keyboard input
                prompt_task = asyncio.create_task(session.prompt_async())

                done, pending = await asyncio.wait(
                    [prompt_task, voice_task] if voice_task else [prompt_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Get result from completed task
                for task in done:
                    if task is prompt_task:
                        user_input = task.result()
                        stop_voice_listener()
                    elif task is voice_task:
                        # Voice task completed - get from queue
                        try:
                            user_input = voice_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        # Restart voice listener for next input
                        voice_task = None

                if user_input is None:
                    continue

                # Check for voice exit command (spoken)
                input_lower = user_input.lower().strip()
                voice_off_phrases = [
                    "exit voice", "voice off", "stop voice", "disable voice",
                    "turn off voice", "voice mode off", "stop listening",
                ]
                if any(phrase in input_lower for phrase in voice_off_phrases):
                    from episodic.commands.voice import voice_off
                    stop_voice_listener()
                    voice_off()
                    continue
            else:
                # Not in voice mode - stop any running voice listener
                stop_voice_listener()
                # Get user input using async prompt
                user_input = await session.prompt_async()

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
            if config.get("voice_mode", False):
                from episodic.voice import get_voice_manager
                get_voice_manager().pause_idle_timer()

            try:
                # Check if it's a command
                if user_input.startswith('/'):
                    should_exit = handle_command(user_input)
                    if should_exit:
                        # Finalize any ongoing topics before exit
                        from episodic.db import database_exists
                        if database_exists() and conversation_manager:
                            conversation_manager.finalize_current_topic()
                        stop_voice_listener()
                        typer.secho("\nGoodbye! 👋", fg=get_system_color())
                        break
                else:
                    # It's a chat message
                    handle_chat_message(user_input)
            finally:
                # Resume idle timer after processing
                if config.get("voice_mode", False):
                    from episodic.voice import get_voice_manager
                    get_voice_manager().resume_idle_timer()

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
                stop_voice_listener()
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
            stop_voice_listener()
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
        
        # Update the module-level instance in conversation.py to use the same instance
        import episodic.conversation
        episodic.conversation.conversation_manager = conversation_manager
        
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
            
            # Update the module-level instance in conversation.py to use the same instance
            import episodic.conversation
            episodic.conversation.conversation_manager = conversation_manager
            
        conversation_manager.initialize_conversation()
        
        # Execute the script
        execute_script(execute)
        
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
