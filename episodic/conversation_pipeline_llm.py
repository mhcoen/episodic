"""
Conversation pipeline: LLM query and post-processing phases.

Phases 7-9 of handle_chat_message():
  - phase_message_augmentation: persona, style, reflection, voice
  - phase_llm_query: dispatches to muse or regular LLM path
  - phase_postprocessing: store response, indexing, topic boundaries

Split from conversation_pipeline.py for size compliance.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from episodic.conversation import ConversationManager
    from episodic.conversation_pipeline import TurnContext


def _select_muse_source_urls(web_context: Dict[str, Any]) -> List[str]:
    """Select display source URLs based on muse source selection config."""
    from episodic.config import config

    if not isinstance(web_context, dict):
        return []

    results = web_context.get("results", [])
    if not isinstance(results, list):
        return []

    urls: List[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)

    sources_config = config.get("muse_sources", "top-three")
    if sources_config == "first-only":
        return urls[:1]
    if sources_config in ("top-three", "selective"):
        return urls[:3]
    return urls


def _print_muse_sources_footer(web_context: Dict[str, Any]) -> None:
    """Print Muse source URLs for display only (not part of spoken stream)."""
    import typer

    from episodic.configuration import get_system_color

    urls = _select_muse_source_urls(web_context)
    if not urls:
        return

    typer.echo("")
    typer.secho("Sources:", fg=get_system_color(), bold=True)
    for i, url in enumerate(urls, 1):
        typer.secho(f"  [{i}]", fg="cyan", nl=False)
        typer.echo(f" {_format_clickable_url(url)}")


def _format_clickable_url(url: str) -> str:
    """Format URL as OSC 8 hyperlink when terminal likely supports it."""
    if not isinstance(url, str) or not url:
        return ""

    if not _supports_osc8_hyperlinks():
        return url

    # OSC 8 format: ESC ] 8 ; ; URL ESC \ LABEL ESC ] 8 ; ; ESC \
    esc = "\033"
    return f"{esc}]8;;{url}{esc}\\{url}{esc}]8;;{esc}\\"


def _supports_osc8_hyperlinks() -> bool:
    """Best-effort terminal capability check for OSC 8 hyperlinks."""
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False

    # macOS Terminal generally does not support OSC 8 hyperlink rendering.
    if os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
        return False

    term = os.environ.get("TERM", "").lower()
    if term == "dumb":
        return False

    # User opt-out for compatibility.
    if os.environ.get("NO_OSC8") == "1" or os.environ.get("EPISODIC_NO_OSC8") == "1":
        return False

    return True


# ---------------------------------------------------------------------------
# Phase 7: Message augmentation
# ---------------------------------------------------------------------------

def phase_message_augmentation(manager: "ConversationManager", ctx: "TurnContext") -> None:
    """Insert memory/reactivation context, apply persona, style, reflection, voice."""
    import typer

    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_system_color
    from episodic.debug_utils import debug_print
    from episodic.topics import _display_topic_evolution

    # Insert memory context if found
    if ctx.memory_context and ctx.messages:
        # Find the position to insert memory context (before current user message)
        if len(ctx.messages) >= 2 and ctx.messages[-1]["role"] == "user":
            # Insert as a system message before the current user message
            memory_msg = {"role": "system", "content": ctx.memory_context}
            ctx.messages.insert(-1, memory_msg)

    # Insert reactivation packet if topic was reactivated
    if ctx.reactivation_packet and ctx.messages:
        if len(ctx.messages) >= 2 and ctx.messages[-1]["role"] == "user":
            # Insert as a system message before the current user message
            reactivation_msg = {"role": "system", "content": ctx.reactivation_packet}
            ctx.messages.insert(-1, reactivation_msg)

    # Display topic evolution if requested
    if config.get("show_topics") and ctx.raw_messages:
        _display_topic_evolution(ctx.user_node_id)

    # Display drift if enabled
    if config.get("show_drift"):
        manager.display_semantic_drift(ctx.user_node_id, cached_drift=ctx.semantic_drift)

    # Display memory indicator if enabled
    if ctx.memory_indicator and config.get("memory_show_indicators", True):
        typer.echo("")
        secho_color(ctx.memory_indicator, fg=get_system_color())

    # Apply active persona/prompt if set
    active_prompt_name = config.get("active_prompt")
    if active_prompt_name:
        from episodic.prompt_manager import load_prompt

        prompt_data = load_prompt(active_prompt_name)
        if prompt_data and prompt_data.get("content"):
            # Insert persona as first system message
            persona_msg = {"role": "system", "content": prompt_data["content"]}
            ctx.messages.insert(0, persona_msg)

    # Add global style and detail prompts to messages for non-muse modes
    if not config.get("muse_mode"):
        from episodic.commands.detail import get_detail_prompt
        from episodic.commands.style import get_style_prompt

        style_prompt = get_style_prompt(
            has_rag=bool(ctx.rag_context),
            rag_length=len(ctx.rag_context) if ctx.rag_context else 0,
            has_web=bool(ctx.web_context),
        )

        detail_prompt = get_detail_prompt()

        # Combine style and detail prompts
        combined_prompt = f"{style_prompt}\n\n{detail_prompt}"

        # Add combined prompt as a system message before the user's current message
        if ctx.messages and ctx.messages[-1]["role"] == "user":
            # Insert combined instruction before the latest user message
            last_user_message = ctx.messages.pop()
            enhanced_content = (
                f"{combined_prompt}\n\nUser: {last_user_message['content']}"
            )
            last_user_message["content"] = enhanced_content
            ctx.messages.append(last_user_message)

    # Apply reflection mode if enabled
    if config.get("reflection_mode", False):
        from episodic.commands.reflection import handle_reflection_in_conversation

        ctx.messages = handle_reflection_in_conversation(ctx.user_input, ctx.messages)

    # Apply voice persona if voice mode is enabled
    if config.get("voice_mode", False):
        from episodic.voice.voice_persona import get_voice_system_prompt_addition

        voice_prompt = get_voice_system_prompt_addition()

        # Insert voice persona as system message near the end
        if ctx.messages and ctx.messages[-1]["role"] == "user":
            last_user_message = ctx.messages.pop()
            ctx.messages.append({"role": "system", "content": voice_prompt})
            ctx.messages.append(last_user_message)
        else:
            ctx.messages.append({"role": "system", "content": voice_prompt})


# ---------------------------------------------------------------------------
# Phase 8: LLM query
# ---------------------------------------------------------------------------

def phase_llm_query(manager: "ConversationManager", ctx: "TurnContext") -> None:
    """Execute LLM query (muse-mode synthesis or regular query)."""
    import typer

    from episodic.config import config
    from episodic.debug_utils import debug_print

    # Muse mode is fail-closed: if web search has no results, do not call LLM.
    if config.get("muse_mode"):
        if ctx.web_context:
            _phase_llm_muse(manager, ctx)
            return

        error_info = None
        if isinstance(ctx.context_debug, dict):
            error_info = ctx.context_debug.get("web_search_error")

        typer.echo("")
        typer.secho("❌ Muse web search unavailable.", fg="yellow", bold=True)
        if isinstance(error_info, dict):
            details = error_info.get("details", [])
            if details:
                typer.secho("Tried: " + "; ".join(details[:3]), fg="yellow")
            else:
                typer.secho(error_info.get("summary", "No provider details available."), fg="yellow")
        else:
            typer.secho(
                "All configured web providers failed or returned no results.",
                fg="yellow",
            )

        typer.secho("No answer generated.", fg="yellow")
        debug_print("Muse fail-closed: no web context available for synthesis", category="muse")
        ctx.early_return = True
        ctx.early_return_value = (None, None)
        return

    _phase_llm_regular(manager, ctx)


def _phase_llm_muse(manager: "ConversationManager", ctx: "TurnContext") -> None:
    """Handle muse-mode web synthesis LLM path.

    Security: synthesis is buffered (not streamed) so canary detection can
    run before the response is displayed or stored (INV-MUSE-6, Erratum 1).
    """
    import typer
    import uuid

    from episodic.benchmark import benchmark_resource
    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_error_color
    from episodic.db import insert_node
    from episodic.debug_utils import debug_print
    from episodic.web_synthesis import synthesize_web_response

    # Generate session canary for injection detection (INV-MUSE-6)
    session_canary = None
    try:
        from episodic.mcp.security.canary import generate_canary
        canary_session_id = f"muse-{uuid.uuid4()}"
        session_canary = generate_canary(canary_session_id)
    except Exception:
        debug_print("Could not generate session canary", category="security")

    # Debug: print conversation history
    if config.get("debug"):
        debug_print(f"Muse mode: passing {len(ctx.messages)} messages to synthesis", category="muse")
        for i, msg in enumerate(ctx.messages):
            debug_print(f"  Message {i}: {msg['role']} - {msg['content'][:50]}...", category="muse")

    # Use web synthesis for muse mode
    try:
        synthesis_result = synthesize_web_response(
            query=ctx.user_input,
            search_results=ctx.web_context,
            conversation_history=ctx.messages,
            model=ctx.model,
            session_canary=session_canary,
        )

        if config.get("debug"):
            debug_print(f"Synthesis result type: {type(synthesis_result)}", category="muse")
            if isinstance(synthesis_result, dict):
                debug_print(f"Synthesis dict keys: {synthesis_result.keys()}", category="muse")
            debug_print(f"Synthesis result: {str(synthesis_result)[:200]}", category="muse")
    except Exception as e:
        typer.secho(f"\n\u274c Web synthesis error: {e}", fg=get_error_color())
        if config.get("debug"):
            import traceback

            traceback.print_exc()
        ctx.early_return = True
        ctx.early_return_value = (None, None)
        return

    # Handle dict synthesis result (always expected from restructured synthesizer)
    if isinstance(synthesis_result, dict) and synthesis_result.get("streaming"):
        from episodic.llm import _execute_llm_query

        # Preserve conversation history by appending synthesis to existing messages
        synthesis_messages = ctx.messages.copy() if ctx.messages else []

        # Add synthesis as the final exchange
        synthesis_messages.append(
            {"role": "system", "content": synthesis_result["system_message"]}
        )
        synthesis_messages.append(
            {"role": "user", "content": synthesis_result["prompt"]}
        )

        # Gap B: Token guard for muse mode
        from episodic.token_guard import TokenBudget, guard_assembly

        token_budget = TokenBudget(
            full_cap=config.get("token_full_cap", 8000),
            summary_min=config.get("token_summary_min", 100),
            overhead_reserve=config.get("token_overhead_reserve", 500),
        )
        synthesis_messages, fallback_response = guard_assembly(
            synthesis_messages, token_budget
        )

        if fallback_response:
            ctx.display_response = fallback_response
            debug_print(
                "Token guard triggered fallback response (muse)", category="memory"
            )
            with benchmark_resource("Database", "insert assistant node"):
                if "/" in ctx.model:
                    provider, model_name = ctx.model.split("/", 1)
                else:
                    provider = None
                    model_name = ctx.model
                assistant_node_id, assistant_short_id = insert_node(
                    ctx.display_response,
                    ctx.user_node_id,
                    role="assistant",
                    provider=provider,
                    model=ctx.model,
                )
            manager.set_current_node_id(assistant_node_id)
            manager.add_nodes_to_current_topic(ctx.user_node_id, assistant_node_id)
            typer.echo("")
            secho_color(ctx.display_response, fg=get_error_color())
            ctx.early_return = True
            ctx.early_return_value = (assistant_node_id, ctx.display_response)
            return

        # INV-MUSE-1: No tools parameter on synthesis call
        try:
            # Erratum 1: Buffered output — no streaming for synthesis
            # so canary check can run pre-display (INV-MUSE-6)
            response_text, _ = _execute_llm_query(
                synthesis_messages,
                model=synthesis_result["model"],
                temperature=synthesis_result.get("temperature", 0.3),
                max_tokens=synthesis_result.get("max_tokens", 1500),
                stream=False,
            )
        except Exception as e:
            # Handle model not found errors gracefully
            error_str = str(e).lower()
            model_name = synthesis_result["model"]

            if (
                "not found" in error_str
                or "404" in error_str
                or "does not exist" in error_str
            ):
                typer.echo("")
                typer.secho(
                    f"\u274c Synthesis model '{model_name}' not available",
                    fg=get_error_color(),
                )
                typer.echo("")

                if model_name.startswith("ollama/"):
                    model_only = model_name.replace("ollama/", "")
                    typer.secho(
                        f"   For Ollama models, pull them first:",
                        fg=get_error_color(),
                    )
                    typer.secho(f"     ollama pull {model_only}", fg="cyan")
                    typer.echo("")

                typer.secho(
                    f"   Change to an available model:", fg=get_error_color()
                )
                typer.secho(
                    f"     /set synthesis_model gpt-4o-mini", fg="cyan"
                )
                typer.secho(
                    f"     /set synthesis_model null  (uses your main chat model)",
                    fg="cyan",
                )
                typer.echo("")
                ctx.early_return = True
                ctx.early_return_value = (None, None)
                return
            else:
                # Re-raise other errors
                raise

        # INV-MUSE-6: Canary detection — discard response if canary leaked
        if session_canary and response_text:
            from episodic.mcp.security.canary import detect_canary
            if detect_canary(response_text, session_canary):
                typer.echo("")
                typer.secho(
                    "\u26a0\ufe0f  Security: synthesis response contained "
                    "injected canary token. Response discarded.",
                    fg="yellow", bold=True,
                )
                ctx.canary_leaked = True
                ctx.early_return = True
                ctx.early_return_value = (None, None)
                return

        # Display buffered response using unified text formatter
        typer.echo("")
        from episodic.unified_streaming import unified_stream_text
        unified_stream_text(response_text or "", model=synthesis_result["model"])
        _print_muse_sources_footer(ctx.web_context)
        ctx.display_response = response_text or ""
        # Flag for source_type in postprocessing
        ctx.muse_synthesis = True
    else:
        # Non-streaming response (legacy path)
        ctx.display_response = synthesis_result


def _phase_llm_regular(manager: "ConversationManager", ctx: "TurnContext") -> None:
    """Handle regular (non-muse) LLM query path."""
    import typer

    from episodic.benchmark import benchmark_resource
    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_error_color, get_llm_color
    from episodic.db import insert_node
    from episodic.debug_utils import debug_print
    from episodic.text_formatting import wrapped_llm_print
    from episodic.unified_streaming import unified_stream_response

    with benchmark_resource("LLM Call", f"main query - {ctx.model}"):
        # Debug: Show messages being sent to LLM
        debug_print(f"Messages to LLM ({len(ctx.messages)} total):", category="memory")
        for i, msg in enumerate(ctx.messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                preview = content[:200].replace("\n", " ")
            else:
                preview = str(content)[:200]
            debug_print(f"  [{i}] {role}: {preview}...", category="memory")

        # Gap B: Full-assembly token assertion
        from episodic.token_guard import TokenBudget, guard_assembly

        token_budget = TokenBudget(
            full_cap=config.get("token_full_cap", 8000),
            summary_min=config.get("token_summary_min", 100),
            overhead_reserve=config.get("token_overhead_reserve", 500),
        )
        ctx.messages, fallback_response = guard_assembly(ctx.messages, token_budget)

        if fallback_response:
            # Token guard triggered abort - use fallback response
            ctx.display_response = fallback_response
            debug_print(
                "Token guard triggered fallback response", category="memory"
            )
            # Skip LLM query and store fallback
            with benchmark_resource("Database", "insert assistant node"):
                if "/" in ctx.model:
                    provider, model_name = ctx.model.split("/", 1)
                else:
                    provider = None
                    model_name = ctx.model
                assistant_node_id, assistant_short_id = insert_node(
                    ctx.display_response,
                    ctx.user_node_id,
                    role="assistant",
                    provider=provider,
                    model=ctx.model,
                )
            manager.set_current_node_id(assistant_node_id)
            manager.add_nodes_to_current_topic(ctx.user_node_id, assistant_node_id)
            typer.echo("")
            secho_color(ctx.display_response, fg=get_error_color())
            ctx.early_return = True
            ctx.early_return_value = (assistant_node_id, ctx.display_response)
            return

        # Query the LLM with streaming
        stream_enabled = config.get("stream_responses", True)

        # Get max_tokens from style setting (if not muse mode)
        from episodic.commands.style import get_style_max_tokens

        style_max_tokens = (
            get_style_max_tokens() if not config.get("muse_mode") else None
        )

        if stream_enabled:
            # Get the stream generator
            with benchmark_resource("LLM", f"query stream - {ctx.model}"):
                from episodic.llm import _execute_llm_query

                llm_kwargs = {
                    "messages": ctx.messages,
                    "model": ctx.model,
                    "stream": True,
                }
                if style_max_tokens:
                    llm_kwargs["max_tokens"] = style_max_tokens
                stream_generator, _ = _execute_llm_query(**llm_kwargs)

            # Stream the response
            typer.echo("")  # Newline before streaming
            full_response = unified_stream_response(
                stream_generator=stream_generator, model=ctx.model
            )
            ctx.display_response = full_response
        else:
            # Non-streaming response
            with benchmark_resource("LLM", f"query - {ctx.model}"):
                from episodic.llm import _execute_llm_query

                llm_kwargs = {
                    "messages": ctx.messages,
                    "model": ctx.model,
                    "stream": False,
                }
                if style_max_tokens:
                    llm_kwargs["max_tokens"] = style_max_tokens
                response, cost_info = _execute_llm_query(**llm_kwargs)

            # Display the response
            if response:
                typer.echo("")
                # Debug: Check if response is duplicated
                if config.get("debug", False):
                    typer.echo(
                        f"[DEBUG] Response length: {len(response)} chars", err=True
                    )
                    typer.echo(
                        f"[DEBUG] First 100 chars: {response[:100]}", err=True
                    )
                wrapped_llm_print(response, fg=get_llm_color())
                ctx.display_response = response
            else:
                ctx.display_response = "[No response from LLM]"
                typer.echo("")
                secho_color(ctx.display_response, fg=get_error_color())


# ---------------------------------------------------------------------------
# Phase 9: Post-processing
# ---------------------------------------------------------------------------

def phase_postprocessing(manager: "ConversationManager", ctx: "TurnContext") -> None:
    """Store response, update state, index, handle topic boundaries."""
    from episodic.benchmark import benchmark_resource
    from episodic.color_utils import secho_color
    from episodic.config import config
    from episodic.configuration import get_system_color
    from episodic.db import insert_node

    # Store the assistant's response
    with benchmark_resource("Database", "insert assistant node"):
        # Extract provider from model if present
        if "/" in ctx.model:
            provider, model_name = ctx.model.split("/", 1)
        else:
            provider = None
            model_name = ctx.model

        # INV-MUSE-2: Muse DAG nodes carry source_type='web_synthesis'
        source_type = 'web_synthesis' if getattr(ctx, 'muse_synthesis', False) else 'chat'
        ctx.assistant_node_id, ctx.assistant_short_id = insert_node(
            ctx.display_response,
            ctx.user_node_id,
            role="assistant",
            provider=provider,
            model=ctx.model,
            source_type=source_type,
        )

    # Update current node
    manager.set_current_node_id(ctx.assistant_node_id)

    # Display disambiguation hint if we have pending correction state
    if manager.pending_correction and not ctx.correction_applied:
        from episodic.ui.disambiguation import format_disambiguation_hint

        mode = "voice" if config.get("voice_mode", False) else "text"
        hint = format_disambiguation_hint(manager.pending_correction.runner_ups, mode)
        if hint:
            secho_color(hint, fg=get_system_color(), dim=True)

    # Auto-index in memory system - fire-and-forget (non-blocking)
    # Index if memory RAG OR topic reactivation is enabled (reactivation needs embeddings)
    if config.get("enable_memory_rag", False) or config.get(
        "enable_topic_reactivation", False
    ):
        from episodic.conversation import _fire_and_forget_index

        user_node = {"id": ctx.user_node_id, "content": ctx.user_input, "role": "user"}
        assistant_node = {
            "id": ctx.assistant_node_id,
            "content": ctx.display_response,
            "role": "assistant",
        }
        # Include topic_start_node_id for anchor retrieval filtering
        topic_id = manager.current_topic[1] if manager.current_topic else None
        _fire_and_forget_index(user_node, assistant_node, topic_id)

    # Track RAG usage if applicable
    if ctx.rag_context:
        manager.context_builder.track_rag_usage(ctx.assistant_node_id)

    # Handle topic boundaries
    manager.topic_handler.handle_topic_boundaries(
        ctx.topic_changed,
        ctx.user_node_id,
        ctx.assistant_node_id,
        ctx.topic_change_info,
        ctx.new_topic_name,
    )

    # Check for first topic creation or update ongoing topic
    if config.get("automatic_topic_detection"):
        if not ctx.topic_changed and not manager.current_topic:
            manager.topic_handler.check_and_create_first_topic(
                ctx.user_node_id, ctx.assistant_node_id
            )
        elif manager.current_topic:
            # Update ongoing topic name if needed
            manager.topic_handler.update_ongoing_topic_name(ctx.assistant_node_id)

        # Update centroid for current topic (at checkpoint intervals)
        if manager.current_topic and config.get("enable_topic_reactivation", False):
            manager.topic_handler.update_current_topic_centroid()

    # Add nodes to topic_nodes table for topic-local context assembly
    # Must be after topic boundaries are handled (so current_topic is set)
    manager.add_nodes_to_current_topic(ctx.user_node_id, ctx.assistant_node_id)
