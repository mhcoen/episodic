"""Muse (web-synthesis) LLM phase and its source-footer helpers.

Split out of conversation_pipeline_llm.py; re-imported there so phase_llm_query
dispatches to _phase_llm_muse (bare-name call resolves via the re-import) and
external imports of the footer helpers are unchanged.
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


