"""
CLI integration for recall module.

Provides helpers to invoke recall from CLI and display/inject results.
Handles disambiguation UI when queries are ambiguous.
"""

import sqlite3
import typer
from datetime import datetime, timezone
from typing import Optional, Tuple

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.debug_system import debug_enabled, debug_print

from .pipeline import RecallResultKind


def handle_recall_query(
    user_input: str,
    conn: sqlite3.Connection,
    now_utc: datetime,
    user_tz: str
) -> Tuple[bool, Optional[str]]:
    """
    Handle a recall query using the new recall module.

    Args:
        user_input: Raw user input
        conn: SQLite connection
        now_utc: Current UTC time
        user_tz: User timezone string

    Returns:
        (handled, context_string)
        - handled: True if this was a recall query
        - context_string: Formatted context for LLM injection (or None if displayed directly)
    """
    from episodic.query import parse_to_ast, parse_query
    from episodic.query.types import DiscussionQuery, MQLCommand, FreeText
    from episodic.recall import recall, get_budget_description

    # Parse to AST
    ast = parse_to_ast(user_input)

    # Only handle DiscussionQuery with the new recall system for now
    if not isinstance(ast, DiscussionQuery):
        return False, None

    if debug_enabled("memory"):
        debug_print(f"DiscussionQuery detected: {ast.query_form}", category="memory")

    # Resolve the query
    resolved = parse_query(user_input, conn=conn, now_utc=now_utc, user_tz=user_tz)

    if debug_enabled("memory"):
        debug_print(f"Resolved: target={resolved.target}, temporal={resolved.temporal}", category="memory")

    # Execute recall
    try:
        result = recall(
            conn=conn,
            query=resolved,
            query_form=ast.query_form
        )
    except Exception as e:
        if debug_enabled("memory"):
            debug_print(f"Recall error: {e}", category="memory")
            import traceback
            traceback.print_exc()
        return False, None

    if debug_enabled("memory"):
        debug_print(f"Budget: {get_budget_description(result.budget)}", category="memory")

    # Handle AMBIGUOUS result - disambiguation needed
    if result.kind == RecallResultKind.AMBIGUOUS:
        selected = _handle_disambiguation(resolved.target or "", result)
        token = result.disambiguation_token

        if selected is None:
            # User chose "0" - skip filtering, show all results
            if debug_enabled("memory"):
                debug_print("User skipped disambiguation, showing unfiltered results", category="memory")
            try:
                result = recall(
                    conn=conn,
                    query=resolved,
                    query_form=ast.query_form,
                    skip_ambiguity_check=True,
                )
            except Exception as e:
                if debug_enabled("memory"):
                    debug_print(f"Recall error after skip: {e}", category="memory")
                return False, None
        else:
            # Re-run recall with selected cluster and token for drift detection
            if debug_enabled("memory"):
                debug_print(f"User selected cluster {selected}", category="memory")

            try:
                result = recall(
                    conn=conn,
                    query=resolved,
                    query_form=ast.query_form,
                    selected_cluster=selected,
                    disambiguation_token=token,
                )
            except Exception as e:
                if debug_enabled("memory"):
                    debug_print(f"Recall error after disambiguation: {e}", category="memory")
                return False, None

        # If we got AMBIGUOUS again (drift detected), recurse
        if result.kind == RecallResultKind.AMBIGUOUS:
            typer.secho("\nData changed, please re-select:", fg="yellow")
            return handle_recall_query(user_input, conn, now_utc, user_tz)

    if debug_enabled("memory"):
        debug_print(f"Topics: {len(result.formatted.conversation_blocks)}, "
                    f"Statements: {len(result.formatted.statement_blocks)}", category="memory")
        if result.ranking and result.ranking.ranked_topics:
            for rt in result.ranking.ranked_topics[:3]:
                debug_print(f"Topic {rt.topic_id}: score={rt.score:.3f}, hits={rt.hit_count}", category="memory")
        if result.promotion:
            total_hits = sum(len(v) for v in result.promotion.by_topic.values())
            debug_print(f"Total promoted hits: {total_hits}", category="memory")

    # Check if empty
    if result.is_empty():
        _display_no_results(resolved.target)
        return True, None

    # Display results
    _display_recall_results(result)

    # Return context string for potential LLM injection
    context_string = result.to_context_string()

    return True, context_string


def _handle_disambiguation(target: str, result) -> Optional[int]:
    """
    Display disambiguation options and get user selection.

    Args:
        target: Query target string
        result: RecallResult with kind=AMBIGUOUS

    Returns:
        Selected cluster option_id, or None if cancelled/skip
    """
    if not result.ambiguity or not result.cluster_options:
        return None

    # Display disambiguation prompt
    typer.secho(f"\nI found multiple plausible topics for '{target}':", fg="yellow", bold=True)
    typer.echo()

    for i, opt in enumerate(result.cluster_options, 1):
        # Build label from terms or snippet
        if opt.label_terms:
            label = ", ".join(opt.label_terms[:3])
        else:
            label = opt.label_snippet[:50] + "..." if len(opt.label_snippet) > 50 else opt.label_snippet

        # Show option with hit count
        typer.secho(f"  {i}. ", fg=get_heading_color(), bold=True, nl=False)
        typer.secho(f"{label} ", fg=get_text_color(), nl=False)
        typer.secho(f"({opt.cluster_size} hits)", fg=get_system_color(), dim=True)

        # Show 1-2 representative snippets
        for j, snippet in enumerate(opt.representative_snippets[:2]):
            if len(snippet) > 70:
                snippet = snippet[:70] + "..."
            typer.secho(f"     \"{snippet}\"", fg=get_text_color(), dim=True)

    typer.echo()
    typer.secho(f"  0. Show all results (no filter)", fg=get_system_color(), dim=True)
    typer.echo()

    # Get user input
    try:
        selection = typer.prompt("Which topic? (number)", default="0")
        sel_num = int(selection)

        if sel_num == 0:
            return None  # User wants unfiltered results

        if 1 <= sel_num <= len(result.cluster_options):
            return result.cluster_options[sel_num - 1].option_id

        typer.secho("Invalid selection.", fg="red")
        return None

    except (ValueError, KeyboardInterrupt):
        return None


def _display_no_results(target: Optional[str]):
    """Display message when no recall results found."""
    if target:
        typer.secho(f"\nNo conversations found about '{target}'", fg="yellow")
    else:
        typer.secho("\nNo matching conversations found", fg="yellow")
    typer.echo()


def _display_recall_results(result):
    """Display recall results to the terminal."""
    from episodic.recall import RecallResult
    
    formatted = result.formatted
    budget = result.budget
    
    # Header
    total_topics = len(formatted.conversation_blocks)
    total_statements = len(formatted.statement_blocks)
    
    typer.secho(f"\nFound {total_topics} topic(s), {total_statements} statement(s)", 
                fg=get_system_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    # Conversation blocks
    for block in formatted.conversation_blocks:
        _display_conversation_block(block, budget)
    
    # Statement blocks
    for block in formatted.statement_blocks:
        _display_statement_block(block)
    
    typer.secho("─" * 60, fg=get_heading_color())
    typer.echo()


def _display_conversation_block(block, budget):
    """Display a conversation block."""
    from episodic.recall import ConversationBlock
    
    # Header
    header_parts = [f"[{block.topic_name}"]
    
    if budget.emphasize_timestamps and block.date_range:
        header_parts.append(f", {block.date_range}")
    
    header_parts.append(f", {block.hit_count} matches")
    header_parts.append("]")
    
    typer.secho("\n" + "".join(header_parts), fg=get_heading_color(), bold=True)
    
    # Summary for compressed topics
    if block.is_compressed and block.summary:
        typer.secho("Summary: ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(block.summary, fg=get_text_color(), dim=True)
        if block.exchanges:
            typer.secho("Relevant exchanges:", fg=get_system_color(), dim=True)
    
    # Exchanges
    for exchange in block.exchanges:
        role_label = "You: " if exchange.role == "user" else "AI:  "
        anchor_marker = " *" if exchange.is_anchor else ""
        
        # Truncate long content
        content = exchange.content
        if len(content) > 300:
            content = content[:300] + "..."
        
        typer.secho(f"{role_label}", fg=get_system_color(), bold=True, nl=False)
        if exchange.is_anchor:
            typer.secho(content, fg=get_text_color())
        else:
            typer.secho(content, fg=get_text_color(), dim=True)


def _display_statement_block(block):
    """Display a statement block (exchange pair)."""
    from episodic.recall import StatementBlock
    
    # Header
    header_parts = ["[Statement"]
    
    if block.topic_name:
        header_parts.append(f" from {block.topic_name}")
    
    if block.timestamp:
        header_parts.append(f", {block.timestamp}")
    
    header_parts.append("]")
    
    typer.secho("\n" + "".join(header_parts), fg=get_heading_color(), bold=True)
    
    # User content
    if block.user_content:
        user_content = block.user_content
        if len(user_content) > 300:
            user_content = user_content[:300] + "..."
        typer.secho("You: ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(user_content, fg=get_text_color())
    
    # Assistant content
    if block.assistant_content:
        assistant_content = block.assistant_content
        if len(assistant_content) > 300:
            assistant_content = assistant_content[:300] + "..."
        typer.secho("AI:  ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(assistant_content, fg=get_text_color())


def get_recall_context_for_llm(
    user_input: str,
    conn: sqlite3.Connection,
    now_utc: datetime,
    user_tz: str
) -> Optional[str]:
    """
    Get recall context to inject into LLM prompt.
    
    Use this when you want to augment an LLM response with conversation history.
    Returns None if no relevant history found.
    """
    from episodic.query import parse_to_ast, parse_query
    from episodic.query.types import DiscussionQuery
    from episodic.recall import recall
    
    ast = parse_to_ast(user_input)
    
    if not isinstance(ast, DiscussionQuery):
        return None
    
    resolved = parse_query(user_input, conn=conn, now_utc=now_utc, user_tz=user_tz)
    
    try:
        result = recall(conn=conn, query=resolved, query_form=ast.query_form)
    except Exception:
        return None
    
    if result.is_empty():
        return None
    
    return result.to_context_string()
