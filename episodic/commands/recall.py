"""
Recall command for conversation history retrieval.

Uses the new topic-based recall system for consistent output.

Examples:
    /recall coffee                    # Find coffee-related topics
    /recall when we discussed Python  # Discussion query
    /recall yesterday coffee          # Temporal filter
    /recall did I say coffee          # Speaker filter
"""

from datetime import datetime, timezone
from typing import List

import typer

from episodic.config import config as app_config
from episodic.configuration import get_text_color, get_system_color


def recall_command(args: List[str]):
    """
    Search conversation history using the topic-based recall system.

    Accepts natural language queries:
        /recall coffee                    # Find coffee-related topics
        /recall when we discussed Python  # Discussion query
        /recall yesterday coffee          # Temporal filter
        /recall did I say coffee          # Speaker filter
    """
    from episodic.db_connection import get_connection
    from episodic.query import parse_to_ast, parse_query
    from episodic.query.types import DiscussionQuery, MQLCommand, FreeText
    from episodic.query.classifier import classify_and_extract_intent
    from episodic.query.types import ResolvedQuery
    from episodic.recall import recall
    from episodic.recall.cli_integration import _display_recall_results, _display_no_results

    # Join args into query string
    raw_query = " ".join(args) if args else ""

    if not raw_query.strip():
        typer.secho("Usage: /recall <query>", fg="yellow")
        typer.secho("Examples:", dim=True)
        typer.secho("  /recall coffee", fg=get_text_color())
        typer.secho("  /recall when we discussed Python", fg=get_text_color())
        typer.secho("  /recall yesterday weather", fg=get_text_color())
        return

    now_utc = datetime.now(timezone.utc)
    user_tz = app_config.get("timezone", "America/Chicago")

    # Parse the query
    ast = parse_to_ast(raw_query)

    if app_config.get("debug"):
        typer.secho(f"\n[Recall] AST: {type(ast).__name__}", fg="cyan", dim=True)

    with get_connection() as conn:
        resolved = None

        # DiscussionQuery - use directly
        if isinstance(ast, DiscussionQuery):
            resolved = parse_query(raw_query, conn=conn, now_utc=now_utc, user_tz=user_tz)
            query_form = ast.query_form

        # MQLCommand or FreeText - may need classifier
        elif isinstance(ast, (MQLCommand, FreeText)):
            # Check if it has explicit memory markers
            has_explicit = False
            if isinstance(ast, MQLCommand):
                has_explicit = (
                    any('explicit_mode:' in r for r in ast.audit.rule_path) or
                    ast.segment.explicit or
                    ast.speaker is not None or
                    ast.temporal is not None or
                    ast.deictic is not None
                )

            if has_explicit:
                # Use CFG parse
                resolved = parse_query(raw_query, conn=conn, now_utc=now_utc, user_tz=user_tz)
                query_form = None
            else:
                # Use classifier to extract intent
                if app_config.get("debug"):
                    typer.secho("[Recall] Using classifier...", fg="cyan", dim=True)

                classification = classify_and_extract_intent(raw_query)

                if app_config.get("debug"):
                    typer.secho(f"[Recall] Classifier: target={classification.target}, "
                               f"is_memory={classification.is_memory_query}", fg="cyan", dim=True)

                # For /recall command, always treat as memory query
                # Use classifier's target if available, otherwise use raw query
                target = classification.target if classification.is_memory_query else raw_query

                # Build ResolvedQuery from classifier
                resolved = ResolvedQuery(
                    mode=classification.mode or 'answer',
                    target=target,
                    segment_explicit=False,
                    segment_query=None,
                    segment_resolved_ids=None,
                    segment_ambiguous=False,
                    segment_candidates=None,
                    temporal=None,
                    speaker=classification.speaker_hint if classification.speaker_hint != 'both' else None,
                    deictic=None,
                    has_broadness_cue=False,
                    audit_trace='{"source": "classifier"}'
                )
                query_form = None

        if resolved is None:
            typer.secho(f"\nCouldn't parse query: {raw_query}", fg="yellow")
            return

        if app_config.get("debug"):
            typer.secho(f"[Recall] Target: {resolved.target}", fg="cyan", dim=True)

        # Run recall
        result = recall(conn=conn, query=resolved, query_form=query_form)

        # Display results
        if result.is_empty():
            _display_no_results(resolved.target)
        else:
            _display_recall_results(result)
