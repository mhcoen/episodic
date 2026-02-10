"""
Memory query routing for Episodic CLI.

This module handles detection and routing of memory queries (recall,
retrieval) vs regular chat messages. Contains the resume-cue detector,
memory query classifier bridge, and the legacy retrieval path.

Extracted from cli_main.py to keep file sizes manageable.
"""

import re

import typer

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


def handle_memory_query(user_input: str) -> bool:
    """
    Check if input is a memory query (DiscussionQuery or explicit MQLCommand).

    If so, route to retrieval system instead of LLM.
    Returns True if handled, False to continue to LLM.
    """
    import sqlite3
    from datetime import datetime, timezone

    from episodic.query import parse_to_ast, parse_query
    from episodic.query.types import DiscussionQuery, FreeText
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
            typer.secho(f"[DEBUG] [MQL] DiscussionQuery \u2192 routing to recall system", fg="cyan", dim=True)

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
                typer.secho(f"[DEBUG] [MQL] {reason} + classifier disabled \u2192 routing to LLM", fg="cyan", dim=True)
            return False

        # Use single LLM call to classify AND extract intent
        from episodic.query.classifier import classify_and_extract_intent
        if config.get("debug"):
            reason = "FreeText" if isinstance(ast, FreeText) else "ambiguous MQLCommand (no explicit markers)"
            typer.secho(f"[DEBUG] [MQL] {reason} \u2192 invoking LLM classifier...", fg="cyan", dim=True)

        classification = classify_and_extract_intent(user_input)

        if config.get("debug"):
            typer.secho(f"[DEBUG] [MQL] Classifier result: is_memory={classification.is_memory_query}, "
                       f"confidence={classification.confidence}", fg="cyan", dim=True)
            if classification.is_memory_query:
                typer.secho(f"[DEBUG] [MQL] Intent: target={classification.target}, mode={classification.mode}, "
                           f"temporal={classification.temporal_hint}, speaker={classification.speaker_hint}", fg="cyan", dim=True)

        if not classification.is_memory_query:
            if config.get("debug"):
                typer.secho("[DEBUG] [MQL] Classifier: GENERAL \u2192 routing to LLM", fg="cyan", dim=True)
            return False

        # Classifier identified as memory query - route to NEW recall system
        if config.get("enable_recall_system", True):
            # Check for resume cues FIRST - if present AND reactivation enabled, fall through to chat
            # Resume cues (e.g., "back to that X thing") indicate conversation continuation, not recall
            if _has_resume_cues(user_input) and config.get("enable_topic_reactivation", False):
                if config.get("debug"):
                    typer.secho("[DEBUG] [MQL] Resume cues detected + reactivation enabled \u2192 falling through to chat", fg="cyan", dim=True)
                return False  # Fall through to chat flow where reactivation will run

            if config.get("debug"):
                typer.secho("[DEBUG] [MQL] Classifier: MEMORY \u2192 routing to recall system", fg="cyan", dim=True)

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
            typer.secho("[DEBUG] [MQL] Classifier: MEMORY \u2192 routing to retrieval (legacy)", fg="cyan", dim=True)

    # Definite non-memory query
    elif not memory_result:
        if config.get("debug"):
            typer.secho("[DEBUG] [MQL] Not a memory query \u2192 routing to LLM", fg="cyan", dim=True)
        return False

    if config.get("debug"):
        typer.secho("[DEBUG] [MQL] Memory query detected \u2192 routing to retrieval (legacy)", fg="cyan", dim=True)

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
                typer.secho(f"[DEBUG] [MQL] {src} \u2192 using classification intent", fg="cyan", dim=True)

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
        typer.secho("\u2500" * 60, fg=get_heading_color())

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

        typer.secho("\n" + "\u2500" * 60, fg=get_heading_color())
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
