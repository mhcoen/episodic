"""
Topic summarization for Episodic.

Generates LLM summaries for topics to populate topic_working_set.summary_md.
This is a lazy/offline operation, not on the hot path.

Supports structured summaries with full provenance tracking.
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from episodic.config import config
from episodic.db_connection import get_connection
from episodic.db_topic_nodes import (
    get_topic_working_set,
    update_topic_summary,
)
from episodic.db_topics import get_all_topics

from .summary_spec import (
    SCHEMA_VERSION,
    SUMMARY_PROMPT,
    LastState,
    StructuredSummary,
)

logger = logging.getLogger(__name__)

# Legacy prompt template (fallback if structured fails)
SUMMARY_PROMPT_TEMPLATE = """Summarize this conversation segment in 2-3 sentences. Focus on:
- Decisions made
- Questions answered
- Open threads or unresolved items

Conversation:
{conversation}

Provide a clear, factual summary."""


@dataclass
class SummaryResult:
    """Result of a summarization operation."""
    topic_start_node_id: str
    topic_name: str
    success: bool
    summary_md: Optional[str] = None
    error: Optional[str] = None
    # Metadata for auditability
    summary_version: int = 0
    last_summarized_turn_idx: int = 0
    model_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    input_token_count: int = 0
    output_token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


def get_exchanges_since_turn(
    topic_start_node_id: str,
    since_turn_idx: Optional[int],
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Get all exchanges in a topic since a specific turn index.

    Args:
        topic_start_node_id: The topic to get exchanges from
        since_turn_idx: Only get exchanges after this turn (None = all)
        conn: Optional database connection

    Returns:
        List of exchange dicts with user/assistant content
    """
    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()

        if since_turn_idx is not None:
            cursor.execute("""
                SELECT tn.node_id, tn.turn_idx, tn.role, n.content
                FROM topic_nodes tn
                JOIN nodes n ON tn.node_id = n.id
                WHERE tn.topic_start_node_id = ?
                AND tn.turn_idx > ?
                AND tn.role IN ('user', 'assistant')
                ORDER BY tn.turn_idx ASC
            """, (topic_start_node_id, since_turn_idx))
        else:
            cursor.execute("""
                SELECT tn.node_id, tn.turn_idx, tn.role, n.content
                FROM topic_nodes tn
                JOIN nodes n ON tn.node_id = n.id
                WHERE tn.topic_start_node_id = ?
                AND tn.role IN ('user', 'assistant')
                ORDER BY tn.turn_idx ASC
            """, (topic_start_node_id,))

        rows = cursor.fetchall()

        # Build exchange pairs
        exchanges = []
        i = 0
        while i < len(rows):
            if rows[i][2] == 'user':
                user_node_id, user_turn_idx, _, user_content = rows[i]
                # Look for following assistant message
                asst_content = None
                asst_node_id = None
                asst_turn_idx = None
                if i + 1 < len(rows) and rows[i + 1][2] == 'assistant':
                    asst_node_id, asst_turn_idx, _, asst_content = rows[i + 1]
                    i += 2
                else:
                    i += 1

                exchanges.append({
                    'user_node_id': user_node_id,
                    'user_content': user_content,
                    'user_turn_idx': user_turn_idx,
                    'assistant_node_id': asst_node_id,
                    'assistant_content': asst_content,
                    'assistant_turn_idx': asst_turn_idx,
                })
            else:
                i += 1

        return exchanges

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_max_turn_idx(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[int]:
    """Get the maximum turn_idx for a topic."""
    def _get(c: sqlite3.Connection) -> Optional[int]:
        cursor = c.cursor()
        cursor.execute("""
            SELECT MAX(turn_idx) FROM topic_nodes
            WHERE topic_start_node_id = ?
        """, (topic_start_node_id,))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def get_stale_topics(
    min_new_exchanges: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Find topics that need summarization.

    A topic is "stale" if:
    - It has no summary yet (last_summarized_turn_idx is NULL)
    - It has grown by min_new_exchanges since last summary

    Args:
        min_new_exchanges: Minimum new exchanges to trigger re-summarization
        conn: Optional database connection

    Returns:
        List of topic dicts with start_node_id, name, exchange_count, etc.
    """
    if min_new_exchanges is None:
        min_new_exchanges = config.get("summary_min_new_exchanges", 4)

    def _get(c: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = c.cursor()

        # Get all topics with their working set state
        cursor.execute("""
            SELECT
                t.start_node_id,
                t.name,
                t.end_node_id,
                ws.last_summarized_turn_idx,
                ws.summary_md,
                (SELECT MAX(turn_idx) FROM topic_nodes WHERE topic_start_node_id = t.start_node_id) as max_turn,
                (SELECT COUNT(*) FROM topic_nodes WHERE topic_start_node_id = t.start_node_id) as node_count
            FROM topics t
            LEFT JOIN topic_working_set ws ON t.start_node_id = ws.topic_start_node_id
            ORDER BY t.rowid ASC
        """)

        stale = []
        for row in cursor.fetchall():
            start_node_id = row[0]
            name = row[1]
            end_node_id = row[2]
            last_summarized = row[3]
            existing_summary = row[4]
            max_turn = row[5]
            node_count = row[6]

            if max_turn is None:
                continue  # Empty topic

            # Calculate new exchanges since last summary
            if last_summarized is None:
                # Never summarized - count all exchanges
                new_exchanges = node_count // 2  # Approximate exchange count
            else:
                # Count turns since last summary
                new_turns = max_turn - last_summarized
                new_exchanges = new_turns // 2  # Approximate

            # Check if stale
            is_stale = (
                last_summarized is None or  # Never summarized
                new_exchanges >= min_new_exchanges  # Has grown enough
            )

            if is_stale:
                stale.append({
                    'start_node_id': start_node_id,
                    'name': name,
                    'end_node_id': end_node_id,
                    'last_summarized_turn_idx': last_summarized,
                    'existing_summary': existing_summary,
                    'max_turn_idx': max_turn,
                    'node_count': node_count,
                    'new_exchanges': new_exchanges,
                })

        return stale

    if conn is not None:
        return _get(conn)

    with get_connection() as c:
        return _get(c)


def compute_node_ids_hash(node_ids: List[str]) -> str:
    """Hash of ordered node IDs for provenance."""
    return hashlib.sha256("|".join(sorted(node_ids)).encode()).hexdigest()[:16]


def preserve_ids(
    old_summary: Optional[StructuredSummary],
    new_summary: StructuredSummary,
) -> StructuredSummary:
    """
    Preserve IDs from old summary when decisions/open_loops match.

    v1 strategy: replace context + last_state, preserve IDs by text matching.
    """
    if not old_summary:
        return new_summary

    # Build lookup by normalized text
    old_decisions = {d.decision.lower().strip(): d.id for d in old_summary.decisions}
    old_loops = {o.question.lower().strip(): o.id for o in old_summary.open_loops}

    # Assign IDs to new decisions
    for d in new_summary.decisions:
        key = d.decision.lower().strip()
        if key in old_decisions:
            d.id = old_decisions[key]

    # Assign IDs to new open_loops
    for o in new_summary.open_loops:
        key = o.question.lower().strip()
        if key in old_loops:
            o.id = old_loops[key]

    return new_summary


def summarize_topic_structured(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Optional[StructuredSummary]:
    """
    Generate structured summary with full provenance.

    Uses replace-only strategy with ID preservation.
    Returns the StructuredSummary or None if summarization was skipped.
    """
    from episodic.db_topic_nodes import ensure_topic_working_set
    from episodic.llm import query_llm

    def _summarize(c: sqlite3.Connection) -> Optional[StructuredSummary]:
        # Get existing working set
        working_set = get_topic_working_set(topic_start_node_id, conn=c)
        topic_name = working_set.get("topic_name", "Unknown") if working_set else "Unknown"

        old_summary = None
        if working_set and working_set.get("summary_json"):
            try:
                old_summary = StructuredSummary.from_json(working_set["summary_json"])
            except Exception as e:
                logger.debug(f"Could not parse old summary JSON: {e}")

        # Get exchanges since last summary (or all if first time)
        last_idx = working_set.get("last_summarized_turn_idx") if working_set else None
        exchanges = get_exchanges_since_turn(
            topic_start_node_id,
            since_turn_idx=None if force else last_idx,
            conn=c,
        )

        min_exchanges = config.get("summary_min_new_exchanges", 4)
        if len(exchanges) < min_exchanges and not force:
            return None

        # Collect all node IDs for provenance
        node_ids = []
        for ex in exchanges:
            if ex.get("user_node_id"):
                node_ids.append(ex["user_node_id"])
            if ex.get("assistant_node_id"):
                node_ids.append(ex["assistant_node_id"])

        input_node_ids_hash = compute_node_ids_hash(node_ids) if node_ids else ""

        # Format exchanges for prompt
        conversation_parts = []
        for ex in exchanges:
            if ex.get("user_content"):
                conversation_parts.append(f"User: {ex['user_content']}")
            if ex.get("assistant_content"):
                conversation_parts.append(f"Assistant: {ex['assistant_content']}")

        exchanges_text = "\n\n".join(conversation_parts)

        # Compute prompt hash
        prompt_hash = hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest()[:16]

        # Get model
        if model is None:
            use_model = config.get("summary_model") or config.get("compression_model")
            if use_model is None:
                use_model = config.get("model", "gpt-4o-mini")
        else:
            use_model = model

        # Call LLM
        prompt = SUMMARY_PROMPT.format(exchanges=exchanges_text)

        try:
            response, cost_info = query_llm(
                prompt,
                system_message="You are a helpful assistant that creates structured summaries. Output only valid JSON.",
                model=use_model,
            )
        except Exception as e:
            logger.error(f"LLM error during structured summarization: {e}")
            return None

        # Parse response
        try:
            # Try to extract JSON from response (might have extra text)
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Remove markdown code blocks
                lines = response_clean.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```json"):
                        in_json = True
                        continue
                    elif line.startswith("```"):
                        in_json = False
                        continue
                    if in_json or (not line.startswith("```")):
                        json_lines.append(line)
                response_clean = "\n".join(json_lines)

            new_summary = StructuredSummary.from_json(response_clean)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse summary JSON: {e}, using fallback")
            # Create minimal valid summary
            new_summary = StructuredSummary(
                schema_version=SCHEMA_VERSION,
                context=response[:500] if response else "Summary generation failed",
                decisions=[],
                open_loops=[],
                last_state=LastState(),
            )

        # Preserve IDs from old summary
        new_summary = preserve_ids(old_summary, new_summary)

        # Compute summary hash
        summary_hash = new_summary.compute_hash()

        # Turn indices
        input_start_turn_idx = (
            exchanges[0].get("user_turn_idx") if exchanges else None
        )
        input_end_turn_idx = (
            exchanges[-1].get("assistant_turn_idx")
            or exchanges[-1].get("user_turn_idx")
            if exchanges
            else None
        )

        # Get max turn idx for the topic
        max_turn = get_max_turn_idx(topic_start_node_id, conn=c)

        # Ensure working set exists
        ensure_topic_working_set(topic_start_node_id, topic_name, conn=c)

        # Update working set with full provenance
        cursor = c.cursor()
        cursor.execute(
            """
            UPDATE topic_working_set SET
                summary_md = ?,
                summary_json = ?,
                schema_version = ?,
                summarizer_model_id = ?,
                prompt_hash = ?,
                input_start_turn_idx = ?,
                input_end_turn_idx = ?,
                input_node_ids_hash = ?,
                summary_hash = ?,
                canonicalizer_version = 1,
                last_summarized_turn_idx = ?,
                last_summarized_at = ?,
                last_updated_at = CURRENT_TIMESTAMP,
                summary_version = summary_version + 1
            WHERE topic_start_node_id = ?
        """,
            (
                new_summary.to_markdown(),
                new_summary.to_canonical_json(),
                SCHEMA_VERSION,
                use_model,
                prompt_hash,
                input_start_turn_idx,
                input_end_turn_idx,
                input_node_ids_hash,
                summary_hash,
                max_turn or input_end_turn_idx,
                datetime.utcnow().isoformat(),
                topic_start_node_id,
            ),
        )
        c.commit()

        return new_summary

    if conn is not None:
        return _summarize(conn)

    with get_connection() as c:
        return _summarize(c)


def summarize_topic(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None,
    model: Optional[str] = None,
    force: bool = False
) -> SummaryResult:
    """
    Generate a summary for a topic if it has grown since last summary.

    Args:
        topic_start_node_id: The topic to summarize
        conn: Optional database connection
        model: Model to use (defaults to compression_model from config)
        force: If True, regenerate even if not stale

    Returns:
        SummaryResult with success/failure and summary text
    """
    from episodic.llm import query_llm

    # Get topic info
    if conn is not None:
        working_set = get_topic_working_set(topic_start_node_id, conn=conn)
    else:
        working_set = get_topic_working_set(topic_start_node_id)

    topic_name = working_set['topic_name'] if working_set else "Unknown"

    # Check if summarization is needed
    last_summarized = working_set.get('last_summarized_turn_idx') if working_set else None
    max_turn = get_max_turn_idx(topic_start_node_id, conn=conn)

    if max_turn is None:
        return SummaryResult(
            topic_start_node_id=topic_start_node_id,
            topic_name=topic_name,
            success=False,
            error="Topic has no exchanges"
        )

    min_new_exchanges = config.get("summary_min_new_exchanges", 4)

    if not force:
        if last_summarized is not None:
            new_turns = max_turn - last_summarized
            if new_turns < min_new_exchanges * 2:  # *2 for user+assistant pairs
                return SummaryResult(
                    topic_start_node_id=topic_start_node_id,
                    topic_name=topic_name,
                    success=True,
                    summary_md=working_set.get('summary_md'),
                    error="No update needed (not enough new exchanges)"
                )

    # Get exchanges to summarize
    exchanges = get_exchanges_since_turn(
        topic_start_node_id,
        since_turn_idx=None if force else last_summarized,
        conn=conn
    )

    if not exchanges:
        return SummaryResult(
            topic_start_node_id=topic_start_node_id,
            topic_name=topic_name,
            success=False,
            error="No exchanges to summarize"
        )

    # Build conversation text
    conversation_parts = []
    for ex in exchanges:
        if ex['user_content']:
            conversation_parts.append(f"User: {ex['user_content']}")
        if ex['assistant_content']:
            conversation_parts.append(f"Assistant: {ex['assistant_content']}")

    conversation_text = "\n\n".join(conversation_parts)

    # Build prompt
    prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation_text)
    prompt_hash = hashlib.md5(SUMMARY_PROMPT_TEMPLATE.encode()).hexdigest()[:8]

    # Determine model
    if model is None:
        model = config.get("summary_model")
        if model is None:
            model = config.get("compression_model")
        if model is None:
            model = config.get("model", "gpt-4o-mini")

    # Call LLM
    try:
        summary, cost_info = query_llm(
            prompt,
            system_message="You are a helpful assistant that creates concise, factual summaries of conversations.",
            model=model
        )

        # Extract token counts from cost_info
        input_tokens = cost_info.get('prompt_tokens', 0) if cost_info else 0
        output_tokens = cost_info.get('completion_tokens', 0) if cost_info else 0

    except Exception as e:
        logger.error(f"LLM error during summarization: {e}")
        return SummaryResult(
            topic_start_node_id=topic_start_node_id,
            topic_name=topic_name,
            success=False,
            error=str(e)
        )

    # Update the working set
    update_success = update_topic_summary(
        topic_start_node_id,
        summary,
        max_turn,
        conn=conn
    )

    if not update_success:
        # Create working set if it doesn't exist
        from episodic.db_topic_nodes import ensure_topic_working_set
        ensure_topic_working_set(topic_start_node_id, topic_name, conn=conn)
        update_success = update_topic_summary(
            topic_start_node_id,
            summary,
            max_turn,
            conn=conn
        )

    # Get updated version
    updated_ws = get_topic_working_set(topic_start_node_id, conn=conn)
    version = updated_ws.get('summary_version', 1) if updated_ws else 1

    return SummaryResult(
        topic_start_node_id=topic_start_node_id,
        topic_name=topic_name,
        success=True,
        summary_md=summary,
        summary_version=version,
        last_summarized_turn_idx=max_turn,
        model_id=model,
        prompt_hash=prompt_hash,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
    )


def summarize_stale_topics(
    force: bool = False,
    model: Optional[str] = None,
    topic_name_filter: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[SummaryResult]:
    """
    Summarize all stale topics.

    Args:
        force: If True, re-summarize even if not stale
        model: Model to use (defaults to config)
        topic_name_filter: Only summarize topics matching this name
        conn: Optional database connection

    Returns:
        List of SummaryResult for each topic processed
    """
    if force:
        # Get all topics
        all_topics = get_all_topics()
        stale = [
            {'start_node_id': t['start_node_id'], 'name': t['name']}
            for t in all_topics
        ]
    else:
        stale = get_stale_topics(conn=conn)

    # Filter by name if specified
    if topic_name_filter:
        stale = [t for t in stale if topic_name_filter.lower() in t['name'].lower()]

    results = []
    for topic in stale:
        result = summarize_topic(
            topic['start_node_id'],
            conn=conn,
            model=model,
            force=force
        )
        results.append(result)

    return results
