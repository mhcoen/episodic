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



from .summarization_helpers import (  # noqa: F401
    get_exchanges_since_turn,
    get_max_turn_idx,
    get_stale_topics,
    compute_node_ids_hash,
    preserve_ids,
)

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
