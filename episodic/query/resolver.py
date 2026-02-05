"""
MQL Resolver

Converts AST (MQLCommand, DiscussionQuery, FreeText) to ResolvedQuery.

Handles:
- Temporal resolution (DST-safe with zoneinfo)
- Segment disambiguation
- Speaker mapping (both -> None)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from .types import (
    AST,
    DiscussionQuery,
    FreeText,
    MQLCommand,
    ResolvedQuery,
    SegmentResolutionResult,
    TemporalSpec,
)


def resolve_temporal(
    spec: TemporalSpec,
    now_utc: datetime,
    user_tz: str
) -> Optional[Tuple[datetime, datetime]]:
    """
    Resolve temporal spec to UTC half-open [start, end).
    Uses zoneinfo for DST-safe computation.

    Returns timezone-aware datetime objects (tzinfo=UTC).
    """
    tz = ZoneInfo(user_tz)
    utc = ZoneInfo("UTC")
    local_now = now_utc.astimezone(tz)

    def midnight(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    today_local = midnight(local_now)

    if spec.kind == "yesterday":
        start = today_local - timedelta(days=1)
        end = today_local

    elif spec.kind == "today":
        start = today_local
        end = today_local + timedelta(days=1)

    elif spec.kind == "last_week":
        days_to_monday = local_now.weekday()
        this_monday = today_local - timedelta(days=days_to_monday)
        start = this_monday - timedelta(days=7)
        end = this_monday

    elif spec.kind == "this_week":
        days_to_monday = local_now.weekday()
        start = today_local - timedelta(days=days_to_monday)
        end = start + timedelta(days=7)

    elif spec.kind == "last_month":
        first_of_month = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_now.month == 1:
            start = first_of_month.replace(year=local_now.year - 1, month=12)
        else:
            start = first_of_month.replace(month=local_now.month - 1)
        end = first_of_month

    elif spec.kind == "this_month":
        start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_now.month == 12:
            end = start.replace(year=local_now.year + 1, month=1)
        else:
            end = start.replace(month=local_now.month + 1)

    elif spec.kind == "last_year":
        start = local_now.replace(year=local_now.year - 1, month=1, day=1,
                                  hour=0, minute=0, second=0, microsecond=0)
        end = local_now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    elif spec.kind == "last_n_days":
        start = today_local - timedelta(days=spec.n)
        end = today_local + timedelta(days=1)

    elif spec.kind == "n_days_ago":
        start = today_local - timedelta(days=spec.n)
        end = start + timedelta(days=1)

    elif spec.kind == "iso_date":
        d = date.fromisoformat(spec.iso_date)
        start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz)
        end = start + timedelta(days=1)

    else:
        return None

    return (start.astimezone(utc), end.astimezone(utc))


def resolve_segment(
    conn: sqlite3.Connection,
    query: str
) -> SegmentResolutionResult:
    """
    Resolve segment query to node IDs with explicit disambiguation.

    Resolution order:
    1. Normalized exact match -> single result
    2. Normalized contains match:
       - Single match -> use it
       - Multiple matches -> return ambiguous with all candidates
       - No matches -> return empty

    Tie-breaking (for ambiguous display): topic.id ASC (deterministic)
    """
    norm_query = query.lower().replace('-', ' ').replace('_', ' ').strip()
    audit_notes = []

    # Get all topics from database
    topics = _get_all_topics(conn)

    # Phase 1: Exact match
    exact_matches = []
    for topic in topics:
        norm_name = topic['name'].lower().replace('-', ' ').replace('_', ' ')
        if norm_name == norm_query:
            exact_matches.append(topic)

    if len(exact_matches) == 1:
        topic = exact_matches[0]
        nodes = _get_segment_nodes(conn, topic['id'])
        audit_notes.append(f"exact_match:topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )

    if len(exact_matches) > 1:
        # Multiple exact matches (rare, but handle deterministically)
        audit_notes.append(f"multiple_exact_matches:count={len(exact_matches)}")
        # Sort by id ASC for deterministic selection
        exact_matches.sort(key=lambda t: t['id'])
        topic = exact_matches[0]
        nodes = _get_segment_nodes(conn, topic['id'])
        audit_notes.append(f"tie_break:selected_topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=True,
            candidates=[{'id': t['id'], 'name': t['name']} for t in exact_matches],
            audit_notes=audit_notes
        )

    # Phase 2: Contains match
    contains_matches = []
    for topic in topics:
        if norm_query in topic['name'].lower():
            contains_matches.append(topic)

    if len(contains_matches) == 0:
        audit_notes.append("no_match")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=[],
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )

    if len(contains_matches) == 1:
        topic = contains_matches[0]
        nodes = _get_segment_nodes(conn, topic['id'])
        audit_notes.append(f"contains_match:topic_id={topic['id']}")
        return SegmentResolutionResult(
            normalized_query=norm_query,
            node_ids=nodes,
            is_ambiguous=False,
            candidates=None,
            audit_notes=audit_notes
        )

    # Multiple contains matches -> AMBIGUOUS
    audit_notes.append(f"ambiguous_contains:count={len(contains_matches)}")
    contains_matches.sort(key=lambda t: t['id'])  # Deterministic order

    return SegmentResolutionResult(
        normalized_query=norm_query,
        node_ids=[],  # Empty because ambiguous - retrieval should not proceed
        is_ambiguous=True,
        candidates=[{'id': t['id'], 'name': t['name']} for t in contains_matches],
        audit_notes=audit_notes
    )


def _get_all_topics(conn: sqlite3.Connection) -> List[dict]:
    """Get all topics from database."""
    cursor = conn.execute("SELECT id, name FROM topics ORDER BY id")
    return [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]


def _get_segment_nodes(conn: sqlite3.Connection, topic_id: str) -> List[str]:
    """Get node IDs for a topic/segment."""
    # Try topic_node_cache first
    cursor = conn.execute(
        "SELECT node_id FROM topic_node_cache WHERE topic_id = ?",
        (topic_id,)
    )
    nodes = [row[0] for row in cursor.fetchall()]
    if nodes:
        return nodes

    # Fallback: get nodes from topics table using start/end node traversal
    cursor = conn.execute(
        "SELECT start_node_id, end_node_id FROM topics WHERE id = ?",
        (topic_id,)
    )
    row = cursor.fetchone()
    if not row:
        return []

    start_node_id, end_node_id = row
    if not start_node_id:
        return []

    # Simple traversal - get all nodes between start and end
    # This is a simplified version; the full implementation would traverse the DAG
    nodes = []
    cursor = conn.execute(
        "SELECT id FROM nodes WHERE id = ? OR parent_id = ?",
        (start_node_id, start_node_id)
    )
    nodes = [row[0] for row in cursor.fetchall()]
    return nodes


class Resolver:
    """
    MQL resolver that converts AST to ResolvedQuery.

    Handles:
    - Temporal -> UTC half-open [start, end)
    - Segment disambiguation
    - Speaker mapping (both -> None)
    """

    def __init__(self, conn: Optional[sqlite3.Connection], now_utc: datetime, user_tz: str):
        self.conn = conn
        self.now_utc = now_utc
        self.user_tz = user_tz

    def resolve(self, ast: AST) -> ResolvedQuery:
        """Resolve AST to ResolvedQuery."""
        if isinstance(ast, FreeText):
            return self._resolve_freetext(ast)

        if isinstance(ast, DiscussionQuery):
            return self._resolve_discussion_query(ast)

        # MQLCommand
        return self._resolve_command(ast)

    def _resolve_command(self, ast: MQLCommand) -> ResolvedQuery:
        """Resolve MQLCommand to ResolvedQuery."""
        target = ast.target.text if ast.target else None

        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)

        # Segment (explicit gate with disambiguation)
        segment_query = None
        segment_resolved_ids = None
        segment_ambiguous = False
        segment_candidates = None

        if ast.segment.explicit:
            segment_query = ast.segment.query
            if self.conn and segment_query:
                result = resolve_segment(self.conn, segment_query)
                segment_resolved_ids = result.node_ids
                segment_ambiguous = result.is_ambiguous
                segment_candidates = result.candidates
            else:
                # No connection available - return empty list for explicit segment
                segment_resolved_ids = []

        # Speaker (map "both" -> None)
        speaker = None
        if ast.speaker:
            if ast.speaker.role == "both":
                speaker = None
            else:
                speaker = ast.speaker.role

        deictic = ast.deictic.kind if ast.deictic else None

        return ResolvedQuery(
            mode=ast.mode.value,
            target=target,
            segment_explicit=ast.segment.explicit,
            segment_query=segment_query,
            segment_resolved_ids=segment_resolved_ids,
            segment_ambiguous=segment_ambiguous,
            segment_candidates=segment_candidates,
            temporal=temporal,
            speaker=speaker,
            deictic=deictic,
            has_broadness_cue=False,  # Only set by DiscussionQuery
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True),
            ast_kind="MQLCommand",
        )

    def _resolve_freetext(self, ast: FreeText) -> ResolvedQuery:
        """
        FreeText -> well-defined ResolvedQuery with explicit defaults.
        """
        return ResolvedQuery(
            mode="answer",
            target=ast.text,  # s_norm
            segment_explicit=False,
            segment_query=None,
            segment_resolved_ids=None,
            segment_ambiguous=False,
            segment_candidates=None,
            temporal=None,
            speaker=None,
            deictic=None,
            has_broadness_cue=False,
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True),
            ast_kind="FreeText",
        )

    def _resolve_discussion_query(self, ast: DiscussionQuery) -> ResolvedQuery:
        """
        DiscussionQuery -> ResolvedQuery with BROWSE mode, no segment scope.
        """
        target = ast.target.text if ast.target else None

        temporal = None
        if ast.temporal:
            temporal = resolve_temporal(ast.temporal, self.now_utc, self.user_tz)

        # Speaker: map "both" -> None
        speaker = None
        if ast.speaker:
            if ast.speaker.role == "both":
                speaker = None
            else:
                speaker = ast.speaker.role

        return ResolvedQuery(
            mode="browse",  # ALWAYS browse for discussion queries
            target=target,
            segment_explicit=False,  # NEVER segment scope for discussion queries
            segment_query=None,
            segment_resolved_ids=None,
            segment_ambiguous=False,
            segment_candidates=None,
            temporal=temporal,
            speaker=speaker,
            deictic=None,
            has_broadness_cue=ast.has_broadness_cue,
            audit_trace=json.dumps(ast.to_dict(), sort_keys=True),
            ast_kind="DiscussionQuery",
        )


def resolve(ast: AST, conn: Optional[sqlite3.Connection], now_utc: datetime, user_tz: str) -> ResolvedQuery:
    """Convenience function to resolve an AST."""
    return Resolver(conn, now_utc, user_tz).resolve(ast)
