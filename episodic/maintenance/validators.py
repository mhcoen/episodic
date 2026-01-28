"""
Validators for summary and anchor invariants.

Provides invariant checking for:
- Summary provenance consistency
- Anchor retrieval correctness
- Cross-module integrity
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from episodic.db_connection import get_connection
from episodic.db_topic_nodes import get_node_topic, get_topic_working_set
from episodic.db_topics import get_all_topics

from .summary_spec import SCHEMA_VERSION, StructuredSummary

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another ValidationResult into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def compute_node_ids_hash(node_ids: List[str]) -> str:
    """Hash of ordered node IDs for provenance."""
    return hashlib.sha256("|".join(sorted(node_ids)).encode()).hexdigest()[:16]


def validate_summary_provenance(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> ValidationResult:
    """
    Validate summary provenance is consistent.

    Checks:
    - summary_hash matches canonical summary bytes
    - schema_version is supported
    - summary_json parses correctly if present
    """
    errors = []
    warnings = []

    def _validate(c: sqlite3.Connection) -> ValidationResult:
        ws = get_topic_working_set(topic_start_node_id, conn=c)
        if not ws:
            return ValidationResult(False, ["No working_set found"], [])

        # Check schema version
        schema_ver = ws.get("schema_version", 0)
        if schema_ver and schema_ver > SCHEMA_VERSION:
            errors.append(f"Unsupported schema version: {schema_ver}")

        # Verify summary_json parses correctly
        summary_json = ws.get("summary_json")
        if summary_json:
            try:
                parsed = StructuredSummary.from_json(summary_json)

                # Verify summary hash
                stored_hash = ws.get("summary_hash")
                if stored_hash:
                    computed_hash = parsed.compute_hash()
                    if computed_hash != stored_hash:
                        errors.append(
                            f"Summary hash mismatch: stored={stored_hash}, computed={computed_hash}"
                        )
            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"Invalid summary_json: {e}")

        # Check provenance fields
        if ws.get("summarizer_model_id") is None and summary_json:
            warnings.append("summary_json present but summarizer_model_id missing")

        if ws.get("prompt_hash") is None and summary_json:
            warnings.append("summary_json present but prompt_hash missing")

        # Check turn indices are consistent
        start_idx = ws.get("input_start_turn_idx")
        end_idx = ws.get("input_end_turn_idx")
        if start_idx is not None and end_idx is not None:
            if end_idx < start_idx:
                errors.append(
                    f"Invalid turn range: start={start_idx} > end={end_idx}"
                )

        return ValidationResult(len(errors) == 0, errors, warnings)

    if conn is not None:
        return _validate(conn)

    with get_connection() as c:
        return _validate(c)


def validate_anchor_invariants(
    anchors: List[Dict[str, Any]],
    active_topic_id: str,
    recency_node_ids: Set[str],
    token_budget: int,
    conn: Optional[sqlite3.Connection] = None,
) -> ValidationResult:
    """
    Validate anchor retrieval invariants.

    Checks:
    - All anchors have topic_start_node_id = active topic
    - No anchors overlap with recency slice
    - Total tokens within budget
    """
    errors = []
    warnings = []

    def _validate(c: sqlite3.Connection) -> ValidationResult:
        total_tokens = 0

        for anchor in anchors:
            node_id = anchor.get("node_id") or anchor.get("user_node_id")
            if not node_id:
                warnings.append("Anchor missing node_id")
                continue

            # Check topic membership
            node_topic = get_node_topic(node_id, conn=c)
            if node_topic and node_topic != active_topic_id:
                errors.append(
                    f"Anchor {node_id[:8]} belongs to topic {node_topic[:8] if node_topic else 'None'}, "
                    f"not {active_topic_id[:8]}"
                )

            # Check recency overlap
            if node_id in recency_node_ids:
                errors.append(f"Anchor {node_id[:8]} overlaps with recency slice")

            total_tokens += anchor.get("tokens", 0)

        # Check budget
        if total_tokens > token_budget:
            errors.append(
                f"Token budget exceeded: {total_tokens} > {token_budget}"
            )

        return ValidationResult(len(errors) == 0, errors, warnings)

    if conn is not None:
        return _validate(conn)

    with get_connection() as c:
        return _validate(c)


def validate_all_topics(
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, ValidationResult]:
    """
    Validate all topics and return results by topic.

    Returns:
        Dict mapping topic_start_node_id to ValidationResult.
    """

    def _validate_all(c: sqlite3.Connection) -> Dict[str, ValidationResult]:
        results = {}
        topics = get_all_topics(conn=c)

        for topic in topics:
            topic_id = topic["start_node_id"]
            results[topic_id] = validate_summary_provenance(topic_id, conn=c)

        return results

    if conn is not None:
        return _validate_all(conn)

    with get_connection() as c:
        return _validate_all(c)


def validate_working_set_completeness(
    topic_start_node_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> ValidationResult:
    """
    Validate that a working set has all required fields populated.

    Checks for Phase 3 completeness:
    - schema_version populated
    - All provenance fields populated if summary exists
    - summary_json parses to valid StructuredSummary
    """
    errors = []
    warnings = []

    def _validate(c: sqlite3.Connection) -> ValidationResult:
        ws = get_topic_working_set(topic_start_node_id, conn=c)
        if not ws:
            return ValidationResult(False, ["No working_set found"], [])

        # Check schema_version
        if ws.get("schema_version") is None:
            warnings.append("schema_version not set (legacy summary)")

        # If we have summary_json, check all provenance fields
        if ws.get("summary_json"):
            required_provenance = [
                "summarizer_model_id",
                "prompt_hash",
                "summary_hash",
            ]
            for field in required_provenance:
                if ws.get(field) is None:
                    warnings.append(f"Provenance field {field} not populated")

            # Verify summary_json parses
            try:
                parsed = StructuredSummary.from_json(ws["summary_json"])
                # Check it has meaningful content
                if not parsed.context and not parsed.decisions and not parsed.open_loops:
                    warnings.append("Structured summary has no content")
            except Exception as e:
                errors.append(f"summary_json invalid: {e}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    if conn is not None:
        return _validate(conn)

    with get_connection() as c:
        return _validate(c)


def format_validation_report(
    results: Dict[str, ValidationResult],
    topic_names: Optional[Dict[str, str]] = None,
) -> str:
    """Format validation results as human-readable report."""
    lines = ["=== Topic Validation Report ===", ""]

    total = len(results)
    valid = sum(1 for r in results.values() if r.valid)
    with_warnings = sum(1 for r in results.values() if r.warnings)
    with_errors = sum(1 for r in results.values() if not r.valid)

    lines.append(f"Total topics: {total}")
    lines.append(f"Valid: {valid}")
    lines.append(f"With warnings: {with_warnings}")
    lines.append(f"With errors: {with_errors}")
    lines.append("")

    # Show details for topics with issues
    if with_errors > 0 or with_warnings > 0:
        lines.append("--- Issues ---")
        for topic_id, result in results.items():
            if not result.valid or result.warnings:
                name = topic_names.get(topic_id, topic_id[:8]) if topic_names else topic_id[:8]
                lines.append(f"\n{name}:")
                for error in result.errors:
                    lines.append(f"  ERROR: {error}")
                for warning in result.warnings:
                    lines.append(f"  WARN: {warning}")

    lines.append("")
    status = "PASS" if with_errors == 0 else "FAIL"
    lines.append(f"Status: {status}")

    return "\n".join(lines)
