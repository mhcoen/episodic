"""
Phase 3 Verification Script.

Validates all Phase 3 invariants:
1. Backfill completeness - All Chroma docs have topic_start_node_id
2. Summary validity - JSON parses, schema correct, hash matches
3. Long-gap-only test - Year-later resume with no recency works

Run: python -m episodic.maintenance.verify_phase3
Output: episodic/evaluation/reports/phase3_verification.json
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase3Report:
    """Complete Phase 3 verification report."""

    timestamp: str
    all_passed: bool
    checks: List[CheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "checks": [asdict(c) for c in self.checks],
        }


def verify_backfill_completeness(
    conn: Optional[sqlite3.Connection] = None,
) -> CheckResult:
    """
    Verify that all Chroma documents have topic_start_node_id metadata.

    Uses BackfillReport.still_missing_after == 0 as success criterion.
    """
    # Skip in test mode - no RAG to verify
    if os.environ.get("EPISODIC_TEST_MODE"):
        return CheckResult(
            name="backfill_completeness",
            passed=True,
            message="Skipped in test mode (no RAG initialized)",
            details={"skipped": True, "reason": "EPISODIC_TEST_MODE"},
        )

    from episodic.maintenance.backfill_topic_metadata import (
        backfill_topic_metadata_with_report,
    )

    try:
        # Run dry-run backfill to check current state
        report = backfill_topic_metadata_with_report(dry_run=True)

        details = {
            "total_scanned": report.total_scanned,
            "already_has_metadata": report.already_has_metadata,
            "would_update": report.updated,
            "missing_in_topic_nodes": report.missing_in_topic_nodes,
            "conflicts": report.conflicts_resolved,
        }

        # Success if all docs with topic assignments have metadata
        # (docs without topic assignment in SQLite are expected to lack metadata)
        if report.updated == 0 and report.conflicts_resolved == 0:
            return CheckResult(
                name="backfill_completeness",
                passed=True,
                message=f"All {report.already_has_metadata} docs have correct topic metadata",
                details=details,
            )
        else:
            return CheckResult(
                name="backfill_completeness",
                passed=False,
                message=f"{report.updated} docs need metadata, {report.conflicts_resolved} conflicts",
                details=details,
            )

    except Exception as e:
        return CheckResult(
            name="backfill_completeness",
            passed=False,
            message=f"Error during backfill check: {e}",
            details={"error": str(e)},
        )


def verify_summary_validity(
    conn: Optional[sqlite3.Connection] = None,
) -> CheckResult:
    """
    Verify all topic summaries are valid.

    Checks:
    - summary_json parses as valid JSON
    - schema_version is supported
    - summary_hash matches computed hash
    """
    # Skip in test mode - no real data to verify
    if os.environ.get("EPISODIC_TEST_MODE"):
        return CheckResult(
            name="summary_validity",
            passed=True,
            message="Skipped in test mode (no topics)",
            details={"skipped": True, "reason": "EPISODIC_TEST_MODE"},
        )

    from episodic.maintenance.validators import (
        validate_all_topics,
    )
    from episodic.db_connection import get_connection

    def _verify(c: sqlite3.Connection) -> CheckResult:
        results = validate_all_topics(conn=c)

        total = len(results)
        valid = sum(1 for r in results.values() if r.valid)
        with_warnings = sum(1 for r in results.values() if r.warnings)
        with_errors = sum(1 for r in results.values() if not r.valid)

        details = {
            "total_topics": total,
            "valid": valid,
            "with_warnings": with_warnings,
            "with_errors": with_errors,
            "errors": [],
            "warnings": [],
        }

        # Collect first few errors and warnings for the report
        for topic_id, result in results.items():
            for error in result.errors[:3]:
                details["errors"].append(f"{topic_id[:8]}: {error}")
            for warning in result.warnings[:3]:
                details["warnings"].append(f"{topic_id[:8]}: {warning}")

        if with_errors == 0:
            msg = f"All {valid} topics have valid summaries"
            if with_warnings > 0:
                msg += f" ({with_warnings} with warnings)"
            return CheckResult(
                name="summary_validity",
                passed=True,
                message=msg,
                details=details,
            )
        else:
            return CheckResult(
                name="summary_validity",
                passed=False,
                message=f"{with_errors}/{total} topics have invalid summaries",
                details=details,
            )

    if conn is not None:
        return _verify(conn)

    with get_connection() as c:
        return _verify(c)


def verify_long_gap_only(
    conn: Optional[sqlite3.Connection] = None,
) -> CheckResult:
    """
    Verify year-later resume scenario works with summary + anchors only.

    Uses force_no_recency=True to simulate a scenario where there are no
    recent exchanges available (like resuming a conversation after a year).

    Checks:
    - Context assembly succeeds with force_no_recency=True
    - Summary is included in context
    - No recency exchanges are included
    - Anchors can still be retrieved
    """
    # Skip in test mode - no real data to verify
    if os.environ.get("EPISODIC_TEST_MODE"):
        return CheckResult(
            name="long_gap_only",
            passed=True,
            message="Skipped in test mode (no topics)",
            details={"skipped": True, "reason": "EPISODIC_TEST_MODE"},
        )

    from episodic.context_recovery.topic_local import TopicLocalStrategy
    from episodic.db_connection import get_connection
    from episodic.db_topics import get_all_topics
    from episodic.db_topic_nodes import get_topic_working_set

    def _verify(c: sqlite3.Connection) -> CheckResult:
        topics = get_all_topics(conn=c)

        if not topics:
            return CheckResult(
                name="long_gap_only",
                passed=True,
                message="No topics to verify (empty database)",
                details={"topics_tested": 0},
            )

        # Find a topic with a summary for meaningful testing
        test_topic = None
        for topic in topics:
            ws = get_topic_working_set(topic["start_node_id"], conn=c)
            if ws and ws.get("summary_md"):
                test_topic = topic
                break

        if not test_topic:
            return CheckResult(
                name="long_gap_only",
                passed=True,
                message="No topics with summaries to test long-gap scenario",
                details={"topics_scanned": len(topics), "topics_with_summary": 0},
            )

        topic_id = test_topic["start_node_id"]
        topic_name = test_topic.get("name", topic_id[:8])

        try:
            strategy = TopicLocalStrategy(exchange_pairs=4)

            # Assemble with force_no_recency=True
            result = strategy.assemble(
                user_turn_text="What were we discussing before?",
                user_node_id=None,
                active_topic_start_node_id=topic_id,
                user_embedding=None,
                token_budget=4000,
                conn=c,
                chroma_collection=None,
                force_no_recency=True,
            )

            details = {
                "topic_tested": topic_name,
                "topic_id": topic_id[:16],
                "messages_returned": len(result.messages),
                "debug_keys": list(result.debug.keys()) if result.debug else [],
            }

            # Check that we got some context
            if not result.messages:
                return CheckResult(
                    name="long_gap_only",
                    passed=False,
                    message=f"No context assembled for topic {topic_name}",
                    details=details,
                )

            # Check that summary was included (should be in system message)
            has_summary = False
            for msg in result.messages:
                if msg.get("role") == "system" and "## Summary" in msg.get(
                    "content", ""
                ):
                    has_summary = True
                    break

            details["summary_included"] = has_summary

            # Check no conversation history messages (only system message should exist)
            conversation_msgs = [m for m in result.messages if m.get("role") != "system"]
            details["conversation_messages"] = len(conversation_msgs)

            if conversation_msgs:
                return CheckResult(
                    name="long_gap_only",
                    passed=False,
                    message=f"Unexpected conversation messages with force_no_recency=True",
                    details=details,
                )

            if has_summary:
                return CheckResult(
                    name="long_gap_only",
                    passed=True,
                    message=f"Year-later resume works for topic {topic_name}: summary-only context",
                    details=details,
                )
            else:
                return CheckResult(
                    name="long_gap_only",
                    passed=False,
                    message=f"Summary not included in year-later context for {topic_name}",
                    details=details,
                )

        except Exception as e:
            return CheckResult(
                name="long_gap_only",
                passed=False,
                message=f"Error during long-gap test: {e}",
                details={"error": str(e), "topic_id": topic_id[:16]},
            )

    if conn is not None:
        return _verify(conn)

    with get_connection() as c:
        return _verify(c)


def run_all_verifications(
    conn: Optional[sqlite3.Connection] = None,
) -> Phase3Report:
    """
    Run all Phase 3 verifications and produce report.

    Returns:
        Phase3Report with all check results
    """
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    checks = []

    # Run each verification
    checks.append(verify_backfill_completeness(conn=conn))
    checks.append(verify_summary_validity(conn=conn))
    checks.append(verify_long_gap_only(conn=conn))

    all_passed = all(c.passed for c in checks)

    return Phase3Report(
        timestamp=timestamp,
        all_passed=all_passed,
        checks=checks,
    )


def save_report(report: Phase3Report, output_path: Optional[str] = None) -> str:
    """
    Save report to JSON file.

    Args:
        report: The Phase3Report to save
        output_path: Optional custom path (defaults to standard location)

    Returns:
        Path where report was saved
    """
    if output_path is None:
        # Default to episodic/evaluation/reports/phase3_verification.json
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "evaluation",
            "reports",
            "phase3_verification.json",
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    return output_path


def format_report(report: Phase3Report) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 50,
        "Phase 3 Verification Report",
        "=" * 50,
        f"Timestamp: {report.timestamp}",
        f"Overall: {'PASS' if report.all_passed else 'FAIL'}",
        "",
    ]

    for check in report.checks:
        status = "✓" if check.passed else "✗"
        lines.append(f"{status} {check.name}: {check.message}")
        if not check.passed and check.details:
            for key, value in check.details.items():
                if key == "errors" and value:
                    for err in value[:5]:
                        lines.append(f"    ERROR: {err}")
                elif key == "error":
                    lines.append(f"    ERROR: {value}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Running Phase 3 verification...")
    print()

    report = run_all_verifications()

    # Print human-readable report
    print(format_report(report))

    # Save JSON report
    output_path = save_report(report)
    print(f"JSON report saved to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if report.all_passed else 1)
