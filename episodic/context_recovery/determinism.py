"""
Determinism tracking for context assembly.

Creates reproducible fingerprints of context assembly decisions to detect
regressions and enable diffing between runs.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class ContextAssemblyFingerprint:
    """
    Deterministic fingerprint of a context assembly decision.

    All fields that affect context assembly are captured for reproducibility.
    The hash is computed from these fields for quick comparison.
    """
    turn_id: str  # user_node_id
    mode: str
    active_topic_start_node_id: Optional[str]
    included_node_ids: List[str]  # sorted for determinism
    token_counts: Dict[str, int] = field(default_factory=dict)
    reactivation_decision: str = "none"
    reactivation_reason: str = ""
    hash: str = ""

    def __post_init__(self):
        """Compute hash after initialization."""
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash from fields."""
        hash_input = json.dumps({
            "mode": self.mode,
            "active_topic": self.active_topic_start_node_id,
            "included_node_ids": sorted(self.included_node_ids),
            "token_counts": self.token_counts,
            "reactivation": self.reactivation_decision,
            "reason": self.reactivation_reason
        }, sort_keys=True)

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def compute_fingerprint(
    user_node_id: str,
    debug_info: Dict[str, Any]
) -> ContextAssemblyFingerprint:
    """
    Compute a deterministic fingerprint for this turn's context assembly.

    Args:
        user_node_id: The node ID for this user turn
        debug_info: The debug dict from ContextAssemblyResult

    Returns:
        ContextAssemblyFingerprint with computed hash
    """
    # Extract fields from debug info
    mode = debug_info.get("mode", "unknown")
    active_topic = debug_info.get("active_topic_start_node_id") or debug_info.get("topic_start_node_id")
    included_ids = sorted(debug_info.get("included_node_ids", []))
    token_counts = debug_info.get("token_counts", {})

    # Handle reactivation info - may be nested or at top level
    reactivation = debug_info.get("reactivation_decision", "none")
    if isinstance(reactivation, dict):
        reactivation = reactivation.get("action", "none")

    reason = debug_info.get("reactivation_reason", "")
    if isinstance(reason, dict):
        reason = str(reason)

    return ContextAssemblyFingerprint(
        turn_id=user_node_id,
        mode=mode,
        active_topic_start_node_id=active_topic,
        included_node_ids=included_ids,
        token_counts=token_counts,
        reactivation_decision=reactivation,
        reactivation_reason=reason
    )


def persist_fingerprint(
    fingerprint: ContextAssemblyFingerprint,
    conn=None
) -> None:
    """
    Store fingerprint for later diffing.

    Persists to context_assembly_debug table if it has a fingerprint_hash column,
    otherwise logs the fingerprint.

    Args:
        fingerprint: The fingerprint to persist
        conn: Optional database connection
    """
    from episodic.db_connection import get_connection

    def _persist(c):
        cursor = c.cursor()

        # Check if fingerprint column exists
        cursor.execute("PRAGMA table_info(context_assembly_debug)")
        columns = [row[1] for row in cursor.fetchall()]

        if "fingerprint_hash" in columns:
            # Update existing row
            cursor.execute("""
                UPDATE context_assembly_debug
                SET fingerprint_hash = ?
                WHERE user_node_id = ?
            """, (fingerprint.hash, fingerprint.turn_id))
        else:
            # Log instead
            logger.debug(
                f"Fingerprint for {fingerprint.turn_id[:8]}: "
                f"mode={fingerprint.mode}, hash={fingerprint.hash}"
            )

    if conn is not None:
        _persist(conn)
    else:
        try:
            with get_connection() as c:
                _persist(c)
        except Exception as e:
            logger.debug(f"Could not persist fingerprint: {e}")


def diff_fingerprints(
    fp1: ContextAssemblyFingerprint,
    fp2: ContextAssemblyFingerprint
) -> Dict[str, Any]:
    """
    Return differences between two fingerprints.

    Args:
        fp1: First fingerprint (e.g., baseline)
        fp2: Second fingerprint (e.g., current)

    Returns:
        Dict of differences, empty if identical
    """
    diffs = {}

    if fp1.mode != fp2.mode:
        diffs["mode"] = {"old": fp1.mode, "new": fp2.mode}

    if fp1.active_topic_start_node_id != fp2.active_topic_start_node_id:
        diffs["active_topic"] = {
            "old": fp1.active_topic_start_node_id,
            "new": fp2.active_topic_start_node_id
        }

    if fp1.included_node_ids != fp2.included_node_ids:
        old_set = set(fp1.included_node_ids)
        new_set = set(fp2.included_node_ids)
        diffs["included_node_ids"] = {
            "added": sorted(new_set - old_set),
            "removed": sorted(old_set - new_set),
            "old_count": len(fp1.included_node_ids),
            "new_count": len(fp2.included_node_ids)
        }

    if fp1.token_counts != fp2.token_counts:
        diffs["token_counts"] = {
            "old": fp1.token_counts,
            "new": fp2.token_counts
        }

    if fp1.reactivation_decision != fp2.reactivation_decision:
        diffs["reactivation_decision"] = {
            "old": fp1.reactivation_decision,
            "new": fp2.reactivation_decision
        }

    if fp1.reactivation_reason != fp2.reactivation_reason:
        diffs["reactivation_reason"] = {
            "old": fp1.reactivation_reason,
            "new": fp2.reactivation_reason
        }

    return diffs


def format_diff(diffs: Dict[str, Any]) -> str:
    """
    Format fingerprint diff for human-readable output.

    Args:
        diffs: Output from diff_fingerprints()

    Returns:
        Formatted string showing differences
    """
    if not diffs:
        return "No differences"

    lines = ["Fingerprint differences:"]

    for key, value in diffs.items():
        if key == "included_node_ids":
            added = value.get("added", [])
            removed = value.get("removed", [])
            lines.append(f"  {key}:")
            lines.append(f"    old_count: {value['old_count']}")
            lines.append(f"    new_count: {value['new_count']}")
            if added:
                lines.append(f"    added: {added[:5]}{'...' if len(added) > 5 else ''}")
            if removed:
                lines.append(f"    removed: {removed[:5]}{'...' if len(removed) > 5 else ''}")
        elif isinstance(value, dict):
            old = value.get("old", "")
            new = value.get("new", "")
            if isinstance(old, str) and len(old) > 20:
                old = old[:20] + "..."
            if isinstance(new, str) and len(new) > 20:
                new = new[:20] + "..."
            lines.append(f"  {key}: {old} -> {new}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
