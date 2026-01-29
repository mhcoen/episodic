"""
Resume moment collection for offline response-quality evaluation.

A "resume moment" is a labeled test case where the user returns to a
previously-discussed topic after an intervening gap.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class ResumeMoment:
    """A labeled resume moment for quality evaluation."""

    moment_id: str
    user_query: str
    expected_active_topic: str  # Topic that should be active after this query
    gap_turns: int  # How long topic was dormant (turns since last engagement)
    category: str  # "short_gap", "medium_gap", "long_gap", "ambiguous", "thin_topic"
    cross_topic_import_expected: bool  # Should context include cross-topic info?
    notes: str  # Human-readable description of the test case

    # Optional fields for richer test cases
    intervening_topics: List[str] = field(default_factory=list)  # Topics between resumptions
    expected_context_contains: List[str] = field(default_factory=list)  # Must appear in context
    expected_context_excludes: List[str] = field(default_factory=list)  # Must NOT appear
    expected_contamination: int = 0  # Expected foreign node count (0 for topic_local)


def load_resume_moments(
    path: Optional[Path] = None,
    category: Optional[str] = None,
) -> List[ResumeMoment]:
    """
    Load resume moments from JSON file.

    Args:
        path: Path to JSON file. Defaults to fixtures/resume_moments.json
        category: Optional filter by category (e.g., "short_gap", "ambiguous")

    Returns:
        List of ResumeMoment dataclasses
    """
    if path is None:
        path = FIXTURES_DIR / "resume_moments.json"

    if not path.exists():
        logger.warning(f"Resume moments file not found at {path}, returning empty list")
        return []

    with open(path) as f:
        data = json.load(f)

    moments = [ResumeMoment(**m) for m in data.get("moments", [])]

    if category:
        moments = [m for m in moments if m.category == category]

    return moments


def save_resume_moments(
    moments: List[ResumeMoment],
    path: Optional[Path] = None,
) -> None:
    """Save resume moments to JSON file."""
    if path is None:
        path = FIXTURES_DIR / "resume_moments.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {
                "version": 1,
                "description": "Resume moment test cases for quality evaluation",
                "moments": [asdict(m) for m in moments],
            },
            f,
            indent=2,
        )


def get_moment_by_id(moment_id: str, moments: Optional[List[ResumeMoment]] = None) -> Optional[ResumeMoment]:
    """Get a specific moment by ID."""
    if moments is None:
        moments = load_resume_moments()

    for m in moments:
        if m.moment_id == moment_id:
            return m
    return None


def get_moments_by_category(category: str, moments: Optional[List[ResumeMoment]] = None) -> List[ResumeMoment]:
    """Get all moments in a category."""
    if moments is None:
        moments = load_resume_moments()

    return [m for m in moments if m.category == category]


def validate_moments(moments: List[ResumeMoment]) -> Dict[str, Any]:
    """
    Validate resume moments for completeness and consistency.

    Returns:
        Dict with validation results and any issues found.
    """
    issues = []
    categories = {}
    ids_seen = set()

    for m in moments:
        # Check for duplicate IDs
        if m.moment_id in ids_seen:
            issues.append(f"Duplicate moment_id: {m.moment_id}")
        ids_seen.add(m.moment_id)

        # Count by category
        categories[m.category] = categories.get(m.category, 0) + 1

        # Validate category
        valid_categories = {"short_gap", "medium_gap", "long_gap", "ambiguous", "thin_topic"}
        if m.category not in valid_categories:
            issues.append(f"{m.moment_id}: Invalid category '{m.category}'")

        # Validate gap_turns matches category
        if m.category == "short_gap" and m.gap_turns > 15:
            issues.append(f"{m.moment_id}: short_gap with gap_turns={m.gap_turns}")
        if m.category == "long_gap" and m.gap_turns < 50:
            issues.append(f"{m.moment_id}: long_gap with gap_turns={m.gap_turns}")

        # Check required fields
        if not m.user_query:
            issues.append(f"{m.moment_id}: Empty user_query")
        if not m.expected_active_topic:
            issues.append(f"{m.moment_id}: Empty expected_active_topic")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "moment_count": len(moments),
        "categories": categories,
        "unique_ids": len(ids_seen),
    }


def summarize_moments(moments: Optional[List[ResumeMoment]] = None) -> str:
    """Return a human-readable summary of moments."""
    if moments is None:
        moments = load_resume_moments()

    validation = validate_moments(moments)

    lines = [
        "=== Resume Moments Summary ===",
        f"Total moments: {validation['moment_count']}",
        "",
        "By category:",
    ]

    for cat, count in sorted(validation["categories"].items()):
        lines.append(f"  {cat}: {count}")

    if validation["issues"]:
        lines.append("")
        lines.append("Issues:")
        for issue in validation["issues"][:10]:
            lines.append(f"  - {issue}")
        if len(validation["issues"]) > 10:
            lines.append(f"  ... and {len(validation['issues']) - 10} more")

    return "\n".join(lines)
