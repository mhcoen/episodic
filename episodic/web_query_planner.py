"""
Query planning helpers for Muse web search fan-out.
"""

from typing import List


def plan_subqueries(query: str, max_queries: int = 3) -> List[str]:
    """Build a small set of related subqueries for parallel provider lookup."""
    q = (query or "").strip()
    if not q:
        return []

    max_queries = max(1, int(max_queries))
    candidates: List[str] = [q]
    ql = q.lower()

    has_time_intent = any(
        token in ql
        for token in ("latest", "today", "recent", "new", "this week", "weekend")
    )
    has_location_intent = " in " in ql or "," in ql

    if has_time_intent:
        candidates.append(f"{q} news")
        candidates.append(f"{q} updates")
    else:
        candidates.append(f"{q} overview")

    if has_location_intent:
        candidates.append(f"{q} official events")

    # Keep order stable and unique.
    seen = set()
    planned: List[str] = []
    for candidate in candidates:
        normalized = " ".join(candidate.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        planned.append(candidate.strip())
        if len(planned) >= max_queries:
            break

    return planned
