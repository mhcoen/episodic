"""Shared KG context types, constants, and pure helpers (no config, no LLM).

Leaf module split out of context_source.py so both the mention dictionary and
the closure/get_kg code can depend on it without a cycle. Re-exported from
context_source.
"""

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must',
    'i', 'me', 'my', 'mine', 'we', 'our', 'you', 'your', 'he', 'she',
    'it', 'they', 'them', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom', 'where', 'when', 'how', 'why',
    'not', 'no', 'nor', 'but', 'and', 'or', 'if', 'then', 'so',
    'at', 'by', 'for', 'from', 'in', 'of', 'on', 'to', 'with', 'about',
    'up', 'out', 'off', 'over', 'under', 'again', 'further',
    'much', 'very', 'just', 'also', 'still', 'too', 'more', 'most',
})

PREDICATE_PRIORITY = {
    'has': 0, 'is_a': 1, 'located_at': 2, 'affiliated_with': 3,
    'part_of': 4, 'related_to': 5, 'powered_by': 6, 'role': 7,
    'studies': 8, 'works_on': 9, 'uses': 10, 'prefers': 11, 'wants': 12,
    'deadline': 1, 'scheduled_for': 2, 'starts_at': 2, 'ends_at': 2, 'recurring': 3,
}

KINSHIP_CUES = frozenset({
    'daughter', 'son', 'wife', 'husband', 'partner', 'family',
    'kid', 'child', 'parent', 'mother', 'father', 'sister', 'brother',
    'spouse', 'sibling',
})
DEVICE_CUES = frozenset({
    'laptop', 'macbook', 'computer', 'machine', 'desktop', 'phone',
    'spec', 'specs', 'ram', 'gpu', 'cpu', 'memory', 'storage',
    'run', 'models', 'inference', 'keyboard', 'monitor', 'display',
    'device', 'setup', 'rig',
})

PAST_TENSE_CUES = re.compile(
    r'\b(used to|previously|before|back when|formerly|in the past|once had)\b',
    re.IGNORECASE,
)
FIRST_PERSON_CUES = re.compile(
    r'\b(my|i|me|mine|myself|i\'m|i\'ve|i\'d|i\'ll|our|we)\b',
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


@dataclass
class EdgeFact:
    subj_name: str
    predicate: str
    obj_name: str
    source_node_id: int
    rank_score: float
    tags: list[str] = field(default_factory=list)
    assertion_id: int | None = None
    subj_entity_id: int = 0
    obj_entity_id: int = 0


@dataclass
class DerivedFact:
    subj_name: str
    predicate: str
    obj_name: str
    rule: str
    source_node_ids: list[int] = field(default_factory=list)
    relevance_score: float = 0.0
    source_seed_id: int = 0
    intermediate_id: int = 0
    intermediate_name: str = ""


@dataclass
class KGContextResult:
    text: str
    matched_entities: list[dict]
    edge_count: int
    derived_count: int
    budget_used: int
    budget_total: int
    cache_status: str
    edges: list['EdgeFact'] = field(default_factory=list)
    derived: list['DerivedFact'] = field(default_factory=list)
    dropped_edges: list['EdgeFact'] = field(default_factory=list)
    dropped_derived: list['DerivedFact'] = field(default_factory=list)
    suppressed: bool = False
    suppressed_reason: str = ""


def _normalize_surface(text: str) -> str:
    text = unicodedata.normalize('NFC', text.casefold())
    return _PUNCT_RE.sub('', text).strip()


def compute_prompt_tokens(text: str) -> set[str]:
    """Normalize, tokenize, drop stopwords and len<3 tokens."""
    tokens = _normalize_surface(text).split()
    return {t for t in tokens if t not in STOPWORDS and len(t) >= 3}


def _parse_tags(tags_raw) -> list[str]:
    if not tags_raw:
        return []
    if isinstance(tags_raw, list):
        return tags_raw
    try:
        parsed = json.loads(tags_raw)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def _get_cache_key(conn: sqlite3.Connection) -> str:
    try:
        hwm = conn.execute(
            "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
        ).fetchone()
        me = conn.execute(
            "SELECT value FROM kg_state WHERE key = 'merge_epoch'"
        ).fetchone()
        return f"{hwm[0] if hwm else '0'}:{me[0] if me else '0'}"
    except sqlite3.OperationalError:
        return "0:0"
