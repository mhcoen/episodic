"""
Episodic Retrieval System

Query understanding and retrieval for conversational memory.
Implements v1.1 spec from docs/design/query_retrieval_v1.1.md

Modules:
- migration: FTS5 setup and migration
- segment_filter: Tri-state segment scoping
- segment: Segment membership and caching  
- lexical: SQLite FTS5 lexical search
- semantic: Chroma semantic search
- fusion: Score normalization and fusion
- display: Exchange display pairing
- pipeline: End-to-end retrieval orchestration
- modes: Mode-specific response formatting
"""

from .segment_filter import SegmentFilter, FilterKind, build_segment_filter
from .migration import migrate_fts5
from .pipeline import retrieve

__all__ = [
    'SegmentFilter',
    'FilterKind', 
    'build_segment_filter',
    'migrate_fts5',
    'retrieve',
]
