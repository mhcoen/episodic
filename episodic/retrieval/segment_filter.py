"""
Segment filter types and builders.

Implements tri-state segment scoping per v1.1 spec section 7.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List


class FilterKind(Enum):
    """Segment filter kinds."""
    NONE = auto()         # No segment restriction requested
    EMPTY = auto()        # Scope requested but empty -> return []
    PENDING_IDS = auto()  # Resolved ids exist, SQL form not chosen
    IN_CLAUSE = auto()    # Use WHERE n.id IN (...)
    TEMP_TABLE = auto()   # Use JOIN temp_table


@dataclass
class SegmentFilter:
    """
    Segment filter with validated invariants.
    
    Invariants per spec 7.2:
    - NONE, EMPTY: node_ids is None and table_name is None
    - PENDING_IDS, IN_CLAUSE: node_ids non-empty and table_name None
    - TEMP_TABLE: table_name non-empty and node_ids None
    """
    kind: FilterKind
    node_ids: Optional[List[str]] = None
    table_name: Optional[str] = None
    
    def __post_init__(self):
        if self.kind in (FilterKind.NONE, FilterKind.EMPTY):
            assert self.node_ids is None and self.table_name is None, \
                f"{self.kind} must have node_ids=None and table_name=None"
        elif self.kind in (FilterKind.PENDING_IDS, FilterKind.IN_CLAUSE):
            assert self.node_ids is not None and len(self.node_ids) > 0, \
                f"{self.kind} must have non-empty node_ids"
            assert self.table_name is None, \
                f"{self.kind} must have table_name=None"
        elif self.kind == FilterKind.TEMP_TABLE:
            assert self.table_name is not None, \
                "TEMP_TABLE must have table_name"
            assert self.node_ids is None, \
                "TEMP_TABLE must have node_ids=None"


def build_segment_filter(segment_node_ids: Optional[List[str]]) -> SegmentFilter:
    """
    Build SegmentFilter from tri-state input.
    
    Args:
        segment_node_ids: Tri-state input
            - None: No scope requested
            - []: Scope requested but empty
            - [ids]: Scope with concrete ids
    
    Returns:
        SegmentFilter with appropriate kind
    """
    if segment_node_ids is None:
        return SegmentFilter(kind=FilterKind.NONE)
    
    # Dedupe with stable order (first occurrence kept)
    seen = set()
    deduped = []
    for nid in segment_node_ids:
        if nid not in seen:
            seen.add(nid)
            deduped.append(nid)
    
    if not deduped:
        return SegmentFilter(kind=FilterKind.EMPTY)
    
    return SegmentFilter(kind=FilterKind.PENDING_IDS, node_ids=deduped)


def plan_sql_filter(
    filter: SegmentFilter,
    other_param_count: int,
    config: dict
) -> SegmentFilter:
    """
    Plan SQL form for PENDING_IDS filter.
    
    Converts PENDING_IDS to IN_CLAUSE if within budget,
    otherwise keeps as PENDING_IDS for temp table at execution.
    
    Args:
        filter: SegmentFilter to plan
        other_param_count: Count of non-segment params in query
        config: Must have segment_filter_in_clause_max, sqlite_max_variable_number
    
    Returns:
        SegmentFilter with IN_CLAUSE or unchanged PENDING_IDS
    """
    if filter.kind != FilterKind.PENDING_IDS:
        return filter
    
    node_ids = filter.node_ids
    max_vars = config['sqlite_max_variable_number']
    available = max_vars - other_param_count
    in_clause_max = config['segment_filter_in_clause_max']
    
    if len(node_ids) <= in_clause_max and len(node_ids) <= available:
        return SegmentFilter(kind=FilterKind.IN_CLAUSE, node_ids=node_ids)
    
    # Keep as PENDING_IDS - execution will create temp table
    return filter
