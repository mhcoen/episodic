# Episodic Query Understanding and Retrieval System
## Design Document v1.1 (Complete Implementation Spec + Test Plan)

---

## 1. Scope and Goals

**Goal:** Answer/browse/summarize recall queries over a conversational DAG using retrieval (lexical + semantic), with correct scoping (segment, temporal, speaker) and strict non-hallucination behavior on empty retrieval.

**Primary constraints:**
- Deterministic, auditable behavior (stable ordering, explicit drop policies, explicit invariants)
- SQLite is authoritative for nodes/topics; Chroma is a derived index over exchanges
- Branching exists (DAG), but "active head ancestry" is the canonical display lineage

---

## 2. Authoritative Data Model and Invariants

### 2.1 SQLite: nodes

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | UUID primary key |
| `content` | TEXT | Message content |
| `parent_id` | TEXT | Nullable at root |
| `role` | TEXT | 'user', 'assistant', or 'system' |
| `created_at` | TEXT | UTC ISO8601 canonical: `YYYY-MM-DDTHH:MM:SS.ffffffZ` |

**Timestamp invariant:**
- `nodes.created_at` MUST be canonical and therefore lexicographically comparable for chronological ordering and range filters
- Write-path MUST emit exactly canonical format
- Read-path MAY parse to datetime for arithmetic (time windows, rounding, etc.)
- Temporal SQL filters MUST use half-open intervals: `created_at >= start AND created_at < end`

### 2.2 SQLite: topics

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | segment_id (primary key) |
| `name` | TEXT | Topic name |
| `start_node_id` | TEXT | First node in segment |
| `end_node_id` | TEXT | Nullable (NULL = ongoing) |

**Segment invariants (intended; violations are audited):**
- **Ancestry invariant:** At creation time, `start_node_id` and `end_node_id` are ancestors of the then-head on the active branch
- **Membership invariant:** Node belongs to segment S iff encountered while walking parent pointers from `effective_end_node_id` back to `start_node_id` (inclusive)
- **Non-overlap invariant:** Node belongs to at most one segment. If violated, first match wins by `topics.id ASC` and an AUDIT warning is emitted with both segment ids

**Important distinction:**
- Overlap resolution (`topics.id ASC`) is for membership if the invariant is violated
- Segment resolver scoring is separate and may prefer "recent" topics for query matching (see 11.3)

### 2.3 Chroma: conversation Collection (Derived)

| Field | Description |
|-------|-------------|
| Document id | `exchange_id` = `user_node_id` |
| Content | `"User: {user_content}\nAssistant: {assistant_content}"` |
| `metadata.user_id` | user_node_id (should match document id) |
| `metadata.assistant_id` | assistant node id used when embedding |
| `metadata.timestamp` | ISO8601 UTC string |
| Distance metric | Cosine distance (lower = better) |

**Configuration requirement:** Collection MUST use cosine distance; this spec assumes lower distance = more similar.

---

## 3. Connection Management Invariants

### 3.1 Explicit Connection Passing (Mandatory)

All DB access functions take `conn: sqlite3.Connection` as the first argument. No function in the retrieval pipeline may call `get_connection()` internally.

**Required signatures:**
```python
def get_topic(conn, segment_id) -> Optional[Dict]: ...
def get_all_topics(conn) -> List[Dict]: ...  # MUST be ORDER BY id ASC
def get_node(conn, node_id) -> Optional[Dict]: ...
def get_head(conn) -> Optional[str]: ...
def build_ancestry_map(conn, end_id) -> Dict[str, Optional[str]]: ...
```

**Rationale:**
- Temp tables are connection-local
- Transactional consistency
- PRAGMA state consistency

### 3.2 Row Factory Invariant

At connection creation: `conn.row_factory = sqlite3.Row` exactly once. Query code must not mutate `row_factory`.

### 3.3 Migration Connection Mode

FTS migration MUST run on a dedicated connection created with `isolation_level=None` (autocommit mode), and MUST issue `BEGIN EXCLUSIVE` before any DML.

---

## 4. FTS5 (Lexical Search) Configuration

### 4.1 Schema

External-content FTS5 table over `nodes.content`:

```sql
CREATE VIRTUAL TABLE nodes_fts USING fts5(
    content,
    content='nodes',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

Assumption: `nodes` is a normal SQLite table with implicit rowid (not `WITHOUT ROWID`).

### 4.2 Sync Triggers

**nodes_fts_ai (AFTER INSERT):**
```sql
INSERT INTO nodes_fts(rowid, content) VALUES (new.rowid, new.content);
```

**nodes_fts_ad (AFTER DELETE):**
```sql
INSERT INTO nodes_fts(nodes_fts, rowid, content) VALUES('delete', old.rowid, old.content);
```

**nodes_fts_au (AFTER UPDATE):**
```sql
INSERT INTO nodes_fts(nodes_fts, rowid, content) VALUES('delete', old.rowid, old.content);
INSERT INTO nodes_fts(rowid, content) VALUES (new.rowid, new.content);
```

### 4.3 Idempotent Migration

**Requirements:**
- Must use migration connection (`isolation_level=None`)
- Must enforce exclusive write lock: `BEGIN EXCLUSIVE`
- Must be idempotent: `DROP TRIGGER IF EXISTS` (all three) before `CREATE TRIGGER`
- Must backfill existing rows: `INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')`

---

## 5. Exchange Model and Mapping

### 5.1 Exchange Definition

- Display unit and semantic index unit is an "exchange": a user node plus its assistant response
- Exchange anchor id (`exchange_id`) is the user node id

### 5.2 Node-to-Exchange Mapping (Pure Function)

**Rules:**
- `role=user` → `exchange_id = node.id`
- `role=assistant` → `exchange_id = node.parent_id` iff `parent_role == user`
- `role=system` (or unknown) → excluded (None)

Lexical query must join parent node to obtain `parent_role` in the same row.

### 5.3 Assistant Pairing for Display

**Rule:**
1. If result has `metadata.assistant_id`, fetch that node and validate:
   - `candidate.role == assistant`
   - `candidate.parent_id == exchange_id`
   - If valid, use it
2. Otherwise fallback: select assistant child of `exchange_id`:
   - If multiple children: pick earliest `created_at` on current head ancestry; else earliest overall

**Rationale:** Semantic index embedded a specific user+assistant pair; browse/answer must show that same assistant variant when possible.

---

## 6. Segment Membership and Caching

### 6.1 Effective End Node

```python
effective_end = topic['end_node_id'] if topic['end_node_id'] else get_head(conn)
```

**Fail-safe:** If `effective_end` is None (empty DB/no head), return empty membership immediately and emit AUDIT debug. Do not attempt `build_ancestry_map`.

### 6.2 Batched Ancestry Traversal

`build_ancestry_map(conn, end_id)`:
- Uses recursive CTE to fetch full ancestry in one query
- Returns `Dict[str, Optional[str]]` mapping `{id: parent_id}` for all nodes on chain

`compute_segment_nodes(conn, segment_id, effective_end)`:
- Returns `(ordered_list, membership_set)`
- Walks from `effective_end` backwards using ancestry_map until `start_node_id` reached
- If `current_id` not in ancestry_map or `start_node_id` not reached: return empty and AUDIT

### 6.3 Cache Policy

Cache key is `segment_id`, but entry includes `effective_end`.

**Lookup:**
- If `cached.effective_end == current_effective_end`: reuse
- Else: recompute and overwrite entry for this `segment_id`

---

## 7. Segment Scoping Semantics (Tri-State)

### 7.1 Tri-State Representation

Segment scope is `Optional[List[str]]`:
- `None`: No segment scope requested (search globally)
- `[]`: Scope requested but resolution yielded empty (fail-safe: return no results)
- `[ids...]`: Scope requested and resolved to concrete node ids

### 7.2 SegmentFilter Kinds

```python
class FilterKind(Enum):
    NONE = auto()         # No segment restriction
    EMPTY = auto()        # Scope requested but empty → return []
    PENDING_IDS = auto()  # Resolved ids, SQL form not chosen
    IN_CLAUSE = auto()    # WHERE n.id IN (...)
    TEMP_TABLE = auto()   # JOIN temp_table
```

**Invariants:**
- `NONE`, `EMPTY`: `node_ids` is None and `table_name` is None
- `PENDING_IDS`, `IN_CLAUSE`: `node_ids` non-empty and `table_name` None
- `TEMP_TABLE`: `table_name` non-empty and `node_ids` None

### 7.3 Building SegmentFilter from Tri-State

```python
def build_segment_filter(segment_node_ids: Optional[List[str]]) -> SegmentFilter:
    if segment_node_ids is None:
        return SegmentFilter(NONE)
    # Dedupe with stable order
    deduped = list(dict.fromkeys(segment_node_ids))
    if not deduped:
        return SegmentFilter(EMPTY)
    return SegmentFilter(PENDING_IDS, node_ids=deduped)
```

### 7.4 Planning SQL Form

**Rule:**
- `available = sqlite_max_variable_number - other_param_count`
- IN_CLAUSE allowed only if `len(node_ids) <= segment_filter_in_clause_max AND len(node_ids) <= available`
- Otherwise, use TEMP_TABLE

### 7.5 Temp Table Safety

- Names generated internally, must match `^[a-zA-Z0-9_]+$`
- Never derived from user input
- Must drop in `finally` block

---

## 8. Lexical Retrieval (SQLite FTS)

### 8.1 BM25 Orientation (Critical)

FTS5 `bm25()` returns "lower is better". Negate at query time:

```sql
SELECT -bm25(nodes_fts) as bm25_score
ORDER BY bm25_score DESC
```

After negation: **larger = better**.

### 8.2 Lexical Query Requirements

Must return:
- `n.id, n.content, n.role, n.parent_id, n.created_at`
- `p.role as parent_role` (LEFT JOIN)
- `bm25_score` (negated)

**Filters:**
- Segment: NONE (no clause), EMPTY (return []), IN_CLAUSE, TEMP_TABLE
- Speaker: `n.role = ?`
- Temporal: `n.created_at >= ? AND n.created_at < ?`

### 8.3 Empty Target Handling

If target is empty/whitespace:
- `mode=browse`: Return recent exchanges
- `mode=answer/summarize`: Return empty

---

## 9. Semantic Retrieval (Chroma)

### 9.1 Adapter (Strict Drop)

Output `List[Dict]` with:
- `exchange_id` (required)
- `distance` (required; drop if missing)
- `metadata` (optional)

### 9.2 Filters

**Segment:** Use `exchange_id` membership (not `metadata.user_id`)

**Temporal:** Parse `metadata.timestamp`; drop if missing/unparseable when filter active

**Speaker:** Semantic disabled entirely when speaker scope specified

---

## 10. Retrieval Pipeline

### 10.1 Inputs

- `target`: Search text
- `scope.segment`: Tri-state
- `scope.temporal`: Half-open `[start, end)`
- `scope.speaker`: Optional role
- `mode`: answer | browse | summarize
- `max_results`, `weights`

### 10.2 Speaker Routing

If speaker specified:
- Disable semantic
- Lexical only with role filter
- Browse still shows full exchange

### 10.3 Over-Fetch

Fetch `max_results * over_fetch_multiplier` per channel before fusion.

### 10.4 Deterministic Ordering Before Normalization

- Semantic: `(distance ASC, exchange_id ASC)`
- Lexical: `(bm25_score DESC, exchange_id ASC)`

### 10.5 Fusion

- Semantic: `invert=True` (lower distance = higher norm)
- Lexical: `invert=False` (higher bm25 = higher norm)
- Missing channel: norm = 0.0
- Final: `(final_score DESC, exchange_id ASC)`

---

## 11. Query Understanding

### 11.1 Deterministic Extractor

Outputs: `target`, `scope.*`, `mode`, `breadth`, `certainty`

### 11.2 Temporal Resolver

User timezone → UTC half-open intervals.

### 11.3 Segment Resolver

- Lexical: Jaccard token overlap
- Semantic: Embedding similarity (cached)
- Combined: `0.4 * lexical + 0.6 * semantic`
- Tiebreaker: `(combined DESC, topic.id DESC)` (prefer recent)
- If below threshold: return `[]`

---

## 12. Modes and Response Constraints

### 12.1 Browse Mode

- Display full exchange even with speaker scope
- Group by segment
- User timezone display

### 12.2 Answer Mode

- Empty retrieval: "I don't have that in our conversation history." (no LLM)
- Non-empty: LLM with hard constraint to use only excerpts

### 12.3 Summarize Mode

- Empty: "No conversations found to summarize." (no LLM)
- Non-empty: Summarize only retrieved excerpts

---

## 13. AUDIT Logging

- Segment membership failures
- Overlap detection (warning)
- Dropped semantic candidates
- Invalid assistant_id
- Placeholder budget exceeded

---

## 14. Configuration

```python
RETRIEVAL_CONFIG = {
    "retrieval": {
        "semantic_weight": 0.6,
        "lexical_weight": 0.4,
        "max_results": 10,
        "over_fetch_multiplier": 3,
        "segment_score_threshold": 0.3,
        "segment_filter_in_clause_max": 100,
        "sqlite_max_variable_number": 999
    },
    "temporal": {
        "timezone": "America/Chicago",
        "db_timezone": "UTC"
    },
    "display": {
        "max_snippet_length": 200
    }
}
```

---

## 15. Implementation Phases

**Phase 1:** Storage + Lexical (migration, FTS, lexical query, exchange mapping)

**Phase 2:** Segment Mechanics (ancestry, cache, tri-state, SegmentFilter)

**Phase 3:** Semantic + Fusion (adapter, filters, normalization, fusion)

**Phase 4:** Query Understanding + Modes (extractor, resolvers, routing, modes)

---

## 16. Success Criteria

1. Migration idempotency
2. Connection passing (no internal `get_connection()`)
3. BM25 orientation correct
4. Segment tri-state semantics
5. Ongoing segment cache invalidation
6. Speaker scope routing
7. Display consistency with `assistant_id`
8. Temporal half-open boundaries
9. Determinism
10. Empty target behavior
