# Spec: KG Edge Deduplication

## Problem
`kg_edges` accumulates duplicate rows for the same semantic triple `(subject_entity_id, predicate, object_entity_id)` when multiple source nodes assert the same fact. Example: `Emma --located_at--> MIT` inserted from both node 570 and node 666. This wastes storage, inflates edge counts, and causes duplicate derived facts in closure rules.

## Fix

### 1. Schema: Add UNIQUE constraint to `kg_edges`

Add a unique index on `(subject_entity_id, predicate, object_entity_id)`:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_kg_edges_triple
ON kg_edges (subject_entity_id, predicate, object_entity_id);
```

Add this to `ensure_kg_schema()` in `schema.py`.

### 2. Applicator: Use INSERT OR REPLACE in `apply_patch()`

Change the edge INSERT in `apply_patch()` from `INSERT INTO kg_edges` to `INSERT OR REPLACE INTO kg_edges`. This means when the same triple is asserted again from a later node, the row is replaced — keeping the most recent `source_node_id` and `assertion_id`.

Alternatively, use `INSERT ... ON CONFLICT(subject_entity_id, predicate, object_entity_id) DO UPDATE SET source_node_id = excluded.source_node_id, assertion_id = excluded.assertion_id` if you want to preserve the original `edge_id` (rowid).

The ON CONFLICT approach is preferable — it preserves the edge_id for any foreign key references while updating provenance.

### 3. Migration

Add a migration step (or handle in `ensure_kg_schema`):
1. Delete duplicate edges, keeping the one with the highest `source_node_id` (most recent):
```sql
DELETE FROM kg_edges WHERE rowid NOT IN (
    SELECT MAX(rowid) FROM kg_edges
    GROUP BY subject_entity_id, predicate, object_entity_id
);
```
2. Then create the unique index.

### 4. Closure dedup (defense in depth)

Even with edge dedup, `apply_closure_rules()` in `context_source.py` can still produce duplicate derived facts if two different closure paths yield the same `(subject, predicate, object)`. Add a dedup pass on the output list keyed on `(subject, predicate, object)`, keeping the first occurrence. This is a one-liner after the closure loop.

## What to verify
- `kg_edges` currently has no unique constraint on the triple columns (check schema.py)
- `apply_patch()` edge insertion SQL (find the INSERT INTO kg_edges statement)
- No FK references to `kg_edges` rowid that would break on REPLACE

## Tests
1. `test_edge_dedup_on_insert`: Apply two patches asserting the same triple from different nodes → only one row in kg_edges, source_node_id is the later node
2. `test_edge_dedup_migration`: Insert duplicates manually, run migration SQL, verify deduped
3. `test_closure_dedup`: Two closure paths yielding same derived triple → only one derived fact in output
