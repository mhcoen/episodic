# Phase 1.2b: Entity Merge + Read-Side Regression Tests

## A. Entity Merge Operation

### 1. Schema Changes

#### 1a. Add tombstone columns to kg_entities

Migration function `_migrate_entity_merge()` in `schema.py`, called from `ensure_kg_schema()`.

Check: if column `merged_into_entity_id` already exists on `kg_entities`, return (idempotent).

Add three columns:
```sql
ALTER TABLE kg_entities ADD COLUMN merged_into_entity_id INTEGER NULL REFERENCES kg_entities(entity_id);
ALTER TABLE kg_entities ADD COLUMN merged_at REAL NULL;
ALTER TABLE kg_entities ADD COLUMN merged_reason TEXT NULL;
```

Add index for fast tombstone filtering:
```sql
CREATE INDEX IF NOT EXISTS idx_kg_entities_merged ON kg_entities(merged_into_entity_id) WHERE merged_into_entity_id IS NOT NULL;
```

#### 1b. Create kg_merges table (append-only log)

```sql
CREATE TABLE IF NOT EXISTS kg_merges (
    merge_id INTEGER PRIMARY KEY,
    survivor_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    merged_id INTEGER NOT NULL REFERENCES kg_entities(entity_id),
    created_at REAL NOT NULL,
    created_by_node_id INTEGER NULL,
    reason TEXT,
    counts TEXT  -- JSON: {"moved_edges": N, "moved_aliases": N, "moved_mentions": N, "dropped_edges": N, "dropped_aliases": N}
);
CREATE INDEX IF NOT EXISTS idx_kg_merges_survivor ON kg_merges(survivor_id);
CREATE INDEX IF NOT EXISTS idx_kg_merges_merged ON kg_merges(merged_id);
```

### 2. Merge Transaction

New function `merge_entities(survivor_id, merged_id, reason, conn, created_by_node_id=None) -> dict` in a new file `episodic/kg/merge.py`.

The entire merge is one SQLite transaction. If any step fails, rollback completely.

#### Input validation
- survivor_id and merged_id must both exist in kg_entities
- survivor_id must NOT have merged_into_entity_id set (not itself tombstoned)
- merged_id must NOT have merged_into_entity_id set (not already merged)
- survivor_id != merged_id

#### Step 1: Rewrite kg_edges

For every edge where subj_entity_id = merged_id or obj_entity_id = merged_id:

1. Compute the new triple: (new_subj, predicate, new_obj) where merged_id is replaced with survivor_id
2. Check if this new triple already exists in kg_edges (via the UNIQUE index)
3. If NO conflict: UPDATE the edge in place (set subj_entity_id or obj_entity_id to survivor_id)
4. If YES conflict: keep the existing edge that has the newer source_node_id (via JOIN to kg_assertions on assertion_id). Delete the other. Deterministic: compare `kg_assertions.source_node_id` for both edges; the edge whose assertion has the higher source_node_id survives.

Track: moved_edges (updated without conflict), dropped_edges (deleted due to conflict).

SQL approach (per edge from merged_id):
```sql
-- For each edge E referencing merged_id:
-- Compute would-be new triple
-- Check: SELECT edge_id, assertion_id FROM kg_edges WHERE subj_entity_id=? AND predicate=? AND obj_entity_id=?
-- If exists (conflict):
--   Compare assertion source_node_ids, delete the one with lower source_node_id
--   If the survivor is the existing edge, delete E
--   If the survivor is E, delete the existing, then update E
-- If no conflict:
--   UPDATE kg_edges SET subj_entity_id=survivor_id WHERE edge_id=E.edge_id (or obj)
```

Implementation: fetch all edges for merged_id into a Python list, process each with conflict detection. This is a small N operation (typically < 20 edges per entity).

#### Step 2: Rewrite kg_entity_aliases

For every alias where entity_id = merged_id:

1. Compute new (entity_id=survivor_id, alias) pair
2. Check if UNIQUE(entity_id, alias) would conflict
3. If NO conflict: UPDATE entity_id to survivor_id
4. If YES conflict: keep the alias row with the lower source_node_id (earliest provenance). Delete the other.

Track: moved_aliases, dropped_aliases.

#### Step 3: Rewrite kg_mentions

For every mention where entity_id = merged_id:

1. Compute new (node_id, span_start, span_end) with entity_id=survivor_id
2. Check if UNIQUE(node_id, span_start, span_end) would conflict
3. If NO conflict: UPDATE entity_id to survivor_id
4. If YES conflict: keep the mention already bound to survivor_id. Delete the merged_id mention.

Track: moved_mentions (for the counts JSON; no dropped_mentions counter needed since the surviving mention already points to the right entity).

#### Step 4: Tombstone the merged entity

```sql
UPDATE kg_entities
SET merged_into_entity_id = ?, merged_at = ?, merged_reason = ?
WHERE entity_id = ?
```

#### Step 5: Write merge log

```sql
INSERT INTO kg_merges (survivor_id, merged_id, created_at, created_by_node_id, reason, counts)
VALUES (?, ?, ?, ?, ?, ?)
```

Where counts is JSON: `{"moved_edges": N, "moved_aliases": N, "moved_mentions": N, "dropped_edges": N, "dropped_aliases": N}`

#### Step 6: Invalidate MentionDictionary cache

After commit, bump the `high_water_mark` in kg_state (or use a separate `merge_epoch` counter) so the MentionDictionary rebuilds on next read-side call.

Simplest approach: use the existing HWM mechanism. Set a flag or increment a merge counter in kg_state:
```sql
INSERT OR REPLACE INTO kg_state (key, value) VALUES ('merge_epoch', CAST(? AS TEXT))
```
Where value = current time as string. MentionDictionary.rebuild() checks both HWM and merge_epoch.

#### Return value

```python
{
    'survivor_id': int,
    'merged_id': int,
    'moved_edges': int,
    'dropped_edges': int,
    'moved_aliases': int,
    'dropped_aliases': int,
    'moved_mentions': int,
}
```

### 3. Read-Side Changes

#### 3a. MentionDictionary.rebuild()

In `context_source.py`, the `rebuild()` method queries `kg_entities` + `kg_entity_aliases` to build the alias map.

Add WHERE filter: `WHERE merged_into_entity_id IS NULL` on the kg_entities query.

Also: on cache invalidation check, include merge_epoch alongside high_water_mark. If either has changed, rebuild.

```python
# In rebuild():
# Old: SELECT entity_id, canonical_name, ... FROM kg_entities
# New: SELECT entity_id, canonical_name, ... FROM kg_entities WHERE merged_into_entity_id IS NULL
```

#### 3b. Neighborhood retrieval

In `retrieve_neighborhood()` (context_source.py), edges already JOIN on kg_entities. Add:
- Filter: exclude edges where either subj or obj entity has merged_into_entity_id IS NOT NULL

This is a safety net — after a merge, no edges should reference tombstoned entities (they were rewritten in step 1). But if any were missed, this prevents them from surfacing.

```sql
-- Add to the JOIN:
AND subj_ent.merged_into_entity_id IS NULL
AND obj_ent.merged_into_entity_id IS NULL
```

### 4. CLI Command

Add `/kg merge <entity_id_1> <entity_id_2>` to `commands/kg.py`.

The lower entity_id becomes the survivor (older = canonical). User can override with `/kg merge <survivor_id> <merged_id> --survivor=<id>`.

Display: show what will be merged (both entity names, edge counts), require confirmation, then execute and print the counts dict.

### 5. Auto-Merge Detection (Optional, implement if straightforward)

After each batch rebuild or real-time extraction, scan for entity pairs where:
- Same canonical_key (should not happen if applicator resolves correctly, but does for the Cherry MX Brown case)
- Same canonical_name (case-insensitive) AND same entity_type

Log candidates to debug output. Do NOT auto-merge — require explicit `/kg merge` command.

Actually — simpler: add a `/kg dupes` command that runs:
```sql
SELECT e1.entity_id, e2.entity_id, e1.canonical_name, e1.entity_type
FROM kg_entities e1
JOIN kg_entities e2 ON e1.entity_id < e2.entity_id
  AND LOWER(e1.canonical_name) = LOWER(e2.canonical_name)
  AND e1.entity_type = e2.entity_type
WHERE e1.merged_into_entity_id IS NULL
  AND e2.merged_into_entity_id IS NULL
ORDER BY e1.canonical_name;
```

---

## B. Read-Side Regression Tests

Create `tests/kg/test_kg_readside.py`. All tests are deterministic, no LLM. Each seeds a temp SQLite with `ensure_kg_schema()`.

### Shared fixture

Seed entities and edges with varied predicates, recency (source_node_id), and tags. Use the full context_source pipeline (get_kg_context → format_kg_context), not internal helpers.

Entities: `<user>` (user:self), Alice (person), MIT (org), MacBook (artifact, alias "laptop"), Python (topic), Rust (topic), React (artifact), ML Lab (org).

Edges (with ascending source_node_ids for recency ranking):
- user:self → related_to → Alice (node 100)
- Alice → located_at → MIT (node 101)
- Alice → studies → Python (node 102)
- user:self → has → MacBook (node 200)
- MacBook → has → 64GB RAM (artifact, node 201)
- user:self → uses → Rust (node 300)
- user:self → uses → React (node 301)
- user:self → works_on → ML Lab (node 400)
- Alice → affiliated_with → ML Lab (node 401)
- user:self → wants → Python certification (topic, node 500)

Each edge needs a corresponding kg_assertions row with status='active'.

### RT1: Ranking and formatting

Input: "Tell me about Alice"

Assert:
1. get_kg_context() returns non-empty result
2. Edges are ordered by PREDICATE_PRIORITY (related_to before uses, has before located_at, etc.)
3. Within same predicate priority, higher source_node_id (more recent) ranks first
4. Output text contains "Alice" edges, formatted as "- subject predicate object [node:N]"

### RT2: Closure caps

Seed a dense neighborhood — user:self related_to 5 people, each located_at a different org (10 closure-eligible pairs for KINSHIP_LOCATION).

Set kg_max_derived=3.

Input: mention all 5 people by name.

Assert:
1. result.derived_count <= 3 (cap respected)
2. Budget not exceeded
3. Derived facts include provenance rule names

### RT3: Budget enforcement

Set kg_budget=60 (very small, ~15 tokens).

Seed 10+ edges for a mentioned entity.

Input: mention that entity.

Assert:
1. len(result.text) // 4 <= 60
2. Highest-priority edges are kept (check first line of output)
3. Lower-priority edges are dropped

### RT4: Merge tombstone exclusion (depends on Part A)

1. Seed two entities: "Cherry MX Brown switches" (id X) and "Cherry MX Brown switches" (id Y), both artifact type
2. Seed edges: MacBook → has → X, Keychron → has → Y
3. Call merge_entities(survivor_id=X, merged_id=Y, reason="duplicate")
4. Rebuild MentionDictionary
5. Call get_kg_context("Cherry MX Brown switches")

Assert:
1. Result contains edges referencing entity X only
2. Entity Y does not appear in output (tombstoned)
3. Keychron → has → X edge exists (rewritten from Y to X)
4. MentionDictionary does not contain entity Y

### RT5: Alias resolution post-merge

1. Seed entity "Neovim" (id A) with alias "vim", entity "Neovim" (id B) with alias "nvim"
2. Merge B into A
3. Rebuild MentionDictionary
4. Call detect_mentions("I use nvim daily")

Assert:
1. Resolves to entity A (survivor)
2. Alias "nvim" is now bound to entity A (moved from B)

---

## Implementation Order

1. Schema migration: `_migrate_entity_merge()` in schema.py (columns + kg_merges table)
2. `episodic/kg/merge.py`: merge_entities() transaction
3. Read-side filters: MentionDictionary rebuild + neighborhood retrieval tombstone exclusion
4. CLI: `/kg merge`, `/kg dupes`
5. Tests: RT1–RT5
6. Run full suite, report results
7. Run `/kg dupes` on live DB, report candidates
8. If Cherry MX Brown duplicate found, merge it and verify
