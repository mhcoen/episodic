# Spec: Real-Time KG Extraction (background thread)

## Goal
After each user turn is committed, spawn a background thread to run the single-node KG extraction pipeline. User sees zero latency impact. KG edges appear by the next turn.

## New file: `episodic/kg/realtime.py`

Two public functions:
1. `extract_node_async(user_node_uuid: str, user_text: str)` — config-gated entry point, spawns daemon thread
2. `_extract_single_node(user_node_uuid, user_text)` — the actual pipeline (private, runs in thread)

### Pipeline in `_extract_single_node`:

```
Step 0: Resolve UUID → rowid via SELECT rowid FROM nodes WHERE id = ?
Step 1: classify_node_intent(user_text) — if 'question', record empty patch, advance HWM, return
Step 2: extract_patch(node_id, lookback=3, conn=None) — LLM call, own connection
Step 3: json.loads → repair_patch → validate_patch → apply_patch — single connection
Step 4: _advance_hwm_if_contiguous(node_id, conn)
```

All imports from existing modules: batch.classify_node_intent, extractor.extract_patch, validator.validate_patch/repair_patch, applicator.apply_patch/record_rejected_patch. No new dependencies.

Error handling: entire function wrapped in try/except, errors go to `debug_print(category="kg")`. Never surfaces to user.

### `_advance_hwm_if_contiguous(node_id: int, conn)` — HWM strategy

- If `node_id == HWM+1`: advance (contiguous)
- If `node_id > HWM+1`: check if all intermediate user nodes have `kg_patches` rows. If yes, advance to node_id. If no, leave HWM (batch fills gap later).
- If `node_id <= HWM`: no-op (already processed)

This handles out-of-order completion when user sends rapid turns.

### `extract_node_async` details

```python
def extract_node_async(user_node_uuid: str, user_text: str) -> None:
    if not config.get("kg_realtime", False):
        return
    thread = threading.Thread(
        target=_extract_single_node,
        args=(user_node_uuid, user_text),
        daemon=True,
        name=f"kg-extract-{user_node_uuid[:8]}"
    )
    thread.start()
```

## Integration call site: conversation.py `handle_chat_message()`

Right after `user_node_id, user_short_id = insert_node(user_input, ...)`, BEFORE topic detection:

```python
# Real-time KG extraction (fire-and-forget, non-blocking)
if config.get("kg_realtime", False):
    from episodic.kg.realtime import extract_node_async
    extract_node_async(user_node_id, user_input)
```

Pattern matches existing `_fire_and_forget_index()`.

## Config

Add to config_defaults.py:
```python
"kg_realtime": False,   # Real-time KG extraction per user turn
```

## Idempotency

Before implementing, CHECK `applicator.py`: does `apply_patch` guard against duplicate node_id? `kg_patches` uses `INSERT OR REPLACE` so the patch record is safe. But verify entities and edges use `INSERT OR IGNORE` or equivalent. If not, add a guard at top of `_extract_single_node` step 3:

```python
existing = conn.execute(
    "SELECT 1 FROM kg_patches WHERE node_id = ? AND applied = 1", (node_id,)
).fetchone()
if existing:
    _advance_hwm_if_contiguous(node_id, conn)
    return  # Already processed (e.g., by batch)
```

## Question skip handling

For QA nodes, use `record_rejected_patch` (not `apply_patch`) with `rejection_reason='qa_node_realtime'`. This records the node as processed so batch skips it.

## Threading details

- `daemon=True` — dies with main process
- Thread name: `f"kg-extract-{user_node_uuid[:8]}"` for debuggability
- Each thread uses `get_connection()` which returns a fresh connection (WAL mode)
- No thread pool — one thread per turn, at conversational pace (~30s intervals) this is fine

## What NOT to change

- `batch.py` — batch works unchanged, skips nodes with existing `kg_patches`
- `context_source.py` — reads edges regardless of creation method
- `extractor.py`, `validator.py`, `applicator.py` — no changes

## Tests (in `tests/kg/test_kg_realtime.py`)

1. `test_realtime_basic`: Insert user node, call `_extract_single_node` synchronously, verify edges
2. `test_realtime_question_skip`: Question → no LLM call, qa patch recorded
3. `test_realtime_hwm_contiguous`: HWM+1 → advances
4. `test_realtime_hwm_gap`: HWM+3 → stays
5. `test_realtime_config_gate`: `kg_realtime=False` → no-op
6. `test_realtime_idempotent_with_batch`: Realtime then batch → no duplication

For tests, mock `extract_patch` to avoid actual LLM calls (return a canned `patch_json`). Test `_extract_single_node` synchronously (don't test threading itself).

## Logging

- Success: `debug_print(f"KG realtime: node {node_id} → {entity_count} entities, {edge_count} edges", category="kg")`
- Skip: `debug_print(f"KG realtime: node {node_id} classified as question, skipped", category="kg")`
- Error: `debug_print(f"KG realtime error for {uuid[:8]}: {e}", category="kg")`
