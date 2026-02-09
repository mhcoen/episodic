# Spec: Real-time KG Extraction (`kg/realtime.py`)

## Summary
After each user turn is committed to the DB, spawn a background thread that runs the single-node KG extraction pipeline. This makes the KG available for `get_kg_context()` on the *next* turn instead of requiring a batch run.

## New file: `episodic/kg/realtime.py`

```python
"""Real-time KG extraction — background thread per user turn."""

import json
import hashlib
import threading

from episodic.debug_utils import debug_print


def extract_node_async(node_uuid: str, content: str) -> None:
    """Fire-and-forget KG extraction for a single user node.
    
    Spawns a daemon thread. Follows the same pattern as
    _fire_and_forget_index() in conversation.py.
    
    Args:
        node_uuid: The UUID string returned by insert_node()
        content: The user message text
    """
    from episodic.config import config
    if not config.get('kg_realtime', False):
        return

    thread = threading.Thread(
        target=_extract_single_node,
        args=(node_uuid, content),
        daemon=True,
        name=f"kg-extract-{node_uuid[:8]}",
    )
    thread.start()


def _extract_single_node(node_uuid: str, content: str) -> None:
    """Run the full extraction pipeline for one node.
    
    Gets its own DB connection (SQLite WAL mode is fine for this).
    Errors are logged but never propagate to the main thread.
    """
    try:
        from episodic.kg.db_kg import _use_conn
        from episodic.kg.schema import ensure_kg_schema
        from episodic.kg.batch import (
            classify_node_intent,
            _get_topic_entity_ids,
            _get_existing_canonical_keys,
            _get_entity_dictionary,
        )
        from episodic.kg.prompt_template import normalize_text
        from episodic.kg.extractor import extract_patch
        from episodic.kg.validator import validate_patch, repair_patch
        from episodic.kg.applicator import apply_patch, record_rejected_patch

        # Step 0: Resolve UUID -> rowid (extract_patch needs integer rowid)
        with _use_conn(None) as conn:
            ensure_kg_schema(conn)
            
            row = conn.execute(
                "SELECT rowid FROM nodes WHERE id = ?", (node_uuid,)
            ).fetchone()
            if not row:
                debug_print(
                    f"KG realtime: node {node_uuid[:8]} not found", 
                    category="kg"
                )
                return
            node_id = row[0]  # integer rowid

            # Step 1: QA classification (deterministic, no LLM)
            intent = classify_node_intent(content)
            if intent == 'question':
                # Record empty patch, advance HWM
                qa_json = json.dumps({
                    'schema_version': 'kg_patch_v1',
                    'node_id': node_id,
                    'assertions': [], 'entities': [], 'aliases': [],
                    'mentions': [], 'edges': [],
                    'notes': 'qa_node_skipped',
                }, separators=(',', ':'))
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=qa_json,
                    patch_hash=hashlib.sha256(qa_json.encode()).hexdigest(),
                    rejection_reason='qa_node_skipped',
                    model_id='skipped_qa',
                    extraction_time_ms=0,
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
                debug_print(
                    f"KG realtime: node {node_id} skipped (question)",
                    category="kg"
                )
                return

        # Step 2: LLM extraction (uses its own connection internally)
        result = extract_patch(node_id, lookback=3, conn=None)

        if result['patch_json'] is None:
            with _use_conn(None) as conn:
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=None,
                    patch_hash=None,
                    rejection_reason=result.get('rejection_reason', 'unknown'),
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
            debug_print(
                f"KG realtime: node {node_id} extraction failed: "
                f"{result.get('rejection_reason')}",
                category="kg"
            )
            return

        # Step 3: Parse patch
        try:
            patch = json.loads(result['patch_json'])
        except (json.JSONDecodeError, TypeError):
            with _use_conn(None) as conn:
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason='patch_json_parse_error',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
            return

        # Step 4: Repair + validate + apply (sequential, needs conn)
        source_text = normalize_text(content)
        
        with _use_conn(None) as conn:
            # Repair
            try:
                repair_patch(patch, source_text)
            except Exception as e:
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason=f'repair_failed: {e}',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
                return

            # Validate
            topic_eids = _get_topic_entity_ids(node_id, conn)
            canonical_keys = _get_existing_canonical_keys(conn)
            ent_dict = _get_entity_dictionary(conn)

            try:
                vresult = validate_patch(
                    patch=patch,
                    source_text=source_text,
                    node_id=node_id,
                    topic_entity_ids=topic_eids,
                    existing_canonical_keys=canonical_keys,
                    conn=conn,
                    entity_dictionary=ent_dict,
                )
            except Exception as e:
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason=f'validation_error: {e}',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
                return

            if not vresult.valid:
                reason = '; '.join(vresult.errors[:5])
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason=f'validation: {reason}',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=conn,
                )
                _advance_hwm_if_ahead(conn, node_id)
                return

            # Apply
            apply_patch(
                patch=patch,
                node_id=node_id,
                patch_json=result['patch_json'],
                patch_hash=result['patch_hash'],
                model_id=result['model_id'],
                extraction_time_ms=result['extraction_time_ms'],
                conn=conn,
            )
            _advance_hwm_if_ahead(conn, node_id)

            debug_print(
                f"KG realtime: node {node_id} extracted "
                f"({len(patch.get('edges', []))} edges)",
                category="kg"
            )

    except Exception as e:
        debug_print(f"KG realtime error: {e}", category="kg")


def _advance_hwm_if_ahead(conn, node_id: int) -> None:
    """Advance high_water_mark to node_id if node_id > current HWM.
    
    Conditional advance — if another thread or batch run already
    advanced past this node, we don't regress.
    """
    try:
        conn.execute(
            "UPDATE kg_state SET value = ? "
            "WHERE key = 'high_water_mark' AND CAST(value AS INTEGER) < ?",
            (str(node_id), node_id)
        )
        conn.commit()
    except Exception:
        pass  # Non-critical — batch will catch up
```

## Call site: `conversation.py`

Add the call in `handle_chat_message()`, immediately after `insert_node` for the user node. This happens unconditionally before the `skip_llm_response` branch.

Find this block (around line ~395):
```python
            # Add the user message to the database
            with benchmark_resource("Database", "insert user node"):
                user_node_id, user_short_id = insert_node(user_input, self.current_node_id, role="user")
```

Add immediately after:
```python
            # Fire-and-forget KG extraction for this user turn
            from episodic.kg.realtime import extract_node_async
            extract_node_async(user_node_id, user_input)
```

That's it for the call site. `extract_node_async` checks `config.get('kg_realtime')` internally and returns immediately if disabled.

## Config

Add default:
- `kg_realtime`: `False` — user enables with `/set kg-realtime true`

Add kebab alias `kg-realtime` -> `kg_realtime` in param_mappings (follow the pattern of kg-context, kg-budget, etc.).

## Design rationale

**Why conditional HWM advance**: If the user sends two turns fast, node N+1's thread might finish before node N's. `_advance_hwm_if_ahead` uses `CAST(value AS INTEGER) < ?` so it never regresses HWM. If node N finishes later and N < current HWM, the UPDATE is a no-op. The next `run_batch` will skip already-processed nodes (they have patches in `kg_patches`).

**Why separate connections**: `extract_patch` opens its own connection for the read (context building). The validate/apply phase opens another. Both are short-lived. SQLite WAL mode handles concurrent readers + one writer fine at conversational pace.

**Why not just call run_batch(max_nodes=1)**: `run_batch` reads pending nodes from HWM, which might include stale unprocessed nodes from before the current session. We only want to process *this specific node*.

**Interaction with batch runs**: Safe. `kg_patches` has `node_id` as primary key with `INSERT OR REPLACE`. If batch processes the same node first, the thread's write overwrites harmlessly. If the thread processes it first, batch skips it (HWM already past it).

## CLAUDE.md update
Add to the end:
```
## Real-time Extraction (kg/realtime.py)
- Config: `kg_realtime` (kg-realtime): Enable per-turn background extraction (default: false)
- Spawns daemon thread per user turn, runs full pipeline (classify → extract → validate → apply)
- HWM advanced conditionally (never regresses)
- Errors logged to debug category "kg", never propagate to main thread
- Safe to run alongside batch: kg_patches has node_id PK, no duplication
```

## No new tests required
The pipeline functions are already tested. The only new code is threading glue + HWM advancement. Can add a test for `_advance_hwm_if_ahead` if desired, but it's 5 lines of SQL.
