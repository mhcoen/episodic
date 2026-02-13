"""Real-time KG extraction — background thread per user turn."""

import json
import hashlib
import threading

from episodic.debug_utils import debug_print


def extract_node_async(
    node_uuid: str,
    content: str,
    source_type: str = "user_input",
    source_id: str = "",
) -> None:
    """Fire-and-forget KG extraction for a single user node.

    Spawns a daemon thread. Follows the same pattern as
    _fire_and_forget_index() in conversation.py.

    Args:
        node_uuid: The UUID string returned by insert_node()
        content: The user message text
        source_type: Source type for L3 source gate check
        source_id: Optional source identifier (e.g., client_id)
    """
    from episodic.config import config
    if not config.get('kg_realtime', False):
        return

    from episodic.mcp.security.source_gate import (
        check_extraction_allowed,
        ExtractionPolicy,
    )

    gate = check_extraction_allowed(source_type, source_id)
    if gate.policy == ExtractionPolicy.BLOCK:
        debug_print(
            f"KG realtime: blocked for source_type={source_type}",
            category="kg",
        )
        return

    thread = threading.Thread(
        target=_extract_single_node,
        args=(node_uuid, content, gate),
        daemon=True,
        name=f"kg-extract-{node_uuid[:8]}",
    )
    thread.start()


def _extract_single_node(node_uuid: str, content: str, gate=None) -> None:
    """Run the full extraction pipeline for one node.

    Gets its own DB connection (SQLite WAL mode is fine for this).
    Errors are logged but never propagate to the main thread.

    Args:
        gate: SourceGateResult from check_extraction_allowed(). If
              gate.policy is QUARANTINE, assertions get quarantined.
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
                    rejection_reason='qa_node_realtime',
                    model_id='skipped_qa',
                    extraction_time_ms=0,
                    conn=conn,
                )
                _advance_hwm_if_contiguous(conn, node_id)
                debug_print(
                    f"KG realtime: node {node_id} classified as question, skipped",
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
                _advance_hwm_if_contiguous(conn, node_id)
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
                _advance_hwm_if_contiguous(conn, node_id)
            return

        # Step 4: Repair + validate + apply (sequential, needs conn)
        source_text = normalize_text(content)

        with _use_conn(None) as conn:
            # Idempotency guard: skip if already processed (e.g., by batch)
            existing = conn.execute(
                "SELECT 1 FROM kg_patches WHERE node_id = ? AND applied = 1",
                (node_id,)
            ).fetchone()
            if existing:
                _advance_hwm_if_contiguous(conn, node_id)
                return

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
                _advance_hwm_if_contiguous(conn, node_id)
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
                _advance_hwm_if_contiguous(conn, node_id)
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
                _advance_hwm_if_contiguous(conn, node_id)
                return

            # Apply
            from episodic.mcp.security.source_gate import ExtractionPolicy

            is_quarantine = (
                gate is not None
                and gate.policy == ExtractionPolicy.QUARANTINE
            )
            apply_patch(
                patch=patch,
                node_id=node_id,
                patch_json=result['patch_json'],
                patch_hash=result['patch_hash'],
                model_id=result['model_id'],
                extraction_time_ms=result['extraction_time_ms'],
                conn=conn,
                quarantine=is_quarantine,
                source_origin=gate.source_origin if gate else "",
            )
            _advance_hwm_if_contiguous(conn, node_id)

            edge_count = len(patch.get('edges', []))
            entity_count = len(patch.get('entities', []))
            debug_print(
                f"KG realtime: node {node_id} -> "
                f"{entity_count} entities, {edge_count} edges",
                category="kg"
            )

    except Exception as e:
        debug_print(f"KG realtime error for {node_uuid[:8]}: {e}", category="kg")


def _advance_hwm_if_contiguous(conn, node_id: int) -> None:
    """Advance high_water_mark if node_id forms a contiguous sequence.

    - node_id == HWM+1: advance directly (contiguous)
    - node_id > HWM+1: check if all intermediate user nodes have kg_patches
      rows. If yes, advance to node_id. If no, leave HWM for batch to fill.
    - node_id <= HWM: no-op (already processed)
    """
    try:
        row = conn.execute(
            "SELECT CAST(value AS INTEGER) FROM kg_state "
            "WHERE key = 'high_water_mark'"
        ).fetchone()
        hwm = row[0] if row else 0

        if node_id <= hwm:
            return  # Already past this node

        if node_id == hwm + 1:
            # Contiguous — advance directly
            conn.execute(
                "UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'",
                (str(node_id),)
            )
            conn.commit()
            return

        # node_id > hwm + 1 — check if all intermediate user nodes are processed
        gap_rows = conn.execute(
            "SELECT n.rowid FROM nodes n "
            "WHERE n.rowid > ? AND n.rowid < ? AND n.role = 'user' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM kg_patches p WHERE p.node_id = n.rowid"
            ")",
            (hwm, node_id)
        ).fetchall()

        if not gap_rows:
            # All intermediate nodes processed — advance to node_id
            conn.execute(
                "UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'",
                (str(node_id),)
            )
            conn.commit()
        # else: gap exists, leave HWM for batch to fill
    except Exception:
        pass  # Non-critical — batch will catch up
