"""Batch processor for KG extraction pipeline."""

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

from .db_kg import _use_conn
from .schema import ensure_kg_schema
from .prompt_template import normalize_text
from .extractor import extract_patch
from .validator import validate_patch, repair_patch, VALIDATOR_VERSION
from .applicator import apply_patch, record_rejected_patch

EXTRACTION_CONCURRENCY = 15


# --- Node intent classification for question filtering ---

_INTERROGATIVES = frozenset({
    'what', 'how', 'why', 'when', 'where', 'who', 'whom', 'which',
    'can', 'could', 'should', 'would', 'do', 'does', 'is', 'are',
    'will', 'was', 'were', 'has', 'have', 'did',
})

_DURABLE_INTENT_MARKERS = (
    'i want to', 'i need to', "i'm trying to", 'my goal is',
    'i plan to', "i'm looking for", 'i decided to',
    "i'd like to", "i'm hoping to", 'i would like to',
    'i want a', 'i need a', "i'm looking for a",
)


def classify_node_intent(content: str) -> str:
    """Classify a user turn as 'question' or 'assertion'.

    A turn is 'question' if it ends with '?' or starts with an
    interrogative word, AND contains no first-person durable intent
    markers.  Returns 'question' or 'assertion'.
    """
    text = content.strip()
    if not text:
        return 'assertion'

    text_lower = text.lower()

    # Durable intent markers override question classification
    for marker in _DURABLE_INTENT_MARKERS:
        if marker in text_lower:
            return 'assertion'

    # Question indicators
    if text.rstrip().endswith('?'):
        return 'question'

    first_word = text_lower.split()[0] if text_lower.split() else ''
    if first_word in _INTERROGATIVES:
        return 'question'

    return 'assertion'


def get_high_water_mark(conn) -> int:
    """Read high_water_mark from kg_state. Returns 0 if not set."""
    try:
        row = conn.execute(
            "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
        ).fetchone()
        return int(row[0]) if row else 0
    except (sqlite3.OperationalError, TypeError, ValueError):
        return 0


def get_skip_list(conn) -> set[int]:
    """Read all node_ids from kg_skiplist."""
    try:
        rows = conn.execute("SELECT node_id FROM kg_skiplist").fetchall()
        return {row[0] for row in rows}
    except sqlite3.OperationalError:
        return set()


def get_pending_nodes(
    hwm: int,
    skip_list: set[int],
    conn,
) -> list[dict]:
    """Fetch all user-role nodes with rowid > hwm, ordered by rowid.

    Exclude node_ids in skip_list. Filter to role='user' (Phase 0).
    Returns list of dicts with: node_id (rowid), content, role.
    """
    try:
        rows = conn.execute(
            "SELECT rowid, content, role FROM nodes "
            "WHERE rowid > ? AND role = 'user' "
            "AND (is_meta_query = 0 OR is_meta_query IS NULL) "
            "ORDER BY rowid",
            (hwm,)
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback: try node_id column (test fixtures)
        try:
            rows = conn.execute(
                "SELECT node_id, content, role FROM nodes "
                "WHERE node_id > ? AND role = 'user' "
                "ORDER BY node_id",
                (hwm,)
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    result = []
    for row in rows:
        nid = row[0]
        if nid in skip_list:
            continue
        content = row[1]
        if not content or not content.strip():
            continue
        result.append({
            'node_id': nid,
            'content': content,
            'role': row[2],
        })
    return result


def _get_topic_entity_ids(node_id: int, conn) -> set[int]:
    """Get entity_ids in the same topic scope as node_id."""
    topic_entity_ids = set()
    try:
        # Get the node's UUID from rowid
        row = conn.execute(
            "SELECT id FROM nodes WHERE rowid = ?", (node_id,)
        ).fetchone()
        if not row:
            return topic_entity_ids
        node_uuid = row[0]

        # Find which topic contains this node
        topic_row = conn.execute(
            "SELECT topic_start_node_id FROM topic_nodes "
            "WHERE node_id = ? LIMIT 1",
            (node_uuid,)
        ).fetchone()
        if not topic_row:
            return topic_entity_ids

        # Get all rowids in this topic
        topic_node_rows = conn.execute(
            "SELECT turn_idx FROM topic_nodes "
            "WHERE topic_start_node_id = ?",
            (topic_row[0],)
        ).fetchall()
        topic_rowids = {r[0] for r in topic_node_rows}

        if topic_rowids:
            placeholders = ','.join('?' * len(topic_rowids))
            ent_rows = conn.execute(
                f"SELECT entity_id FROM kg_entities "
                f"WHERE created_node_id IN ({placeholders})",
                list(topic_rowids)
            ).fetchall()
            topic_entity_ids = {r[0] for r in ent_rows}
    except sqlite3.OperationalError:
        pass
    return topic_entity_ids


def _get_existing_canonical_keys(conn) -> dict[str, int]:
    """Get all non-null canonical_keys mapped to entity_ids."""
    try:
        rows = conn.execute(
            "SELECT canonical_key, entity_id FROM kg_entities "
            "WHERE canonical_key IS NOT NULL"
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    except sqlite3.OperationalError:
        return {}


def _get_entity_dictionary(conn) -> list[dict]:
    """Build entity dictionary for domain/range validation."""
    try:
        rows = conn.execute(
            "SELECT entity_id, entity_type FROM kg_entities"
        ).fetchall()
        return [{'entity_id': r[0], 'entity_type': r[1]} for r in rows]
    except sqlite3.OperationalError:
        return []


def run_batch(
    lookback: int = 3,
    max_nodes: int | None = None,
    conn=None,
    progress_callback: Optional[Callable] = None,
    dry_run: bool = False,
    db_path: str | None = None,
) -> dict:
    """Run the extraction batch from the current high-water mark.

    Extraction is parallelized (ThreadPoolExecutor, concurrency=5).
    Each extraction thread uses its own DB connection (conn=None).
    Validation and application are sequential in node_id order,
    since entity resolution depends on prior nodes.

    Parameters:
    - lookback: number of preceding turns for extraction context
    - max_nodes: if set, process at most this many nodes per run
    - conn: DB connection (used for validate/apply, not extraction)
    - progress_callback: optional callable(node_id, index, total)
    - dry_run: if True, extract and validate but do not apply
    - db_path: if set, extraction threads open connections to this DB
      instead of the production DB (used by eval harness)
    """
    with _use_conn(conn) as c:
        ensure_kg_schema(c)

        hwm_before = get_high_water_mark(c)
        skip_list = get_skip_list(c)
        pending = get_pending_nodes(hwm_before, skip_list, c)

        if max_nodes:
            pending = pending[:max_nodes]

        summary = {
            'nodes_processed': 0,
            'patches_applied': 0,
            'patches_rejected': 0,
            'errors': [],
            'hwm_before': hwm_before,
            'hwm_after': hwm_before,
            'node_classifications': {'question': 0, 'assertion': 0},
            'nodes_qa_filtered': 0,
            'edges_stripped_question_filter': 0,
        }

        if not pending:
            return summary

        total = len(pending)

        # Phase 1: Classify nodes, skip LLM for QA nodes
        extraction_results = {}  # node_id -> result dict
        nodes_to_extract = []
        for node in pending:
            node_id = node['node_id']
            if classify_node_intent(node['content']) == 'question':
                qa_json = json.dumps({
                    'schema_version': 'kg_patch_v1',
                    'node_id': node_id,
                    'assertions': [],
                    'entities': [],
                    'aliases': [],
                    'mentions': [],
                    'edges': [],
                    'notes': 'qa_node_skipped',
                }, separators=(',', ':'))
                extraction_results[node_id] = {
                    'node_id': node_id,
                    'patch_json': qa_json,
                    'patch_hash': hashlib.sha256(
                        qa_json.encode()
                    ).hexdigest(),
                    'applied': 0,
                    'rejection_reason': None,
                    'model_id': 'skipped_qa',
                    'extraction_time_ms': 0,
                    'raw_output': '',
                }
                summary['nodes_qa_filtered'] += 1
            else:
                nodes_to_extract.append(node)

        # Phase 1b: Parallel extraction (each thread opens its own conn)
        def _extract_with_conn(node_id, lb, path):
            if path:
                ec = sqlite3.connect(path)
                ec.execute("PRAGMA foreign_keys=ON")
            else:
                ec = None
            try:
                return extract_patch(node_id, lb, conn=ec)
            finally:
                if ec:
                    ec.close()

        with ThreadPoolExecutor(max_workers=EXTRACTION_CONCURRENCY) as pool:
            future_to_node = {
                pool.submit(_extract_with_conn, node['node_id'], lookback, db_path): node
                for node in nodes_to_extract
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                node_id = node['node_id']
                try:
                    extraction_results[node_id] = future.result()
                except Exception as e:
                    extraction_results[node_id] = {
                        'node_id': node_id,
                        'patch_json': None,
                        'patch_hash': None,
                        'applied': 0,
                        'rejection_reason': f'thread_error: {e}',
                        'model_id': 'unknown',
                        'extraction_time_ms': 0,
                        'raw_output': '',
                    }

        # Phase 2: Sequential validate + apply (in node_id order)
        for i, node in enumerate(pending):
            node_id = node['node_id']
            source_text = normalize_text(node['content'])
            node_class = classify_node_intent(node['content'])
            summary['node_classifications'][node_class] += 1
            summary['nodes_processed'] += 1

            if progress_callback:
                progress_callback(node_id, i + 1, total)

            result = extraction_results[node_id]

            # Check extraction failure
            if result['patch_json'] is None:
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=None,
                    patch_hash=None,
                    rejection_reason=result.get('rejection_reason', 'unknown'),
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': result.get('rejection_reason', 'unknown'),
                })
                c.execute(
                    "UPDATE kg_state SET value = ? "
                    "WHERE key = 'high_water_mark'",
                    (str(node_id),)
                )
                c.commit()
                continue

            # Parse patch
            try:
                patch = json.loads(result['patch_json'])
            except (json.JSONDecodeError, TypeError):
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason='patch_json_parse_error',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': 'patch_json_parse_error',
                })
                continue

            # Repair common LLM errors before validation
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
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': f'repair_failed: {e}',
                })
                c.execute(
                    "UPDATE kg_state SET value = ? "
                    "WHERE key = 'high_water_mark'",
                    (str(node_id),)
                )
                c.commit()
                continue

            # Validate
            topic_eids = _get_topic_entity_ids(node_id, c)
            canonical_keys = _get_existing_canonical_keys(c)
            ent_dict = _get_entity_dictionary(c)

            try:
                vresult = validate_patch(
                    patch=patch,
                    source_text=source_text,
                    node_id=node_id,
                    topic_entity_ids=topic_eids,
                    existing_canonical_keys=canonical_keys,
                    conn=c,
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
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': f'validation_error: {e}',
                })
                c.execute(
                    "UPDATE kg_state SET value = ? "
                    "WHERE key = 'high_water_mark'",
                    (str(node_id),)
                )
                c.commit()
                continue

            if not vresult.valid:
                reason = '; '.join(vresult.errors[:5])
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason=f'validation: {reason}',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': f'validation: {reason}',
                })
                c.execute(
                    "UPDATE kg_state SET value = ? "
                    "WHERE key = 'high_water_mark'",
                    (str(node_id),)
                )
                c.commit()
                continue

            if dry_run:
                summary['patches_applied'] += 1
                continue

            # Apply
            try:
                apply_patch(
                    patch=patch,
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=c,
                )
                summary['patches_applied'] += 1
                summary['hwm_after'] = node_id
            except Exception as e:
                # Transaction failure — stop processing
                record_rejected_patch(
                    node_id=node_id,
                    patch_json=result['patch_json'],
                    patch_hash=result['patch_hash'],
                    rejection_reason=f'apply_failed: {e}',
                    model_id=result['model_id'],
                    extraction_time_ms=result['extraction_time_ms'],
                    conn=c,
                )
                summary['patches_rejected'] += 1
                summary['errors'].append({
                    'node_id': node_id,
                    'reason': f'apply_failed: {e}',
                })
                break  # Stop on transaction failure

        # Update final HWM
        summary['hwm_after'] = get_high_water_mark(c)
        return summary


def run_rebuild(
    lookback: int = 3,
    conn=None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Full rebuild: drop all KG data and reprocess from node 0."""
    with _use_conn(conn) as c:
        # Delete in FK-safe order
        for table in [
            'kg_edges', 'kg_mentions', 'kg_entity_aliases',
            'kg_assertions', 'kg_patches', 'kg_curations', 'kg_skiplist',
        ]:
            try:
                c.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass

        # Delete all entities except user:self
        try:
            c.execute(
                "DELETE FROM kg_entities WHERE canonical_key != 'user:self'"
            )
        except sqlite3.OperationalError:
            pass

        # Reset HWM
        try:
            c.execute(
                "UPDATE kg_state SET value = '0' "
                "WHERE key = 'high_water_mark'"
            )
        except sqlite3.OperationalError:
            pass

        c.commit()

        # Re-run batch with no limit
        return run_batch(
            lookback=lookback,
            conn=c,
            progress_callback=progress_callback,
        )


def add_to_skiplist(
    node_id: int,
    reason: str = '',
    conn=None,
):
    """Add a node to the skip list and advance HWM if stuck."""
    with _use_conn(conn) as c:
        c.execute(
            "INSERT OR IGNORE INTO kg_skiplist "
            "(node_id, reason, created_at) VALUES (?, ?, ?)",
            (node_id, reason, time.time())
        )

        # If this node has a patch, mark it rejected
        c.execute(
            "UPDATE kg_patches SET applied = 0, "
            "rejection_reason = 'skipped_by_user' "
            "WHERE node_id = ?",
            (node_id,)
        )

        # If HWM is stuck at or before this node, advance past
        hwm = get_high_water_mark(c)
        if hwm <= node_id:
            c.execute(
                "UPDATE kg_state SET value = ? "
                "WHERE key = 'high_water_mark'",
                (str(node_id),)
            )

        c.commit()
