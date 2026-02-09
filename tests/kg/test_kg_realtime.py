"""Tests for episodic.kg.realtime — real-time KG extraction.

All tests call _extract_single_node synchronously (no threading).
extract_patch is mocked to avoid LLM calls.
"""

import json
import sqlite3
import time
import uuid

import pytest

from episodic.kg.schema import ensure_kg_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db(conn):
    """Set up a minimal DB with nodes table + KG schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user',
            parent_id TEXT
        )
    """)
    ensure_kg_schema(conn)
    conn.commit()


def _insert_user_node(conn, content="I use Neovim for coding"):
    """Insert a user node. Returns (uuid, rowid)."""
    node_uuid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO nodes (id, content, role, parent_id) VALUES (?, ?, 'user', NULL)",
        (node_uuid, content),
    )
    rowid = conn.execute("SELECT rowid FROM nodes WHERE id = ?", (node_uuid,)).fetchone()[0]
    conn.commit()
    return node_uuid, rowid


def _get_hwm(conn):
    """Read current HWM."""
    row = conn.execute(
        "SELECT CAST(value AS INTEGER) FROM kg_state WHERE key = 'high_water_mark'"
    ).fetchone()
    return row[0] if row else 0


def _set_hwm(conn, value):
    """Set HWM to a specific value."""
    conn.execute(
        "UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'",
        (str(value),),
    )
    conn.commit()


def _make_canned_patch(node_id):
    """Return a valid canned patch_json and result dict."""
    patch = {
        'schema_version': 'kg_patch_v1',
        'node_id': node_id,
        'assertions': [{
            'span_start': 0, 'span_end': 20,
            'asserted_by': 'user', 'polarity': 'affirm',
            'certainty': 'explicit', 'tags': [],
        }],
        'entities': [{
            'entity_type': 'artifact',
            'canonical_name': 'Neovim',
            'canonical_key': None,
        }],
        'aliases': [],
        'mentions': [{
            'entity_ref': 0, 'surface_form': 'Neovim',
            'span_start': 6, 'span_end': 12,
        }],
        'edges': [{
            'subject_entity_ref': 'user:self',
            'predicate': 'uses',
            'object_entity_ref': 0,
            'assertion_ref': 0,
        }],
    }
    patch_json = json.dumps(patch)
    import hashlib
    patch_hash = hashlib.sha256(patch_json.encode()).hexdigest()
    return patch, patch_json, patch_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rt_db(tmp_path):
    """In-memory SQLite with nodes + KG schema."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    _create_test_db(conn)
    yield conn, db_path
    conn.close()


class _MockConfig:
    def __init__(self, overrides=None):
        self._data = {'kg_realtime': True, 'debug': False}
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None):
        return self._data.get(key, default)


# ---------------------------------------------------------------------------
# T1: test_realtime_config_gate
# ---------------------------------------------------------------------------

def test_realtime_config_gate(monkeypatch):
    """kg_realtime=False → extract_node_async is a no-op (no thread spawned)."""
    import threading
    from episodic.kg.realtime import extract_node_async

    monkeypatch.setattr('episodic.config.config', _MockConfig({'kg_realtime': False}))

    initial_threads = threading.active_count()
    extract_node_async("fake-uuid", "some text")
    # No thread spawned
    assert threading.active_count() <= initial_threads


# ---------------------------------------------------------------------------
# T2: test_realtime_question_skip
# ---------------------------------------------------------------------------

def test_realtime_question_skip(rt_db, monkeypatch):
    """Question node → no LLM call, qa patch recorded with 'qa_node_realtime'."""
    conn, db_path = rt_db
    # config not needed — _extract_single_node doesn't check config

    node_uuid, node_id = _insert_user_node(conn, "What is Python?")
    _set_hwm(conn, node_id - 1)

    # Mock _use_conn to return our test connection
    from contextlib import contextmanager

    @contextmanager
    def mock_use_conn(c=None):
        yield conn

    monkeypatch.setattr('episodic.kg.db_kg._use_conn', mock_use_conn)
    # Also patch the applicator's _use_conn for record_rejected_patch
    monkeypatch.setattr('episodic.kg.applicator._use_conn', mock_use_conn)

    from episodic.kg.realtime import _extract_single_node
    _extract_single_node(node_uuid, "What is Python?")

    # Check patch was recorded
    row = conn.execute(
        "SELECT rejection_reason FROM kg_patches WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == 'qa_node_realtime'

    # HWM should advance (contiguous: node_id == hwm+1)
    assert _get_hwm(conn) == node_id


# ---------------------------------------------------------------------------
# T3: test_realtime_hwm_contiguous
# ---------------------------------------------------------------------------

def test_realtime_hwm_contiguous(rt_db, monkeypatch):
    """HWM+1 → advances HWM."""
    conn, db_path = rt_db

    node_uuid, node_id = _insert_user_node(conn, "What time is it?")
    _set_hwm(conn, node_id - 1)

    from episodic.kg.realtime import _advance_hwm_if_contiguous
    _advance_hwm_if_contiguous(conn, node_id)

    assert _get_hwm(conn) == node_id


# ---------------------------------------------------------------------------
# T4: test_realtime_hwm_gap
# ---------------------------------------------------------------------------

def test_realtime_hwm_gap(rt_db, monkeypatch):
    """HWM+3 with unprocessed intermediate nodes → HWM stays."""
    conn, db_path = rt_db

    # Insert 3 nodes: n1, n2, n3
    _, n1 = _insert_user_node(conn, "node 1")
    _, n2 = _insert_user_node(conn, "node 2")
    _, n3 = _insert_user_node(conn, "node 3")

    # Set HWM to before n1
    _set_hwm(conn, n1 - 1)

    # Try to advance to n3 — n1 and n2 have no patches, so HWM should stay
    from episodic.kg.realtime import _advance_hwm_if_contiguous
    _advance_hwm_if_contiguous(conn, n3)

    assert _get_hwm(conn) == n1 - 1  # unchanged

    # Now record patches for n1 and n2
    for nid in (n1, n2):
        conn.execute(
            "INSERT OR REPLACE INTO kg_patches "
            "(node_id, patch_json, patch_hash, validator_version, applied, "
            "rejection_reason, model_id, extraction_time_ms) "
            "VALUES (?, '{}', 'hash', 'v1', 1, NULL, 'test', 0)",
            (nid,),
        )
    conn.commit()

    # Now advance to n3 — all intermediate nodes processed
    _advance_hwm_if_contiguous(conn, n3)
    assert _get_hwm(conn) == n3


# ---------------------------------------------------------------------------
# T5: test_realtime_basic (mock extract_patch)
# ---------------------------------------------------------------------------

def test_realtime_basic(rt_db, monkeypatch):
    """Full pipeline with mocked extract_patch → edges created."""
    conn, db_path = rt_db
    # config not needed — _extract_single_node doesn't check config

    node_uuid, node_id = _insert_user_node(conn, "I use Neovim for coding")
    _set_hwm(conn, node_id - 1)

    patch, patch_json, patch_hash = _make_canned_patch(node_id)

    from contextlib import contextmanager

    @contextmanager
    def mock_use_conn(c=None):
        yield conn

    monkeypatch.setattr('episodic.kg.db_kg._use_conn', mock_use_conn)
    monkeypatch.setattr('episodic.kg.applicator._use_conn', mock_use_conn)

    # Mock extract_patch to return our canned result
    def mock_extract_patch(nid, lookback=3, conn=None):
        return {
            'node_id': nid,
            'patch_json': patch_json,
            'patch_hash': patch_hash,
            'applied': 0,
            'rejection_reason': None,
            'model_id': 'test-model',
            'extraction_time_ms': 100,
        }

    monkeypatch.setattr('episodic.kg.extractor.extract_patch', mock_extract_patch)

    from episodic.kg.realtime import _extract_single_node
    _extract_single_node(node_uuid, "I use Neovim for coding")

    # Check patch was applied
    row = conn.execute(
        "SELECT applied FROM kg_patches WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == 1

    # Check HWM advanced
    assert _get_hwm(conn) == node_id


# ---------------------------------------------------------------------------
# T6: test_realtime_idempotent_with_batch
# ---------------------------------------------------------------------------

def test_realtime_idempotent_with_batch(rt_db, monkeypatch):
    """If batch already processed a node, realtime skips it."""
    conn, db_path = rt_db
    # config not needed — _extract_single_node doesn't check config

    node_uuid, node_id = _insert_user_node(conn, "I use Neovim for coding")
    _set_hwm(conn, node_id - 1)

    # Simulate batch having already processed this node
    conn.execute(
        "INSERT OR REPLACE INTO kg_patches "
        "(node_id, patch_json, patch_hash, validator_version, applied, "
        "rejection_reason, model_id, extraction_time_ms) "
        "VALUES (?, '{\"edges\":[]}', 'batch_hash', 'v1', 1, NULL, 'batch-model', 50)",
        (node_id,),
    )
    conn.commit()

    from contextlib import contextmanager

    @contextmanager
    def mock_use_conn(c=None):
        yield conn

    monkeypatch.setattr('episodic.kg.db_kg._use_conn', mock_use_conn)
    monkeypatch.setattr('episodic.kg.applicator._use_conn', mock_use_conn)

    # Mock extract_patch — should NOT be called
    call_count = [0]

    def mock_extract_patch(nid, lookback=3, conn=None):
        call_count[0] += 1
        return {
            'node_id': nid, 'patch_json': '{}', 'patch_hash': 'h',
            'applied': 0, 'rejection_reason': None,
            'model_id': 'test', 'extraction_time_ms': 0,
        }

    monkeypatch.setattr('episodic.kg.extractor.extract_patch', mock_extract_patch)

    from episodic.kg.realtime import _extract_single_node
    _extract_single_node(node_uuid, "I use Neovim for coding")

    # extract_patch was called (step 2 happens before idempotency check in step 4)
    # but apply_patch should not overwrite the batch result
    row = conn.execute(
        "SELECT model_id FROM kg_patches WHERE node_id = ?",
        (node_id,),
    ).fetchone()
    # Should still be batch-model (idempotency guard prevents overwrite)
    assert row[0] == 'batch-model'
