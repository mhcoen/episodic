"""Tests for episodic.kg.applicator."""

import json
import sqlite3
import time

import pytest

from episodic.kg.applicator import apply_patch, record_rejected_patch
from episodic.kg.schema import ensure_kg_schema
from episodic.kg.validator import VALIDATOR_VERSION


@pytest.fixture
def kg_db():
    """Create an in-memory DB with KG schema + nodes table."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    conn.execute(
        "INSERT INTO nodes VALUES (1, 'I use Vim daily.', 'user')"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (2, 'I prefer Python over Java.', 'user')"
    )
    ensure_kg_schema(conn)
    yield conn
    conn.close()


def _minimal_patch(node_id=1):
    """Build a minimal valid patch."""
    return {
        'schema_version': 'kg_patch_v1',
        'node_id': node_id,
        'assertions': [{
            'assertion_key': 'a1',
            'span_start': 0,
            'span_end': 17,
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        }],
        'entities': [],
        'aliases': [],
        'mentions': [{
            'mention_key': 'm1',
            'span_start': 6,
            'span_end': 9,
            'surface_text': 'Vim',
            'entity_ref': None,
            'confidence': 0.9,
            'source_assertion': 'a1',
        }],
        'edges': [],
        'notes': None,
    }


def test_apply_minimal_patch(kg_db):
    """Single assertion + single mention applied to empty DB."""
    patch = _minimal_patch()
    patch_json = json.dumps(patch)

    result = apply_patch(
        patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100,
        conn=kg_db,
    )
    assert result['applied'] is True
    assert result['assertions_created'] == 1
    assert result['mentions_created'] == 1

    # Verify assertion in DB
    row = kg_db.execute(
        "SELECT COUNT(*) FROM kg_assertions"
    ).fetchone()
    assert row[0] == 1


def test_apply_with_new_entity(kg_db):
    """New entity created, entity_id returned correctly."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['mentions'][0]['entity_ref'] = 'e1'
    patch_json = json.dumps(patch)

    result = apply_patch(
        patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100,
        conn=kg_db,
    )
    assert result['entities_created'] == 1
    assert result['entities_resolved'] == 0

    # Verify entity in DB (user:self + new entity = 2)
    row = kg_db.execute(
        "SELECT COUNT(*) FROM kg_entities"
    ).fetchone()
    assert row[0] == 2


def test_apply_with_resolved_entity(kg_db):
    """Existing entity resolved, no new row in kg_entities."""
    # First, create an entity
    kg_db.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('artifact', 'vim:editor', 'Vim', 1, ?)",
        (time.time(),)
    )
    kg_db.commit()

    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': 'vim:editor',
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch_json = json.dumps(patch)

    result = apply_patch(
        patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100,
        conn=kg_db,
    )
    assert result['entities_resolved'] == 1
    assert result['entities_created'] == 0


def test_apply_with_edge(kg_db):
    """Edge references correct entity_ids and assertion_id."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    patch_json = json.dumps(patch)

    result = apply_patch(
        patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100,
        conn=kg_db,
    )
    assert result['edges_created'] == 1

    # Verify edge in DB
    row = kg_db.execute(
        "SELECT subj_entity_id, predicate, obj_entity_id FROM kg_edges"
    ).fetchone()
    assert row is not None
    assert row[1] == 'uses'


def test_apply_with_alias(kg_db):
    """Alias inserted into kg_entity_aliases."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['aliases'] = [{
        'entity_ref': 'e1',
        'alias_text': 'vi',
        'source_assertion': 'a1',
        'span_start': 6,
        'span_end': 9,
    }]
    patch_json = json.dumps(patch)

    result = apply_patch(
        patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100,
        conn=kg_db,
    )
    assert result['aliases_created'] == 1


def test_duplicate_alias_ignored(kg_db):
    """INSERT OR IGNORE handles duplicate alias gracefully."""
    patch = _minimal_patch()
    # Use canonical_key so second apply resolves to same entity_id
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': 'vim:editor',
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['aliases'] = [{
        'entity_ref': 'e1',
        'alias_text': 'vi',
        'source_assertion': 'a1',
        'span_start': 6,
        'span_end': 9,
    }]
    patch_json = json.dumps(patch)

    # Apply twice — second resolves same entity via canonical_key
    apply_patch(patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100, conn=kg_db)

    # Second apply with different node_id to avoid patch UNIQUE constraint
    patch['node_id'] = 2
    patch_json2 = json.dumps(patch)
    apply_patch(patch, 2, patch_json2, 'def456', 'gpt-4o-mini', 100, conn=kg_db)

    # Only one alias should exist (same entity_id + same alias text = UNIQUE conflict)
    row = kg_db.execute(
        "SELECT COUNT(*) FROM kg_entity_aliases"
    ).fetchone()
    assert row[0] == 1


def test_hwm_advances(kg_db):
    """High water mark updated to node_id after successful apply."""
    patch = _minimal_patch()
    patch_json = json.dumps(patch)

    apply_patch(patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100, conn=kg_db)

    row = kg_db.execute(
        "SELECT value FROM kg_state WHERE key = 'high_water_mark'"
    ).fetchone()
    assert row[0] == '1'


def test_record_rejected_patch(kg_db):
    """Rejected patch stored with applied=0 and reason."""
    record_rejected_patch(
        node_id=1,
        patch_json='{}',
        patch_hash='abc',
        rejection_reason='test_reason',
        model_id='gpt-4o-mini',
        extraction_time_ms=50,
        conn=kg_db,
    )

    row = kg_db.execute(
        "SELECT applied, rejection_reason FROM kg_patches WHERE node_id = 1"
    ).fetchone()
    assert row[0] == 0
    assert row[1] == 'test_reason'


def test_user_self_entity_id(kg_db):
    """user:self resolved to correct entity_id (not hardcoded)."""
    patch = _minimal_patch()
    patch['entities'] = [{
        'entity_key': 'e1',
        'entity_type': 'artifact',
        'canonical_name': 'Vim',
        'canonical_key': None,
        'created_by_assertion': 'a1',
        'resolution_hint': None,
    }]
    patch['edges'] = [{
        'subj_ref': 'user:self',
        'predicate': 'uses',
        'obj_ref': 'e1',
        'source_assertion': 'a1',
        'confidence': 0.9,
    }]
    patch_json = json.dumps(patch)

    apply_patch(patch, 1, patch_json, 'abc123', 'gpt-4o-mini', 100, conn=kg_db)

    # Get user:self entity_id
    row = kg_db.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()
    user_self_id = row[0]

    # Verify edge uses the right ID
    edge = kg_db.execute(
        "SELECT subj_entity_id FROM kg_edges"
    ).fetchone()
    assert edge[0] == user_self_id
