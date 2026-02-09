"""Phase 1.2 tests: new predicates, domain/range enforcement, read-side priority."""

import sqlite3
import time

import pytest

from episodic.kg.validator import (
    validate_patch,
    ALLOWED_PREDICATES,
    DOMAIN_RANGE,
    STRIP_EDGE_DOMAIN_RANGE_VIOLATION,
)
from episodic.kg.context_source import (
    PREDICATE_PRIORITY,
    get_kg_context,
    EdgeFact,
    _mention_dict,
)
from episodic.kg.schema import ensure_kg_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_patch(text, subj_type, subj_name, obj_type, obj_name, predicate):
    """Build a minimal patch with one edge for validator testing."""
    return {
        'schema_version': 'kg_patch_v1',
        'node_id': 1,
        'assertions': [{
            'assertion_key': 'a1',
            'span_start': 0,
            'span_end': len(text),
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        }],
        'entities': [
            {
                'entity_key': 'e1',
                'entity_type': subj_type,
                'canonical_name': subj_name,
                'canonical_key': None,
                'created_by_assertion': 'a1',
                'resolution_hint': None,
            },
            {
                'entity_key': 'e2',
                'entity_type': obj_type,
                'canonical_name': obj_name,
                'canonical_key': None,
                'created_by_assertion': 'a1',
                'resolution_hint': None,
            },
        ],
        'aliases': [],
        'mentions': [
            {
                'mention_key': 'm1',
                'span_start': text.index(subj_name),
                'span_end': text.index(subj_name) + len(subj_name),
                'surface_text': subj_name,
                'entity_ref': 'e1',
                'confidence': 0.9,
                'source_assertion': 'a1',
            },
            {
                'mention_key': 'm2',
                'span_start': text.index(obj_name),
                'span_end': text.index(obj_name) + len(obj_name),
                'surface_text': obj_name,
                'entity_ref': 'e2',
                'confidence': 0.9,
                'source_assertion': 'a1',
            },
        ],
        'edges': [{
            'subj_ref': 'e1',
            'predicate': predicate,
            'obj_ref': 'e2',
            'source_assertion': 'a1',
            'confidence': 0.9,
        }],
        'notes': None,
    }


# ---------------------------------------------------------------------------
# T1: studies predicate accepted by validator
# ---------------------------------------------------------------------------

def test_studies_accepted():
    """studies(person, topic) passes validation."""
    text = "Emma studies computer science at the university."
    patch = _make_patch(text, 'person', 'Emma', 'topic', 'computer science', 'studies')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1
    assert result.cleaned_patch['edges'][0]['predicate'] == 'studies'


# ---------------------------------------------------------------------------
# T2: affiliated_with predicate accepted by validator
# ---------------------------------------------------------------------------

def test_affiliated_with_accepted():
    """affiliated_with(person, org) passes validation."""
    text = "Sarah is affiliated with Google as a researcher."
    patch = _make_patch(text, 'person', 'Sarah', 'org', 'Google', 'affiliated_with')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1
    assert result.cleaned_patch['edges'][0]['predicate'] == 'affiliated_with'


# ---------------------------------------------------------------------------
# T3: works_on predicate accepted by validator
# ---------------------------------------------------------------------------

def test_works_on_accepted():
    """works_on(person, artifact) passes validation."""
    text = "Jake is building a marketplace for local artisans."
    patch = _make_patch(text, 'person', 'Jake', 'artifact', 'marketplace', 'works_on')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 1
    assert result.cleaned_patch['edges'][0]['predicate'] == 'works_on'


# ---------------------------------------------------------------------------
# T4: studies domain/range enforcement
# ---------------------------------------------------------------------------

def test_studies_domain_range_violation():
    """studies(artifact, topic) violates domain — stripped."""
    text = "MacBook studies computer science somehow."
    patch = _make_patch(text, 'artifact', 'MacBook', 'topic', 'computer science', 'studies')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_domain_range_violation' in w for w in result.warnings)


# ---------------------------------------------------------------------------
# T5: affiliated_with domain/range enforcement
# ---------------------------------------------------------------------------

def test_affiliated_with_domain_range_violation():
    """affiliated_with(artifact, org) violates domain — stripped."""
    text = "MacBook is affiliated with Apple in some way."
    patch = _make_patch(text, 'artifact', 'MacBook', 'org', 'Apple', 'affiliated_with')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_domain_range_violation' in w for w in result.warnings)


# ---------------------------------------------------------------------------
# T6: works_on domain/range enforcement
# ---------------------------------------------------------------------------

def test_works_on_domain_range_violation():
    """works_on(org, artifact) violates domain — stripped."""
    text = "Acme Corp is working on a new widget product."
    patch = _make_patch(text, 'org', 'Acme Corp', 'artifact', 'widget', 'works_on')
    result = validate_patch(patch, text, 1, set(), {})
    assert result.valid
    assert len(result.cleaned_patch['edges']) == 0
    assert any('edge_domain_range_violation' in w for w in result.warnings)


# ---------------------------------------------------------------------------
# T7: read-side predicate priority
# ---------------------------------------------------------------------------

def test_read_side_predicate_priority():
    """Edges with new predicates are returned and ranked per PREDICATE_PRIORITY."""
    conn = sqlite3.connect(':memory:')
    ensure_kg_schema(conn)

    user_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]

    # Create entities
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('person', NULL, 'Emma', 1, ?)",
        (time.time(),),
    )
    emma_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('topic', NULL, 'physics', 2, ?)",
        (time.time(),),
    )
    physics_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('org', NULL, 'Stanford', 3, ?)",
        (time.time(),),
    )
    stanford_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('artifact', NULL, 'thesis project', 4, ?)",
        (time.time(),),
    )
    thesis_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Nodes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user'
        )
    """)
    for nid in range(1, 5):
        conn.execute("INSERT OR IGNORE INTO nodes VALUES (?, ?, 'user')",
                     (nid, f"node {nid}"))

    # Assertions
    for nid in range(1, 4):
        conn.execute(
            "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
            "asserted_by, polarity, certainty, status, tags) "
            "VALUES (?, 0, 10, 'user', 'affirm', 'explicit', 'active', '[]')",
            (nid,),
        )

    a1 = 1  # studies
    a2 = 2  # affiliated_with
    a3 = 3  # works_on

    # Edges with new predicates
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'studies', ?, ?)", (emma_id, physics_id, a1))
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'affiliated_with', ?, ?)", (emma_id, stanford_id, a2))
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'works_on', ?, ?)", (emma_id, thesis_id, a3))

    conn.execute("UPDATE kg_state SET value = '5' WHERE key = 'high_water_mark'")
    conn.commit()

    # Force rebuild of mention dict
    _mention_dict._hwm = ""

    result = get_kg_context("Tell me about Emma", conn)
    assert result is not None

    # All three new predicate edges should be present
    predicates = {e.predicate for e in result.edges}
    assert 'studies' in predicates, f"studies not in {predicates}"
    assert 'affiliated_with' in predicates, f"affiliated_with not in {predicates}"
    assert 'works_on' in predicates, f"works_on not in {predicates}"

    # Verify ranking: affiliated_with (3) < studies (8) < works_on (9)
    aff = [e for e in result.edges if e.predicate == 'affiliated_with'][0]
    stu = [e for e in result.edges if e.predicate == 'studies'][0]
    wk = [e for e in result.edges if e.predicate == 'works_on'][0]
    assert aff.rank_score > stu.rank_score, "affiliated_with should rank higher than studies"

    conn.close()


# ---------------------------------------------------------------------------
# T8: all predicates in PREDICATE_PRIORITY dict
# ---------------------------------------------------------------------------

def test_all_predicates_in_priority():
    """All 13 predicates from ALLOWED_PREDICATES have PREDICATE_PRIORITY entries."""
    for pred in ALLOWED_PREDICATES:
        assert pred in PREDICATE_PRIORITY, (
            f"Predicate '{pred}' missing from PREDICATE_PRIORITY"
        )
    assert len(PREDICATE_PRIORITY) == len(ALLOWED_PREDICATES), (
        f"PREDICATE_PRIORITY has {len(PREDICATE_PRIORITY)} entries but "
        f"ALLOWED_PREDICATES has {len(ALLOWED_PREDICATES)}"
    )
