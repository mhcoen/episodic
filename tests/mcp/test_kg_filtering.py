"""Tests for MCP security KG filtering (spec tests 65-72 + I4/I12).

Uses an in-memory SQLite DB with KG schema. Tests that:
- Untrusted sources produce quarantined assertions
- Quarantined assertions are excluded from context assembly
- Promotion moves assertions out of quarantine with audit trail
"""

import json
import sqlite3
import time

import pytest

from episodic.kg.schema import ensure_kg_schema
from episodic.kg.applicator import apply_patch
from episodic.kg.context_source import retrieve_neighborhood, get_kg_context
from episodic.mcp.security.source_gate import (
    ExtractionPolicy,
    check_extraction_allowed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory DB with KG schema + a minimal nodes table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    # Minimal nodes table (needed by applicator HWM update)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    # Need topic_nodes for get_kg_context
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_nodes (
            node_id TEXT,
            topic_start_node_id TEXT,
            turn_idx INTEGER
        )
    """)
    # Seed a user node
    conn.execute("INSERT INTO nodes (id, content, role) VALUES ('n1', 'test', 'user')")
    conn.execute("INSERT INTO nodes (id, content, role) VALUES ('n2', 'test2', 'user')")
    ensure_kg_schema(conn)
    return conn


def _make_patch(entities, assertions, edges):
    """Build a minimal KG patch dict."""
    return {
        'schema_version': 'kg_patch_v1',
        'entities': entities,
        'assertions': assertions,
        'edges': edges,
        'aliases': [],
        'mentions': [],
    }


def _insert_patch(conn, node_id, entities, edges, quarantine=False,
                  source_origin=""):
    """Insert entities+edges via apply_patch with quarantine control."""
    assertions = []
    for i, edge in enumerate(edges):
        akey = f"a{i}"
        assertions.append({
            'assertion_key': akey,
            'span_start': i * 5,
            'span_end': i * 5 + 4,
            'asserted_by': 'user',
            'polarity': 'affirm',
            'certainty': 'explicit',
            'status': 'active',
            'tags': [],
        })
        edge['source_assertion'] = akey

    patch = _make_patch(entities, assertions, edges)
    patch_json = json.dumps(patch)
    import hashlib
    patch_hash = hashlib.sha256(patch_json.encode()).hexdigest()

    return apply_patch(
        patch=patch,
        node_id=node_id,
        patch_json=patch_json,
        patch_hash=patch_hash,
        model_id='test',
        extraction_time_ms=0,
        conn=conn,
        quarantine=quarantine,
        source_origin=source_origin,
    )


# ---------------------------------------------------------------------------
# Spec test 65: Entities extracted from untrusted source get quarantined=1
# ---------------------------------------------------------------------------

def test_65_entities_from_untrusted_source_quarantined():
    conn = _make_db()
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:bob',
        'canonical_name': 'Bob',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]

    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin="mcp:test-client")

    row = conn.execute(
        "SELECT quarantined, source_origin FROM kg_assertions"
    ).fetchone()
    assert row[0] == 1, "Assertion should be quarantined"
    assert row[1] == "mcp:test-client"


# ---------------------------------------------------------------------------
# Spec test 66: get_kg_context() excludes quarantined assertions
# ---------------------------------------------------------------------------

def test_66_context_excludes_quarantined_assertions():
    conn = _make_db()

    # Insert a trusted edge
    entities_trusted = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:alice',
        'canonical_name': 'Alice',
    }]
    edges_trusted = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities_trusted, edges_trusted,
                  quarantine=False)

    # Insert a quarantined edge for same entity
    edges_quarantined = [{
        'subj_ref': 'e1',
        'predicate': 'located_at',
        'obj_ref': 'user:self',
    }]
    # We need a second node for the second patch
    _insert_patch(conn, 2, [], edges_quarantined,
                  quarantine=True, source_origin="mcp:attacker")

    # Retrieve neighborhood for Alice's entity_id
    alice_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'Alice'"
    ).fetchone()[0]

    facts = retrieve_neighborhood(alice_id, conn)
    predicates = [f.predicate for f in facts]
    assert 'works_at' in predicates, "Trusted edge should be visible"
    assert 'located_at' not in predicates, "Quarantined edge should be filtered"


# ---------------------------------------------------------------------------
# Spec test 67: Cross-trust entities not merged
# (quarantined entity with same canonical_key stays separate)
# ---------------------------------------------------------------------------

def test_67_cross_trust_boundary_no_merge():
    """Quarantined and trusted patches create separate assertions."""
    conn = _make_db()

    # Trusted patch
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:carol',
        'canonical_name': 'Carol',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges, quarantine=False)

    # Quarantined patch referencing same entity via canonical_key
    entities_q = [{
        'entity_key': 'e2',
        'entity_type': 'person',
        'canonical_key': 'person:carol',
        'canonical_name': 'Carol',
    }]
    edges_q = [{
        'subj_ref': 'e2',
        'predicate': 'located_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 2, entities_q, edges_q,
                  quarantine=True, source_origin="mcp:attacker")

    # Both assertions exist but with different quarantine status
    rows = conn.execute(
        "SELECT quarantined, source_origin FROM kg_assertions "
        "ORDER BY assertion_id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == 0  # trusted
    assert rows[1][0] == 1  # quarantined
    assert rows[1][1] == "mcp:attacker"


# ---------------------------------------------------------------------------
# Spec test 68: Edges bridging trusted/quarantined inherit quarantine
# ---------------------------------------------------------------------------

def test_68_edge_quarantine_inheritance():
    """Edge with quarantined assertion is excluded from context."""
    conn = _make_db()

    # Trusted entity
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:dave',
        'canonical_name': 'Dave',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges, quarantine=False)

    # Quarantined edge linking Dave to something else
    entities_q = [{
        'entity_key': 'e2',
        'entity_type': 'org',
        'canonical_key': 'org:evilcorp',
        'canonical_name': 'EvilCorp',
    }]
    edges_q = [{
        'subj_ref': 'e1',
        'predicate': 'affiliated_with',
        'obj_ref': 'e2',
    }]
    _insert_patch(conn, 2, entities_q, edges_q,
                  quarantine=True, source_origin="mcp:attacker")

    dave_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'Dave'"
    ).fetchone()[0]

    facts = retrieve_neighborhood(dave_id, conn)
    predicates = [f.predicate for f in facts]
    assert 'works_at' in predicates
    assert 'affiliated_with' not in predicates, \
        "Quarantined edge should not appear in context"


# ---------------------------------------------------------------------------
# Spec test 69: Closure rules do not traverse quarantined edges
# ---------------------------------------------------------------------------

def test_69_closure_rules_skip_quarantined():
    """apply_closure_rules should not traverse quarantined edges."""
    from episodic.kg.context_source import apply_closure_rules

    conn = _make_db()

    # user:self -> related_to -> Person (trusted)
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:eve',
        'canonical_name': 'Eve',
    }]
    edges = [{
        'subj_ref': 'user:self',
        'predicate': 'related_to',
        'obj_ref': 'e1',
    }]
    _insert_patch(conn, 1, entities, edges, quarantine=False)

    # Eve -> located_at -> City (QUARANTINED)
    entities_q = [{
        'entity_key': 'e2',
        'entity_type': 'topic',
        'canonical_key': 'topic:poisonville',
        'canonical_name': 'Poisonville',
    }]
    edges_q = [{
        'subj_ref': 'e1',
        'predicate': 'located_at',
        'obj_ref': 'e2',
    }]
    _insert_patch(conn, 2, entities_q, edges_q,
                  quarantine=True, source_origin="mcp:attacker")

    user_self_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]
    eve_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'Eve'"
    ).fetchone()[0]

    derived = apply_closure_rules(
        seed_ids=[user_self_id, eve_id],
        edges=[],
        conn=conn,
        max_derived=10,
        prompt_tokens={'eve', 'daughter'},
        user_self_id=user_self_id,
    )

    # KINSHIP_LOCATION closure should NOT fire since located_at edge is quarantined
    derived_objs = [d.obj_name for d in derived]
    assert 'Poisonville' not in derived_objs, \
        "Closure should not traverse quarantined edges"


# ---------------------------------------------------------------------------
# Spec test 70: /kg promote sets quarantined=0, creates promotion record
# ---------------------------------------------------------------------------

def test_70_promote_sets_quarantined_zero():
    conn = _make_db()

    # Insert quarantined assertion
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:frank',
        'canonical_name': 'Frank',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin="mcp:test-client")

    # Get the assertion_id
    aid = conn.execute(
        "SELECT assertion_id FROM kg_assertions WHERE quarantined = 1"
    ).fetchone()[0]
    old_origin = conn.execute(
        "SELECT source_origin FROM kg_assertions WHERE assertion_id = ?",
        (aid,)
    ).fetchone()[0]

    # Simulate promotion
    conn.execute(
        "UPDATE kg_assertions SET quarantined = 0, "
        "source_origin = 'user_promoted_from_' || source_origin "
        "WHERE assertion_id = ?",
        (aid,)
    )
    conn.execute(
        "INSERT INTO kg_promotions "
        "(assertion_id, promoted_at, promoted_by, source_origin) "
        "VALUES (?, ?, 'cli_user', ?)",
        (aid, time.time(), old_origin)
    )
    conn.commit()

    # Verify
    row = conn.execute(
        "SELECT quarantined, source_origin FROM kg_assertions "
        "WHERE assertion_id = ?", (aid,)
    ).fetchone()
    assert row[0] == 0
    assert row[1] == "user_promoted_from_mcp:test-client"

    promo = conn.execute(
        "SELECT source_origin FROM kg_promotions WHERE assertion_id = ?",
        (aid,)
    ).fetchone()
    assert promo[0] == "mcp:test-client"


# ---------------------------------------------------------------------------
# Spec test 71: Promoted triples appear in get_kg_context()
# ---------------------------------------------------------------------------

def test_71_promoted_triples_in_context():
    conn = _make_db()

    # Insert quarantined assertion
    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:grace',
        'canonical_name': 'Grace',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin="mcp:test-client")

    grace_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'Grace'"
    ).fetchone()[0]

    # Before promotion: no edges
    facts = retrieve_neighborhood(grace_id, conn)
    assert len(facts) == 0

    # Promote
    aid = conn.execute(
        "SELECT assertion_id FROM kg_assertions WHERE quarantined = 1"
    ).fetchone()[0]
    conn.execute(
        "UPDATE kg_assertions SET quarantined = 0 WHERE assertion_id = ?",
        (aid,)
    )
    conn.commit()

    # After promotion: edge visible
    facts = retrieve_neighborhood(grace_id, conn)
    assert len(facts) == 1
    assert facts[0].predicate == 'works_at'


# ---------------------------------------------------------------------------
# Spec test 72: Bulk promotion ("all") works and logs correctly
# ---------------------------------------------------------------------------

def test_72_bulk_promotion():
    conn = _make_db()

    # Insert multiple quarantined assertions
    entities = [
        {
            'entity_key': 'e1', 'entity_type': 'person',
            'canonical_key': 'person:hank', 'canonical_name': 'Hank',
        },
        {
            'entity_key': 'e2', 'entity_type': 'org',
            'canonical_key': 'org:acme', 'canonical_name': 'Acme Corp',
        },
    ]
    edges = [
        {'subj_ref': 'e1', 'predicate': 'works_at', 'obj_ref': 'e2'},
        {'subj_ref': 'e1', 'predicate': 'located_at', 'obj_ref': 'user:self'},
    ]
    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin="mcp:claude-code")

    # All assertions are quarantined
    count = conn.execute(
        "SELECT COUNT(*) FROM kg_assertions WHERE quarantined = 1"
    ).fetchone()[0]
    assert count == 2

    # Bulk promote all
    rows = conn.execute(
        "SELECT assertion_id, source_origin FROM kg_assertions "
        "WHERE quarantined = 1"
    ).fetchall()

    now = time.time()
    for aid, origin in rows:
        conn.execute(
            "UPDATE kg_assertions SET quarantined = 0, "
            "source_origin = 'user_promoted_from_' || source_origin "
            "WHERE assertion_id = ?",
            (aid,)
        )
        conn.execute(
            "INSERT INTO kg_promotions "
            "(assertion_id, promoted_at, promoted_by, source_origin) "
            "VALUES (?, ?, 'cli_user', ?)",
            (aid, now, origin)
        )
    conn.commit()

    # Verify all promoted
    remaining = conn.execute(
        "SELECT COUNT(*) FROM kg_assertions WHERE quarantined = 1"
    ).fetchone()[0]
    assert remaining == 0

    # Verify audit trail
    promos = conn.execute(
        "SELECT COUNT(*) FROM kg_promotions"
    ).fetchone()[0]
    assert promos == 2


# ---------------------------------------------------------------------------
# Test I4 integration: source gate blocks email_content extraction
# ---------------------------------------------------------------------------

def test_i4_email_content_blocked_by_source_gate():
    result = check_extraction_allowed("email_content")
    assert result.policy == ExtractionPolicy.BLOCK


# ---------------------------------------------------------------------------
# Test I12: MCP client content is quarantined
# ---------------------------------------------------------------------------

def test_i12_mcp_client_content_quarantined():
    result = check_extraction_allowed("mcp_client", source_id="some-client")
    assert result.policy == ExtractionPolicy.QUARANTINE


def test_i12_quarantined_excluded_from_context_assembly():
    """End-to-end: mcp_client content quarantined and excluded from context."""
    conn = _make_db()

    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:iris',
        'canonical_name': 'Iris',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]

    # Check gate, then apply with quarantine
    gate = check_extraction_allowed("mcp_client", "claude-code")
    assert gate.policy == ExtractionPolicy.QUARANTINE

    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin=gate.source_origin)

    iris_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'Iris'"
    ).fetchone()[0]

    facts = retrieve_neighborhood(iris_id, conn)
    assert len(facts) == 0, "Quarantined assertions should not appear"


# ---------------------------------------------------------------------------
# Test: Default source_type (user_input) is ALLOW — existing paths unbroken
# ---------------------------------------------------------------------------

def test_default_source_type_is_allow():
    """The default source_type maps to ALLOW, not breaking existing paths."""
    result = check_extraction_allowed("user_input")
    assert result.policy == ExtractionPolicy.ALLOW


def test_web_synthesis_quarantined_and_excluded():
    """web_synthesis source type produces quarantined assertions excluded from context."""
    conn = _make_db()

    gate = check_extraction_allowed("web_synthesis")
    assert gate.policy == ExtractionPolicy.QUARANTINE

    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:muse_person',
        'canonical_name': 'MusePerson',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges,
                  quarantine=True, source_origin=gate.source_origin)

    # Verify quarantined=1
    row = conn.execute(
        "SELECT quarantined, source_origin FROM kg_assertions"
    ).fetchone()
    assert row[0] == 1, "web_synthesis assertion should be quarantined"
    assert row[1] == "web_synthesis"

    # Verify excluded from context
    eid = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_name = 'MusePerson'"
    ).fetchone()[0]
    facts = retrieve_neighborhood(eid, conn)
    assert len(facts) == 0, "Quarantined web_synthesis edges should not appear in context"


def test_apply_patch_default_no_quarantine():
    """apply_patch without quarantine param produces quarantined=0."""
    conn = _make_db()

    entities = [{
        'entity_key': 'e1',
        'entity_type': 'person',
        'canonical_key': 'person:jack',
        'canonical_name': 'Jack',
    }]
    edges = [{
        'subj_ref': 'e1',
        'predicate': 'works_at',
        'obj_ref': 'user:self',
    }]
    _insert_patch(conn, 1, entities, edges)  # No quarantine args

    row = conn.execute(
        "SELECT quarantined FROM kg_assertions"
    ).fetchone()
    assert row[0] == 0, "Default should be non-quarantined"
