"""Tests for episodic.kg.graph_builder."""

import sqlite3
import pytest

from episodic.kg.graph_builder import (
    build_kg_graph, graph_to_cytoscape_json, PREDICATE_COLORS, DEFAULT_EDGE_COLOR,
)


# --- Schema SQL for KG tables ---

KG_SCHEMA_SQL = """
CREATE TABLE kg_entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    canonical_key TEXT,
    canonical_name TEXT NOT NULL,
    created_node_id INTEGER,
    created_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE kg_assertions (
    assertion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    asserted_by TEXT NOT NULL,
    polarity TEXT NOT NULL DEFAULT 'affirm',
    certainty TEXT NOT NULL DEFAULT 'explicit',
    status TEXT NOT NULL DEFAULT 'active',
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE kg_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subj_entity_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    obj_entity_id INTEGER NOT NULL,
    assertion_id INTEGER NOT NULL,
    FOREIGN KEY (subj_entity_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (obj_entity_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (assertion_id) REFERENCES kg_assertions(assertion_id)
);

CREATE TABLE kg_entity_aliases (
    entity_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES kg_entities(entity_id)
);

CREATE TABLE nodes (
    node_id INTEGER PRIMARY KEY,
    content TEXT
);
"""


@pytest.fixture
def kg_db(tmp_path):
    """Create a test database with KG schema and sample data."""
    db_path = tmp_path / "test_kg.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(KG_SCHEMA_SQL)

    # Insert user:self entity (entity_id=1)
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('person', 'user:self', 'self', 0, 0.0)"
    )
    # Insert Vim entity (entity_id=2)
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('artifact', NULL, 'Vim', 10, 1.0)"
    )
    # Insert Python entity (entity_id=3)
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('topic', NULL, 'Python', 15, 2.0)"
    )
    # Insert Anthropic entity (entity_id=4)
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('org', NULL, 'ACME Corp', 20, 3.0)"
    )

    # Insert aliases for Vim
    conn.execute("INSERT INTO kg_entity_aliases VALUES (2, 'vi')")
    conn.execute("INSERT INTO kg_entity_aliases VALUES (2, 'nvim')")

    # Insert assertion + edge: self -> uses -> Vim
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, asserted_by, polarity, certainty, status, tags) "
        "VALUES (10, 0, 24, 'user', 'affirm', 'explicit', 'active', '[]')"
    )
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'uses', 2, 1)"
    )

    # Insert assertion + edge: self -> wants -> Python
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, asserted_by, polarity, certainty, status, tags) "
        "VALUES (15, 0, 22, 'user', 'affirm', 'explicit', 'active', '[]')"
    )
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'wants', 3, 2)"
    )

    # Insert assertion + edge: self -> role -> ACME Corp (with TIME_PAST tag)
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, asserted_by, polarity, certainty, status, tags) "
        """VALUES (20, 0, 30, 'user', 'affirm', 'explicit', 'active', '["TIME_PAST"]')"""
    )
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'role', 4, 3)"
    )

    # Insert node content for span text resolution
    conn.execute("INSERT INTO nodes VALUES (10, 'I use Vim for everything')")
    conn.execute("INSERT INTO nodes VALUES (15, 'I want to learn Python')")
    conn.execute("INSERT INTO nodes VALUES (20, 'I worked at ACME Corp previously')")

    conn.commit()
    return conn


@pytest.fixture
def empty_kg_db(tmp_path):
    """Create a test database with KG schema but no data."""
    db_path = tmp_path / "empty_kg.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(KG_SCHEMA_SQL)
    conn.commit()
    return conn


@pytest.fixture
def no_kg_db(tmp_path):
    """Create a database without KG tables."""
    db_path = tmp_path / "no_kg.db"
    conn = sqlite3.connect(str(db_path))
    conn.commit()
    return conn


# --- Tests ---

def test_build_empty_graph(empty_kg_db):
    """Empty KG produces graph with no nodes/edges."""
    G = build_kg_graph(conn=empty_kg_db)
    assert len(G.nodes()) == 0
    assert len(G.edges()) == 0


def test_build_no_kg_tables(no_kg_db):
    """Missing KG tables return empty graph without crashing."""
    G = build_kg_graph(conn=no_kg_db)
    assert len(G.nodes()) == 0
    assert len(G.edges()) == 0


def test_build_basic_graph(kg_db):
    """Insert entities + edges, verify NetworkX graph structure."""
    G = build_kg_graph(conn=kg_db)
    # 4 entities, but all should have edges via 'self'
    # self(1) has degree 3, Vim(2) degree 1, Python(3) degree 1, ACME(4) degree 1
    assert len(G.nodes()) == 4
    assert len(G.edges()) == 3

    # Check self node
    self_data = G.nodes['e1']
    assert self_data['entity_type'] == 'person'
    assert self_data['is_user_self'] is True
    assert self_data['degree'] == 3

    # Check Vim node
    vim_data = G.nodes['e2']
    assert vim_data['entity_type'] == 'artifact'
    assert 'vi' in vim_data['aliases']
    assert 'nvim' in vim_data['aliases']


def test_entity_type_filter(kg_db):
    """Filter by entity_type excludes non-matching nodes and their edges."""
    G = build_kg_graph(entity_types=['person', 'artifact'], conn=kg_db)
    # Only self + Vim should be present (Python and ACME filtered out)
    types = {G.nodes[n]['entity_type'] for n in G.nodes()}
    assert 'topic' not in types
    assert 'org' not in types
    assert 'person' in types
    assert 'artifact' in types
    # Only the uses edge should exist (wants and role targets filtered out)
    assert len(G.edges()) == 1


def test_predicate_filter(kg_db):
    """Filter by predicate hides non-matching edges; orphaned nodes pruned."""
    G = build_kg_graph(predicates=['uses'], conn=kg_db)
    # Only self + Vim should remain (Python and ACME become isolates, pruned)
    assert len(G.edges()) == 1
    assert 'e1' in G  # self
    assert 'e2' in G  # Vim
    assert 'e3' not in G  # Python (pruned isolate)
    assert 'e4' not in G  # ACME (pruned isolate)


def test_user_self_always_kept(kg_db):
    """user:self node kept even when degree=0 after filtering."""
    # Filter predicates that don't exist — no edges match
    G = build_kg_graph(predicates=['nonexistent'], conn=kg_db)
    # Self should still be present even with no edges
    assert 'e1' in G
    assert G.nodes['e1']['is_user_self'] is True
    assert G.nodes['e1']['degree'] == 0


def test_node_id_range_filter(kg_db):
    """Filter by node_id range on assertions."""
    # Only include assertions from node_id 10-14 (Vim edge only)
    G = build_kg_graph(node_id_range=(10, 14), conn=kg_db)
    assert len(G.edges()) == 1
    edge_data = list(G.edges(data=True))[0][2]
    assert edge_data['predicate'] == 'uses'


def test_tags_filter(kg_db):
    """Filter by tags on assertions."""
    G = build_kg_graph(tags=['TIME_PAST'], conn=kg_db)
    assert len(G.edges()) == 1
    edge_data = list(G.edges(data=True))[0][2]
    assert edge_data['has_time_past'] is True


def test_cytoscape_json_format(kg_db):
    """Output matches Cytoscape.js elements JSON schema."""
    G = build_kg_graph(conn=kg_db)
    cy = graph_to_cytoscape_json(G)

    assert 'nodes' in cy
    assert 'edges' in cy
    assert len(cy['nodes']) == 4
    assert len(cy['edges']) == 3

    # Check node format
    node = cy['nodes'][0]
    assert 'data' in node
    assert 'id' in node['data']
    assert 'entity_type' in node['data']
    assert 'canonical_name' in node['data']
    assert 'degree' in node['data']

    # Check edge format
    edge = cy['edges'][0]
    assert 'data' in edge
    assert 'id' in edge['data']
    assert 'source' in edge['data']
    assert 'target' in edge['data']
    assert 'predicate' in edge['data']
    assert 'edgeColor' in edge['data']


def test_edge_color_mapping(kg_db):
    """Each predicate gets correct edgeColor in data."""
    G = build_kg_graph(conn=kg_db)
    cy = graph_to_cytoscape_json(G)

    edge_colors = {e['data']['predicate']: e['data']['edgeColor'] for e in cy['edges']}
    assert edge_colors['uses'] == PREDICATE_COLORS['uses']
    assert edge_colors['wants'] == PREDICATE_COLORS['wants']
    assert edge_colors['role'] == PREDICATE_COLORS['role']


def test_time_past_flag(kg_db):
    """Edge with TIME_PAST tag has has_time_past=True in data."""
    G = build_kg_graph(conn=kg_db)
    cy = graph_to_cytoscape_json(G)

    role_edge = next(e for e in cy['edges'] if e['data']['predicate'] == 'role')
    assert role_edge['data']['has_time_past'] is True

    uses_edge = next(e for e in cy['edges'] if e['data']['predicate'] == 'uses')
    assert uses_edge['data']['has_time_past'] is False


def test_span_text_resolution(kg_db):
    """Span text is resolved from nodes table."""
    G = build_kg_graph(conn=kg_db)
    cy = graph_to_cytoscape_json(G)

    uses_edge = next(e for e in cy['edges'] if e['data']['predicate'] == 'uses')
    assert uses_edge['data']['span_text'] == 'I use Vim for everything'


def test_empty_cytoscape_json():
    """Empty graph produces valid JSON structure."""
    import networkx as nx
    G = nx.DiGraph()
    cy = graph_to_cytoscape_json(G)
    assert cy == {'nodes': [], 'edges': []}


def test_unknown_predicate_gets_default_color(kg_db):
    """Predicates not in PREDICATE_COLORS get default color."""
    # Add an edge with unknown predicate
    kg_db.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, asserted_by, status, tags) "
        "VALUES (10, 0, 5, 'user', 'active', '[]')"
    )
    kg_db.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'likes', 2, 4)"
    )
    kg_db.commit()

    G = build_kg_graph(conn=kg_db)
    cy = graph_to_cytoscape_json(G)
    likes_edge = next(e for e in cy['edges'] if e['data']['predicate'] == 'likes')
    assert likes_edge['data']['edgeColor'] == DEFAULT_EDGE_COLOR
