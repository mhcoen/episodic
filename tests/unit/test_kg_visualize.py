"""Tests for episodic.kg.visualize."""

import sqlite3
import os
import pytest

from episodic.kg.visualize import render_kg_html, visualize_kg


# Reuse the schema from graph_builder tests
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
    db_path = tmp_path / "test_viz.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(KG_SCHEMA_SQL)

    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('person', 'user:self', 'self', 0, 0.0)"
    )
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, created_node_id, created_at) "
        "VALUES ('artifact', NULL, 'Vim', 10, 1.0)"
    )
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, asserted_by, status, tags) "
        "VALUES (10, 0, 24, 'user', 'active', '[]')"
    )
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (1, 'uses', 2, 1)"
    )
    conn.execute("INSERT INTO nodes VALUES (10, 'I use Vim for everything')")
    conn.commit()
    return conn


@pytest.fixture
def empty_kg_db(tmp_path):
    """Create a test database with KG schema but no data."""
    db_path = tmp_path / "empty_viz.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(KG_SCHEMA_SQL)
    conn.commit()
    return conn


def test_render_kg_html_produces_valid_html(kg_db):
    """Output contains DOCTYPE, cytoscape CDN, and graph data."""
    html = render_kg_html(conn=kg_db)
    assert '<!DOCTYPE html>' in html
    assert 'cytoscape' in html.lower()
    assert 'cytoscape.min.js' in html
    assert 'graphData' in html
    assert 'self' in html  # entity name
    assert 'Vim' in html   # entity name


def test_render_empty_graph(empty_kg_db):
    """Empty graph produces valid HTML with empty elements."""
    html = render_kg_html(conn=empty_kg_db)
    assert '<!DOCTYPE html>' in html
    assert '"nodes": []' in html
    assert '"edges": []' in html


def test_render_contains_controls(kg_db):
    """HTML contains filter controls, panel, status bar."""
    html = render_kg_html(conn=kg_db)
    assert 'id="controls"' in html
    assert 'id="panel"' in html
    assert 'id="statusbar"' in html
    assert 'id="search-input"' in html
    assert 'type-filter' in html
    assert 'pred-filter' in html
    assert 'layout-btn' in html


def test_save_to_file(kg_db, tmp_path):
    """visualize_kg(save_path=...) writes file to disk."""
    save_path = str(tmp_path / "output.html")
    result = visualize_kg(save_path=save_path, conn=kg_db)
    assert result == save_path
    assert os.path.exists(save_path)

    with open(save_path, 'r') as f:
        content = f.read()
    assert '<!DOCTYPE html>' in content
    assert 'cytoscape.min.js' in content


def test_save_creates_directory(kg_db, tmp_path):
    """save_path with non-existent parent directory is created."""
    save_path = str(tmp_path / "subdir" / "output.html")
    result = visualize_kg(save_path=save_path, conn=kg_db)
    assert os.path.exists(save_path)


def test_layout_parameter_cose(kg_db):
    """initial_layout Jinja2 variable set correctly for cose."""
    html = render_kg_html(layout='cose', conn=kg_db)
    assert "LAYOUTS['cose']" in html


def test_layout_parameter_concentric(kg_db):
    """initial_layout set correctly for concentric."""
    html = render_kg_html(layout='concentric', conn=kg_db)
    assert "LAYOUTS['concentric']" in html


def test_layout_parameter_grid(kg_db):
    """initial_layout set correctly for grid."""
    html = render_kg_html(layout='grid', conn=kg_db)
    assert "LAYOUTS['grid']" in html


def test_max_degree_at_least_one(empty_kg_db):
    """max_degree is at least 1 even with no nodes."""
    html = render_kg_html(conn=empty_kg_db)
    # Should not contain MAX_DEGREE = 0 (would break mapData)
    assert 'MAX_DEGREE = 0' not in html


def test_dark_theme_colors(kg_db):
    """HTML contains dark theme background colors."""
    html = render_kg_html(conn=kg_db)
    assert '#0f0f1a' in html or '#12121f' in html  # body/cy background
    assert '#16213e' in html  # control bar
    assert '#4A90D9' in html  # person color
    assert '#E8833A' in html  # artifact color
