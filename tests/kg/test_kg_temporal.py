"""Tests for KG temporal predicates (Phase A)."""

import sqlite3
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import typer

from episodic.kg.schema import ensure_kg_schema
from episodic.kg.validator import (
    ALLOWED_PREDICATES,
    DOMAIN_RANGE,
    validate_patch,
    resolve_entity_type,
)
from episodic.kg.context_source import PREDICATE_PRIORITY


# ---------------------------------------------------------------------------
# Shared helpers (reused from test_kg_readside.py pattern)
# ---------------------------------------------------------------------------

TEMPORAL_PREDICATES = {'deadline', 'scheduled_for', 'starts_at', 'ends_at', 'recurring'}


def _fresh_db():
    """Create an in-memory DB with KG schema seeded."""
    conn = sqlite3.connect(':memory:')
    ensure_kg_schema(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user'
        )
    """)
    return conn


def _add_entity(conn, etype, name, node_id, ckey=None):
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (etype, ckey, name, node_id, time.time()),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_assertion(conn, node_id):
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
        "asserted_by, polarity, certainty, status, tags) "
        "VALUES (?, 0, 10, 'user', 'affirm', 'explicit', 'active', '[]')",
        (node_id,),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_edge(conn, subj_id, pred, obj_id, node_id):
    conn.execute(f"INSERT OR IGNORE INTO nodes VALUES ({node_id}, 'test', 'user')")
    aid = _add_assertion(conn, node_id)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, ?, ?, ?)", (subj_id, pred, obj_id, aid),
    )


# ---------------------------------------------------------------------------
# Test 1: ALLOWED_PREDICATES includes all temporal predicates
# ---------------------------------------------------------------------------

class TestTemporalPredicatesInSchema:
    def test_allowed_predicates_includes_temporal(self):
        for pred in TEMPORAL_PREDICATES:
            assert pred in ALLOWED_PREDICATES, f"{pred} missing from ALLOWED_PREDICATES"

    def test_domain_range_entries_exist(self):
        for pred in TEMPORAL_PREDICATES:
            assert pred in DOMAIN_RANGE, f"{pred} missing from DOMAIN_RANGE"

    def test_deadline_domain_range(self):
        subj, obj = DOMAIN_RANGE['deadline']
        assert subj == {'artifact', 'topic', 'org'}
        assert obj == {'topic'}

    def test_scheduled_for_domain_range(self):
        subj, obj = DOMAIN_RANGE['scheduled_for']
        assert subj == {'person', 'artifact', 'topic', 'org'}
        assert obj == {'topic'}

    def test_starts_at_domain_range(self):
        subj, obj = DOMAIN_RANGE['starts_at']
        assert subj == {'person', 'artifact', 'topic', 'org'}
        assert obj == {'topic'}

    def test_ends_at_domain_range(self):
        subj, obj = DOMAIN_RANGE['ends_at']
        assert subj == {'person', 'artifact', 'topic', 'org'}
        assert obj == {'topic'}

    def test_recurring_domain_range(self):
        subj, obj = DOMAIN_RANGE['recurring']
        assert subj == {'artifact', 'topic', 'org'}
        assert obj == {'topic'}


# ---------------------------------------------------------------------------
# Test 2: PREDICATE_PRIORITY includes temporal predicates
# ---------------------------------------------------------------------------

class TestTemporalPredicatePriority:
    def test_all_temporal_predicates_have_priority(self):
        for pred in TEMPORAL_PREDICATES:
            assert pred in PREDICATE_PRIORITY, f"{pred} missing from PREDICATE_PRIORITY"

    def test_deadline_is_high_priority(self):
        assert PREDICATE_PRIORITY['deadline'] == 1

    def test_scheduled_starts_ends_are_priority_2(self):
        assert PREDICATE_PRIORITY['scheduled_for'] == 2
        assert PREDICATE_PRIORITY['starts_at'] == 2
        assert PREDICATE_PRIORITY['ends_at'] == 2

    def test_recurring_is_priority_3(self):
        assert PREDICATE_PRIORITY['recurring'] == 3


# ---------------------------------------------------------------------------
# Test 3: Validator accepts valid temporal edges
# ---------------------------------------------------------------------------

class TestValidatorAcceptsTemporal:
    def test_deadline_edge_accepted(self):
        source = "AAAI deadline is March 15"
        patch = {
            'schema_version': 'kg_patch_v1',
            'node_id': 1,
            'assertions': [{
                'assertion_key': 'a1',
                'span_start': 0, 'span_end': len(source),
                'asserted_by': 'user', 'polarity': 'affirm',
                'certainty': 'explicit', 'status': 'active', 'tags': [],
            }],
            'entities': [
                {
                    'entity_key': 'e1', 'entity_type': 'org',
                    'canonical_name': 'AAAI', 'canonical_key': 'org:aaai',
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
                {
                    'entity_key': 'e2', 'entity_type': 'topic',
                    'canonical_name': 'March 15', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
            ],
            'aliases': [],
            'mentions': [
                {
                    'mention_key': 'm1', 'span_start': 0, 'span_end': 4,
                    'surface_text': 'AAAI', 'entity_ref': 'e1',
                    'confidence': 0.95, 'source_assertion': 'a1',
                },
                {
                    'mention_key': 'm2', 'span_start': 17, 'span_end': 25,
                    'surface_text': 'March 15', 'entity_ref': 'e2',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
            ],
            'edges': [{
                'subj_ref': 'e1', 'predicate': 'deadline', 'obj_ref': 'e2',
                'source_assertion': 'a1', 'confidence': 0.9,
            }],
            'notes': None,
        }
        result = validate_patch(
            patch, source, node_id=1,
            topic_entity_ids=set(),
            existing_canonical_keys={},
        )
        assert result.valid
        assert len(result.cleaned_patch['edges']) == 1
        assert result.cleaned_patch['edges'][0]['predicate'] == 'deadline'

    def test_recurring_edge_accepted(self):
        source = "team standup is every Monday at 9am"
        patch = {
            'schema_version': 'kg_patch_v1',
            'node_id': 2,
            'assertions': [{
                'assertion_key': 'a1',
                'span_start': 0, 'span_end': len(source),
                'asserted_by': 'user', 'polarity': 'affirm',
                'certainty': 'explicit', 'status': 'active', 'tags': [],
            }],
            'entities': [
                {
                    'entity_key': 'e1', 'entity_type': 'topic',
                    'canonical_name': 'team standup', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
                {
                    'entity_key': 'e2', 'entity_type': 'topic',
                    'canonical_name': 'Monday at 9am', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
            ],
            'aliases': [],
            'mentions': [
                {
                    'mention_key': 'm1', 'span_start': 0, 'span_end': 13,
                    'surface_text': 'team standup ', 'entity_ref': 'e1',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
                {
                    'mention_key': 'm2', 'span_start': 22, 'span_end': 35,
                    'surface_text': 'Monday at 9am', 'entity_ref': 'e2',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
            ],
            'edges': [{
                'subj_ref': 'e1', 'predicate': 'recurring', 'obj_ref': 'e2',
                'source_assertion': 'a1', 'confidence': 0.9,
            }],
            'notes': None,
        }
        result = validate_patch(
            patch, source, node_id=2,
            topic_entity_ids=set(),
            existing_canonical_keys={},
        )
        assert result.valid
        assert len(result.cleaned_patch['edges']) == 1
        assert result.cleaned_patch['edges'][0]['predicate'] == 'recurring'


# ---------------------------------------------------------------------------
# Test 4: Validator rejects bad domain for temporal predicates
# ---------------------------------------------------------------------------

class TestValidatorRejectsBadDomain:
    def test_person_subject_rejected_for_deadline(self):
        """deadline requires artifact|topic|org subject, not person."""
        source = "John deadline March 15"
        patch = {
            'schema_version': 'kg_patch_v1',
            'node_id': 3,
            'assertions': [{
                'assertion_key': 'a1',
                'span_start': 0, 'span_end': len(source),
                'asserted_by': 'user', 'polarity': 'affirm',
                'certainty': 'explicit', 'status': 'active', 'tags': [],
            }],
            'entities': [
                {
                    'entity_key': 'e1', 'entity_type': 'person',
                    'canonical_name': 'John', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
                {
                    'entity_key': 'e2', 'entity_type': 'topic',
                    'canonical_name': 'March 15', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
            ],
            'aliases': [],
            'mentions': [
                {
                    'mention_key': 'm1', 'span_start': 0, 'span_end': 4,
                    'surface_text': 'John', 'entity_ref': 'e1',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
                {
                    'mention_key': 'm2', 'span_start': 14, 'span_end': 22,
                    'surface_text': 'March 15', 'entity_ref': 'e2',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
            ],
            'edges': [{
                'subj_ref': 'e1', 'predicate': 'deadline', 'obj_ref': 'e2',
                'source_assertion': 'a1', 'confidence': 0.9,
            }],
            'notes': None,
        }
        result = validate_patch(
            patch, source, node_id=3,
            topic_entity_ids=set(),
            existing_canonical_keys={},
        )
        assert result.valid  # patch is valid, edge is stripped
        assert len(result.cleaned_patch['edges']) == 0
        assert any('domain_range_violation' in w for w in result.warnings)

    def test_person_subject_rejected_for_recurring(self):
        """recurring requires artifact|topic|org subject, not person."""
        source = "John recurring Mondays"
        patch = {
            'schema_version': 'kg_patch_v1',
            'node_id': 4,
            'assertions': [{
                'assertion_key': 'a1',
                'span_start': 0, 'span_end': len(source),
                'asserted_by': 'user', 'polarity': 'affirm',
                'certainty': 'explicit', 'status': 'active', 'tags': [],
            }],
            'entities': [
                {
                    'entity_key': 'e1', 'entity_type': 'person',
                    'canonical_name': 'John', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
                {
                    'entity_key': 'e2', 'entity_type': 'topic',
                    'canonical_name': 'Mondays', 'canonical_key': None,
                    'created_by_assertion': 'a1', 'resolution_hint': None,
                },
            ],
            'aliases': [],
            'mentions': [
                {
                    'mention_key': 'm1', 'span_start': 0, 'span_end': 4,
                    'surface_text': 'John', 'entity_ref': 'e1',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
                {
                    'mention_key': 'm2', 'span_start': 15, 'span_end': 21,
                    'surface_text': 'ondays', 'entity_ref': 'e2',
                    'confidence': 0.9, 'source_assertion': 'a1',
                },
            ],
            'edges': [{
                'subj_ref': 'e1', 'predicate': 'recurring', 'obj_ref': 'e2',
                'source_assertion': 'a1', 'confidence': 0.9,
            }],
            'notes': None,
        }
        result = validate_patch(
            patch, source, node_id=4,
            topic_entity_ids=set(),
            existing_canonical_keys={},
        )
        assert result.valid
        assert len(result.cleaned_patch['edges']) == 0


# ---------------------------------------------------------------------------
# Test 5: /kg deadlines command
# ---------------------------------------------------------------------------

class TestKgDeadlinesCommand:
    def test_deadlines_lists_temporal_edges(self):
        conn = _fresh_db()
        eid1 = _add_entity(conn, 'org', 'AAAI', 1, 'org:aaai')
        eid2 = _add_entity(conn, 'topic', 'March 15', 1)
        eid3 = _add_entity(conn, 'topic', 'team standup', 2)
        eid4 = _add_entity(conn, 'topic', 'Mondays 9am', 2)
        _add_edge(conn, eid1, 'deadline', eid2, 1)
        _add_edge(conn, eid3, 'recurring', eid4, 2)
        # Also add a non-temporal edge
        eid5 = _add_entity(conn, 'person', 'Alice', 3, 'person:alice')
        _add_edge(conn, eid5, 'uses', eid3, 3)

        @contextmanager
        def fake_use_conn():
            yield conn

        from episodic.commands.kg import kg_deadlines
        output_lines = []
        with patch.object(typer, 'secho', side_effect=lambda msg, **kw: output_lines.append(msg)):
            with patch('episodic.kg.db_kg.kg_tables_exist', return_value=True):
                with patch('episodic.kg.db_kg._use_conn', fake_use_conn):
                    kg_deadlines()

        text = '\n'.join(output_lines)
        assert 'Temporal edges:' in text
        assert 'AAAI deadline March 15' in text
        assert 'team standup recurring Mondays 9am' in text
        # Non-temporal edge should not appear
        assert 'uses' not in text

    def test_deadlines_empty(self):
        conn = _fresh_db()
        # Only non-temporal edges
        eid1 = _add_entity(conn, 'person', 'Bob', 1, 'person:bob')
        eid2 = _add_entity(conn, 'artifact', 'Neovim', 1, 'artifact:neovim')
        _add_edge(conn, eid1, 'uses', eid2, 1)

        @contextmanager
        def fake_use_conn():
            yield conn

        from episodic.commands.kg import kg_deadlines
        output_lines = []
        with patch.object(typer, 'secho', side_effect=lambda msg, **kw: output_lines.append(msg)):
            with patch('episodic.kg.db_kg.kg_tables_exist', return_value=True):
                with patch('episodic.kg.db_kg._use_conn', fake_use_conn):
                    kg_deadlines()

        text = '\n'.join(output_lines)
        assert 'No temporal edges found.' in text


# ---------------------------------------------------------------------------
# Test 6: Standard scoring — temporal edges scored same as any predicate
# ---------------------------------------------------------------------------

class TestTemporalEdgeScoring:
    def test_score_direct_edges_works_with_temporal(self):
        from episodic.kg.context_source import _score_direct_edges, EdgeFact
        edges = [
            EdgeFact(
                subj_name='AAAI', predicate='deadline', obj_name='March 15',
                source_node_id=100, rank_score=0.0, tags=[],
                subj_entity_id=1, obj_entity_id=2,
            ),
            EdgeFact(
                subj_name='standup', predicate='recurring', obj_name='Mondays',
                source_node_id=101, rank_score=0.0, tags=[],
                subj_entity_id=3, obj_entity_id=4,
            ),
        ]
        from episodic.kg.context_source import compute_prompt_tokens
        tokens = compute_prompt_tokens("AAAI deadline March")
        max_overlap = _score_direct_edges(edges, tokens, {1})
        assert max_overlap > 0
        # deadline edge should have higher score due to token overlap
        assert edges[0].rank_score > edges[1].rank_score
