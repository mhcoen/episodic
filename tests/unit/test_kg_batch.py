"""Tests for episodic.kg.batch."""

import sqlite3
import pytest

from episodic.kg.batch import (
    get_high_water_mark, get_skip_list, get_pending_nodes,
    add_to_skiplist, classify_node_intent,
)
from episodic.kg.schema import ensure_kg_schema


@pytest.fixture
def batch_db():
    """In-memory DB with nodes + KG schema."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user',
            is_meta_query INTEGER DEFAULT 0
        )
    """)
    # Insert test nodes
    conn.execute(
        "INSERT INTO nodes VALUES (1, 'Hello there.', 'user', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (2, 'Hi! How can I help?', 'assistant', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (3, 'I use Python daily.', 'user', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (4, 'Python is great.', 'assistant', 0)"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (5, 'I prefer Vim over VS Code.', 'user', 0)"
    )
    ensure_kg_schema(conn)
    conn.commit()
    yield conn
    conn.close()


def test_get_high_water_mark_initial(batch_db):
    """HWM is 0 on fresh DB."""
    hwm = get_high_water_mark(batch_db)
    assert hwm == 0


def test_get_pending_nodes(batch_db):
    """Returns user nodes after HWM, excludes skip list."""
    pending = get_pending_nodes(0, set(), batch_db)
    # Should get user nodes only (1, 3, 5)
    node_ids = [n['node_id'] for n in pending]
    assert 1 in node_ids
    assert 3 in node_ids
    assert 5 in node_ids
    # Assistant nodes should not be included
    assert 2 not in node_ids
    assert 4 not in node_ids


def test_get_pending_nodes_after_hwm(batch_db):
    """Only returns nodes after HWM."""
    pending = get_pending_nodes(3, set(), batch_db)
    node_ids = [n['node_id'] for n in pending]
    assert 1 not in node_ids
    assert 3 not in node_ids
    assert 5 in node_ids


def test_get_pending_nodes_skips(batch_db):
    """Skip list nodes are excluded."""
    pending = get_pending_nodes(0, {3}, batch_db)
    node_ids = [n['node_id'] for n in pending]
    assert 3 not in node_ids
    assert 1 in node_ids
    assert 5 in node_ids


def test_add_to_skiplist(batch_db):
    """Node added to skip list, HWM advanced if stuck."""
    add_to_skiplist(3, reason='test', conn=batch_db)

    skip = get_skip_list(batch_db)
    assert 3 in skip

    # Verify reason stored
    row = batch_db.execute(
        "SELECT reason FROM kg_skiplist WHERE node_id = 3"
    ).fetchone()
    assert row[0] == 'test'


def test_skiplist_respected(batch_db):
    """Skipped nodes not returned by get_pending_nodes."""
    add_to_skiplist(1, conn=batch_db)
    add_to_skiplist(5, conn=batch_db)

    pending = get_pending_nodes(0, get_skip_list(batch_db), batch_db)
    node_ids = [n['node_id'] for n in pending]
    assert 1 not in node_ids
    assert 5 not in node_ids
    assert 3 in node_ids


def test_add_to_skiplist_advances_hwm(batch_db):
    """Adding node to skiplist advances HWM if stuck."""
    # Set HWM to 2 (stuck before node 3)
    batch_db.execute(
        "UPDATE kg_state SET value = '2' WHERE key = 'high_water_mark'"
    )
    batch_db.commit()

    add_to_skiplist(3, conn=batch_db)

    hwm = get_high_water_mark(batch_db)
    assert hwm >= 3


def test_get_skip_list_empty(batch_db):
    """Skip list is empty on fresh DB."""
    skip = get_skip_list(batch_db)
    assert skip == set()


def test_pending_nodes_empty_content(batch_db):
    """Nodes with empty content are skipped."""
    batch_db.execute(
        "INSERT INTO nodes VALUES (6, '', 'user', 0)"
    )
    batch_db.execute(
        "INSERT INTO nodes VALUES (7, '   ', 'user', 0)"
    )
    batch_db.commit()

    pending = get_pending_nodes(0, set(), batch_db)
    node_ids = [n['node_id'] for n in pending]
    assert 6 not in node_ids
    assert 7 not in node_ids


# --- classify_node_intent tests ---

class TestClassifyNodeIntent:
    """Tests for the question vs assertion classifier."""

    # Pure questions → 'question'
    def test_ends_with_question_mark(self):
        assert classify_node_intent('What is quantum computing?') == 'question'

    def test_starts_with_interrogative(self):
        assert classify_node_intent('How does backpropagation work') == 'question'

    def test_can_you_explain(self):
        assert classify_node_intent('Can you explain gradient descent?') == 'question'

    def test_is_there_a_way(self):
        assert classify_node_intent('Is there a way to speed this up?') == 'question'

    def test_tell_me_about(self):
        # Starts with implicit interrogative structure, no ? but no intent marker
        assert classify_node_intent('Tell me about machine learning') == 'assertion'
        # With question mark
        assert classify_node_intent('Tell me about machine learning?') == 'question'

    def test_do_you_know(self):
        assert classify_node_intent('Do you know about transformers?') == 'question'

    # Assertions → 'assertion'
    def test_plain_statement(self):
        assert classify_node_intent('I use Neovim for all my coding.') == 'assertion'

    def test_preference(self):
        assert classify_node_intent('I prefer Python over Java.') == 'assertion'

    def test_ownership(self):
        assert classify_node_intent('My MacBook has 64 gigs of RAM.') == 'assertion'

    # Durable intent markers override question form → 'assertion'
    def test_want_to_overrides_question(self):
        assert classify_node_intent(
            'I want to learn Rust, where should I start?'
        ) == 'assertion'

    def test_need_to_overrides_question(self):
        assert classify_node_intent(
            'I need to find a good database, any ideas?'
        ) == 'assertion'

    def test_looking_for_overrides(self):
        assert classify_node_intent(
            "I'm looking for a standing desk, what do you recommend?"
        ) == 'assertion'

    def test_plan_to_overrides(self):
        assert classify_node_intent(
            'I plan to publish in CL, is that realistic?'
        ) == 'assertion'

    def test_goal_overrides(self):
        assert classify_node_intent(
            'My goal is to finish by June, how should I plan?'
        ) == 'assertion'

    def test_id_like_to_overrides(self):
        assert classify_node_intent(
            "I'd like to learn about compilers, can you help?"
        ) == 'assertion'

    # Edge cases
    def test_empty_string(self):
        assert classify_node_intent('') == 'assertion'

    def test_whitespace_only(self):
        assert classify_node_intent('   ') == 'assertion'

    def test_multiline_question(self):
        assert classify_node_intent(
            'I was reading about transformers.\nHow do they work?'
        ) == 'question'

    def test_command_like(self):
        # Doesn't start with interrogative, no ?
        assert classify_node_intent('Explain quantum computing') == 'assertion'
