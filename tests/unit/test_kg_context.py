"""Tests for episodic.kg.context_source — KG read-side context injection."""

import json
import sqlite3
import time

import pytest

from episodic.kg.schema import ensure_kg_schema
from episodic.kg.context_source import (
    MentionDictionary,
    EdgeFact,
    DerivedFact,
    KGContextResult,
    retrieve_neighborhood,
    apply_closure_rules,
    format_kg_context,
    get_kg_context,
    _normalize_surface,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kg_db():
    """In-memory SQLite with KG schema + nodes table + sample data."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    conn.execute("INSERT INTO nodes VALUES (1, 'I use Neovim daily.', 'user')")
    conn.execute("INSERT INTO nodes VALUES (2, 'My wife Sarah lives in Austin.', 'user')")
    conn.execute("INSERT INTO nodes VALUES (3, 'I have a ThinkPad X1 Carbon.', 'user')")
    conn.execute("INSERT INTO nodes VALUES (4, 'The ThinkPad has 32GB RAM.', 'user')")
    conn.execute("INSERT INTO nodes VALUES (5, 'I used to live in Denver.', 'user')")
    ensure_kg_schema(conn)

    # Get user:self entity_id
    user_self_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]

    # Create entities
    entities = [
        # (entity_type, canonical_key, canonical_name, created_node_id)
        ('artifact', None, 'Neovim', 1),
        ('person', None, 'Sarah', 2),
        ('artifact', None, 'Austin', 2),
        ('artifact', None, 'ThinkPad X1 Carbon', 3),
        ('artifact', None, '32GB RAM', 4),
        ('artifact', None, 'Denver', 5),
        ('topic', None, 'Python', 1),
    ]
    entity_ids = {}
    for etype, ckey, cname, node_id in entities:
        conn.execute(
            "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
            "created_node_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (etype, ckey, cname, node_id, time.time()),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        entity_ids[cname] = eid

    # Create aliases
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, span_start, span_end) "
        "VALUES (?, 'vim', 1, 6, 9)",
        (entity_ids['Neovim'],),
    )
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, span_start, span_end) "
        "VALUES (?, 'ThinkPad', 3, 11, 19)",
        (entity_ids['ThinkPad X1 Carbon'],),
    )

    # Create assertions
    assertions = [
        # (source_node_id, span_start, span_end, tags)
        (1, 0, 19, '[]'),           # a1: user uses Neovim
        (2, 0, 30, '[]'),           # a2: user related_to Sarah
        (2, 20, 30, '[]'),          # a3: Sarah located_at Austin
        (3, 0, 29, '[]'),           # a4: user has ThinkPad
        (4, 0, 25, '[]'),           # a5: ThinkPad has 32GB RAM
        (5, 0, 27, '["TIME_PAST"]'),  # a6: user located_at Denver (past)
        (1, 0, 19, None),           # a7: user uses Python -- duplicate span, use different
    ]
    assertion_ids = []
    for i, (nid, ss, se, tags) in enumerate(assertions):
        # Ensure unique (source_node_id, span_start, span_end)
        actual_ss = ss + i * 100  # offset to avoid UNIQUE conflict
        actual_se = se + i * 100
        conn.execute(
            "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
            "asserted_by, polarity, certainty, status, tags) "
            "VALUES (?, ?, ?, 'user', 'affirm', 'explicit', 'active', ?)",
            (nid, actual_ss, actual_se, tags),
        )
        aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        assertion_ids.append(aid)

    # Create edges
    edges = [
        # (subj_entity_id, predicate, obj_entity_id, assertion_id)
        (user_self_id, 'uses', entity_ids['Neovim'], assertion_ids[0]),
        (user_self_id, 'related_to', entity_ids['Sarah'], assertion_ids[1]),
        (entity_ids['Sarah'], 'located_at', entity_ids['Austin'], assertion_ids[2]),
        (user_self_id, 'has', entity_ids['ThinkPad X1 Carbon'], assertion_ids[3]),
        (entity_ids['ThinkPad X1 Carbon'], 'has', entity_ids['32GB RAM'], assertion_ids[4]),
        (user_self_id, 'located_at', entity_ids['Denver'], assertion_ids[5]),
        (user_self_id, 'uses', entity_ids['Python'], assertion_ids[6]),
    ]
    for subj, pred, obj, aid in edges:
        conn.execute(
            "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
            "VALUES (?, ?, ?, ?)",
            (subj, pred, obj, aid),
        )

    # Update high_water_mark
    conn.execute(
        "UPDATE kg_state SET value = '5' WHERE key = 'high_water_mark'"
    )
    conn.commit()

    yield conn, user_self_id, entity_ids
    conn.close()


# ---------------------------------------------------------------------------
# Test 1: Basic entity match + edge retrieval
# ---------------------------------------------------------------------------

def test_basic_entity_match_and_retrieval(kg_db, monkeypatch):
    """Detect 'Neovim' in user text and retrieve its edges."""
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    md = MentionDictionary()
    md.rebuild(conn)
    matches = md.detect_mentions("Tell me about Neovim", max_entities=5)

    assert len(matches) >= 1
    eids = [m[0] for m in matches]
    assert entity_ids['Neovim'] in eids

    # Retrieve edges
    facts = retrieve_neighborhood(
        entity_ids['Neovim'], conn, user_text="Tell me about Neovim"
    )
    assert len(facts) >= 1
    predicates = [f.predicate for f in facts]
    assert 'uses' in predicates


# ---------------------------------------------------------------------------
# Test 2: Alias matching
# ---------------------------------------------------------------------------

def test_alias_matching(kg_db, monkeypatch):
    """'vim' should match Neovim via alias, not create a false match inside 'Neovim'."""
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    md = MentionDictionary()
    md.rebuild(conn)

    # "vim" as standalone should match
    matches = md.detect_mentions("I love vim", max_entities=5)
    assert len(matches) >= 1
    assert entity_ids['Neovim'] in [m[0] for m in matches]

    # "Neovim" should match with longest-match-first (not 'vim' inside 'neovim')
    matches2 = md.detect_mentions("Tell me about Neovim", max_entities=5)
    assert len(matches2) >= 1
    # Should match the canonical, not the alias
    forms = [m[1] for m in matches2]
    assert 'neovim' in forms


# ---------------------------------------------------------------------------
# Test 3: KINSHIP_LOCATION closure
# ---------------------------------------------------------------------------

def test_kinship_location_closure(kg_db, monkeypatch):
    """user:self --related_to--> Sarah, Sarah --located_at--> Austin
    When Sarah is mentioned, derive 'Sarah located_at Austin'.
    """
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    # Get edges for user:self (which includes related_to Sarah)
    facts = retrieve_neighborhood(
        user_self_id, conn, user_text="How is Sarah doing?"
    )

    # Apply closure with Sarah as matched entity
    derived = apply_closure_rules(
        [entity_ids['Sarah']], facts, conn, max_derived=3
    )

    assert len(derived) >= 1
    kinship = [d for d in derived if d.rule == 'KINSHIP_LOCATION']
    assert len(kinship) >= 1
    assert kinship[0].subj_name == 'Sarah'
    assert kinship[0].predicate == 'located_at'
    assert kinship[0].obj_name == 'Austin'


# ---------------------------------------------------------------------------
# Test 4: DEVICE_SPEC closure
# ---------------------------------------------------------------------------

def test_device_spec_closure(kg_db, monkeypatch):
    """user:self --has--> ThinkPad, ThinkPad --has--> 32GB RAM
    When ThinkPad is mentioned, derive 'ThinkPad has 32GB RAM'.
    """
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    facts = retrieve_neighborhood(
        user_self_id, conn, user_text="What about my ThinkPad?"
    )

    derived = apply_closure_rules(
        [entity_ids['ThinkPad X1 Carbon']], facts, conn, max_derived=3
    )

    assert len(derived) >= 1
    device = [d for d in derived if d.rule == 'DEVICE_SPEC']
    assert len(device) >= 1
    assert device[0].subj_name == 'ThinkPad X1 Carbon'
    assert device[0].predicate == 'has'
    assert device[0].obj_name == '32GB RAM'


# ---------------------------------------------------------------------------
# Test 5: Budget enforcement
# ---------------------------------------------------------------------------

def test_budget_enforcement(kg_db, monkeypatch):
    """With a very tight budget, fewer facts should be included."""
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    # Create many facts
    facts = []
    for i in range(20):
        facts.append(EdgeFact(
            subj_name=f'Entity{i}',
            predicate='uses',
            obj_name=f'Tool{i}',
            source_node_id=i,
            rank_score=float(i),
            tags=[],
        ))

    # Very tight budget
    text = format_kg_context(facts, [], budget_tokens=30)
    # Should have header + only a few facts
    lines = text.strip().split('\n') if text else []
    assert len(lines) < 22  # less than 20 facts + header

    # Zero budget
    text_zero = format_kg_context(facts, [], budget_tokens=0)
    assert text_zero == ""


# ---------------------------------------------------------------------------
# Test 6: Insertion ordering (KG before RAG)
# ---------------------------------------------------------------------------

def test_insertion_ordering(kg_db, monkeypatch):
    """KG context should be inserted at insert_pos (after system msgs, before conversation)."""
    conn, user_self_id, entity_ids = kg_db

    cfg = _mock_config({'kg_context': True})
    monkeypatch.setattr('episodic.kg.context_source.config', cfg)

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Tell me about Neovim"},
        {"role": "assistant", "content": "Neovim is great."},
    ]

    md = MentionDictionary()
    md.rebuild(conn)
    matches = md.detect_mentions("Tell me about Neovim", max_entities=5)

    matched_ids = [m[0] for m in matches]
    co_mentioned = set(matched_ids)
    all_facts = []
    for eid, _, _ in matches:
        facts = retrieve_neighborhood(eid, conn, user_text="Tell me about Neovim",
                                       co_mentioned_ids=co_mentioned - {eid})
        all_facts.extend(facts)

    derived = apply_closure_rules(matched_ids, all_facts, conn)
    text = format_kg_context(all_facts, derived, budget_tokens=500)

    if text:
        # Insert at position after system messages
        insert_pos = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "system":
                insert_pos = i
                break
        messages.insert(insert_pos, {"role": "system", "content": text})

    # Verify ordering: system prompt first, KG context second, then user
    assert messages[0]["content"] == "System prompt"
    assert messages[1]["role"] == "system"
    assert "Known facts" in messages[1]["content"]
    assert messages[2]["role"] == "user"


# ---------------------------------------------------------------------------
# Test 7: TIME_PAST filtering
# ---------------------------------------------------------------------------

def test_time_past_filtering(kg_db, monkeypatch):
    """TIME_PAST edges excluded by default, included with past-tense cues."""
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    # Without past-tense cues — should NOT include Denver
    facts_no_past = retrieve_neighborhood(
        user_self_id, conn, user_text="Where do I live?"
    )
    obj_names = [f.obj_name for f in facts_no_past]
    assert 'Denver' not in obj_names

    # With past-tense cues — should include Denver
    facts_with_past = retrieve_neighborhood(
        user_self_id, conn, user_text="Where did I used to live?"
    )
    obj_names_past = [f.obj_name for f in facts_with_past]
    assert 'Denver' in obj_names_past


# ---------------------------------------------------------------------------
# Test 8: Cache invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidation(kg_db, monkeypatch):
    """HWM change should trigger rebuild."""
    conn, user_self_id, entity_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _mock_config())

    md = MentionDictionary()
    status1 = md.rebuild(conn)
    assert status1 == "rebuilt"

    # Second call without HWM change => hit
    status2 = md.rebuild(conn)
    assert status2 == "hit"

    # Change HWM
    conn.execute(
        "UPDATE kg_state SET value = '10' WHERE key = 'high_water_mark'"
    )
    conn.commit()

    status3 = md.rebuild(conn)
    assert status3 == "rebuilt"

    # Verify the dict is still functional after rebuild
    matches = md.detect_mentions("Neovim", max_entities=5)
    assert len(matches) >= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _mock_config:
    """Mock config object for testing."""

    def __init__(self, overrides=None):
        self._data = {
            'kg_context': False,
            'kg_max_entities': 5,
            'kg_max_edges': 5,
            'kg_max_derived': 3,
            'kg_budget': 500,
            'kg_include_past': False,
            'debug': False,
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None):
        return self._data.get(key, default)
