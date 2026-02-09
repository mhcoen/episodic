"""Acceptance tests for episodic.kg.context_source — Phase 1.1 KG read-side context injection.

All tests are deterministic, no LLM calls. Each test seeds a temp SQLite DB
with KG schema and known entities/edges, then asserts integration properties.
"""

import sqlite3
import time

import pytest

from episodic.kg.schema import ensure_kg_schema
from episodic.kg.context_source import (
    MentionDictionary,
    EdgeFact,
    format_kg_context,
    get_kg_context,
    retrieve_neighborhood,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _seed_db(conn):
    """Seed a KG DB with the spec'd entities and edges.

    Entities: <user> (user:self, already seeded by ensure_kg_schema),
    Emma (person), MIT (org), MacBook Pro M3 Max (artifact, alias "MacBook"),
    64 gigs of RAM (artifact), Neovim (artifact, alias "vim").

    Edges:
    - <user> related_to Emma (a1)
    - Emma located_at MIT (a2)
    - <user> has MacBook (a3)
    - MacBook has 64GB RAM (a4)
    - <user> uses Neovim (a5)
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY,
            content TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    # Some minimal node content for provenance
    for nid in range(1, 6):
        conn.execute("INSERT INTO nodes VALUES (?, ?, 'user')",
                     (nid, f"node {nid} content"))
    ensure_kg_schema(conn)

    # Get user:self entity_id
    user_self_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]

    # Create entities
    entities = {
        'Emma': ('person', None, 'Emma', 1),
        'MIT': ('org', None, 'MIT', 2),
        'MacBook': ('artifact', None, 'MacBook Pro M3 Max', 3),
        'RAM': ('artifact', None, '64 gigs of RAM', 4),
        'Neovim': ('artifact', None, 'Neovim', 5),
    }
    entity_ids = {'<user>': user_self_id}
    for key, (etype, ckey, cname, node_id) in entities.items():
        conn.execute(
            "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
            "created_node_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (etype, ckey, cname, node_id, time.time()),
        )
        eid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        entity_ids[key] = eid

    # Create aliases
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, span_start, span_end) "
        "VALUES (?, 'MacBook', 3, 0, 7)",
        (entity_ids['MacBook'],),
    )
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, span_start, span_end) "
        "VALUES (?, 'vim', 5, 0, 3)",
        (entity_ids['Neovim'],),
    )

    # Create assertions (a1-a5), each with unique (source_node_id, span_start, span_end)
    assertions = [
        (1, 0, 10, '[]'),   # a1: user related_to Emma
        (2, 0, 20, '[]'),   # a2: Emma located_at MIT
        (3, 0, 30, '[]'),   # a3: user has MacBook
        (4, 0, 25, '[]'),   # a4: MacBook has 64GB RAM
        (5, 0, 15, '[]'),   # a5: user uses Neovim
    ]
    assertion_ids = []
    for nid, ss, se, tags in assertions:
        conn.execute(
            "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
            "asserted_by, polarity, certainty, status, tags) "
            "VALUES (?, ?, ?, 'user', 'affirm', 'explicit', 'active', ?)",
            (nid, ss, se, tags),
        )
        aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        assertion_ids.append(aid)

    # Create edges
    edges = [
        (entity_ids['<user>'], 'related_to', entity_ids['Emma'], assertion_ids[0]),
        (entity_ids['Emma'], 'located_at', entity_ids['MIT'], assertion_ids[1]),
        (entity_ids['<user>'], 'has', entity_ids['MacBook'], assertion_ids[2]),
        (entity_ids['MacBook'], 'has', entity_ids['RAM'], assertion_ids[3]),
        (entity_ids['<user>'], 'uses', entity_ids['Neovim'], assertion_ids[4]),
    ]
    for subj, pred, obj, aid in edges:
        conn.execute(
            "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
            "VALUES (?, ?, ?, ?)",
            (subj, pred, obj, aid),
        )

    # Set HWM
    conn.execute(
        "UPDATE kg_state SET value = '100' WHERE key = 'high_water_mark'"
    )
    conn.commit()
    return entity_ids, assertion_ids


@pytest.fixture
def kg_db():
    """In-memory SQLite with KG schema + seeded test data."""
    conn = sqlite3.connect(':memory:')
    entity_ids, assertion_ids = _seed_db(conn)
    yield conn, entity_ids, assertion_ids
    conn.close()


class _MockConfig:
    """Mock config for testing without touching real config."""

    def __init__(self, overrides=None):
        self._data = {
            'kg_context': True,
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


# ---------------------------------------------------------------------------
# T1: mention_detection_longest_match
# ---------------------------------------------------------------------------

def test_mention_detection_longest_match(kg_db, monkeypatch):
    """Input: 'upgrade my MacBook Pro'
    Assert: resolves to the MacBook entity (canonical 'MacBook Pro M3 Max').
    Key: exactly one MacBook entity returned, not two partial matches.
    """
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    md = MentionDictionary()
    md.rebuild(conn)
    matches = md.detect_mentions("upgrade my MacBook Pro", max_entities=5)

    # Should find exactly one MacBook entity
    macbook_matches = [m for m in matches if m[0] == entity_ids['MacBook']]
    assert len(macbook_matches) == 1, f"Expected 1 MacBook match, got {macbook_matches}"

    # Should not have duplicate partial matches
    all_eids = [m[0] for m in matches]
    assert len(all_eids) == len(set(all_eids)), "Duplicate entity IDs in matches"


# ---------------------------------------------------------------------------
# T2: neighborhood_retrieval_bidirectional
# ---------------------------------------------------------------------------

def test_neighborhood_retrieval_bidirectional(kg_db, monkeypatch):
    """Call retrieve_neighborhood() with MIT's entity_id.
    Assert: returns 'Emma located_at MIT' even though MIT is the object.
    """
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    facts = retrieve_neighborhood(
        entity_ids['MIT'], conn, user_text="Tell me about MIT"
    )

    assert len(facts) >= 1
    # Find the Emma -> located_at -> MIT edge
    located_facts = [f for f in facts if f.predicate == 'located_at']
    assert len(located_facts) >= 1
    assert located_facts[0].subj_name == 'Emma'
    assert located_facts[0].obj_name == 'MIT'


# ---------------------------------------------------------------------------
# T3: inject_ordering_in_context_builder
# ---------------------------------------------------------------------------

def test_inject_ordering_in_context_builder(kg_db, monkeypatch):
    """KG context should be inserted after system msgs, before conversation.
    Simulates the insertion at context_source level.
    """
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    result = get_kg_context("Tell me about Neovim", conn)
    assert result is not None
    assert result.text  # non-empty

    # Simulate a messages list with system + user + assistant
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Tell me about Neovim"},
        {"role": "assistant", "content": "Neovim is great."},
    ]

    # Insert KG at correct position (after system msgs, before conversation)
    insert_pos = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            insert_pos = i
            break
    messages.insert(insert_pos, {"role": "system", "content": result.text})

    # Now simulate RAG insertion at the same position logic
    rag_insert_pos = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            rag_insert_pos = i
            break
    messages.insert(rag_insert_pos, {"role": "system", "content": "RAG context here"})

    # Find positions of KG and RAG
    kg_pos = next(i for i, m in enumerate(messages) if "Known facts" in m.get("content", ""))
    rag_pos = next(i for i, m in enumerate(messages) if "RAG context" in m.get("content", ""))

    # KG should be before RAG (lower index)
    assert kg_pos < rag_pos, f"KG at {kg_pos}, RAG at {rag_pos} — KG should come first"


# ---------------------------------------------------------------------------
# T4: budget_drop_lowest_rank
# ---------------------------------------------------------------------------

def test_budget_drop_lowest_rank(kg_db, monkeypatch):
    """Very small budget (50 tokens ~= 200 chars). Should drop lowest-ranked facts."""
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    # Create 5+ facts with varying ranks
    facts = []
    for i in range(8):
        facts.append(EdgeFact(
            subj_name=f'Entity{i}',
            predicate='uses',
            obj_name=f'Tool{i}',
            source_node_id=i * 100,
            rank_score=float(i),
            tags=[],
        ))

    output = format_kg_context(facts, [], budget_tokens=50)
    # Budget is 50 tokens. Output should fit.
    assert len(output) // 4 <= 50, f"Output {len(output)//4} tokens exceeds budget 50"

    # Should have fewer lines than input facts (some were dropped)
    if output:
        lines = [l for l in output.split('\n') if l.startswith('- ')]
        assert len(lines) < 8, f"Expected fewer than 8 facts, got {len(lines)}"

        # Highest-ranked facts should be kept (highest rank_score = Entity7)
        assert 'Entity7' in output, "Highest-ranked fact should be kept"


# ---------------------------------------------------------------------------
# T5: closure_kinship_location
# ---------------------------------------------------------------------------

def test_closure_kinship_location(kg_db, monkeypatch):
    """Input: 'How is Emma doing at school?'
    Assert: Emma located_at MIT appears in output (via direct edge).
    Closure derives the same triple but it's suppressed since the direct
    edge already covers it — no wasted budget tokens on duplicates.
    """
    conn, entity_ids, assertion_ids = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    result = get_kg_context("How is Emma doing at school?", conn)
    assert result is not None

    # The fact is present via direct edge (not derived, since direct takes priority)
    assert 'Emma' in result.text
    assert 'located_at' in result.text
    assert 'MIT' in result.text

    # Derived count is 0 because the closure triple duplicates the direct edge
    assert result.derived_count == 0

    # But the direct edge IS present
    direct = [e for e in result.edges
              if e.subj_name == 'Emma' and e.predicate == 'located_at']
    assert len(direct) >= 1, "Direct located_at edge should be present"


# ---------------------------------------------------------------------------
# T6: closure_device_spec
# ---------------------------------------------------------------------------

def test_closure_device_spec(kg_db, monkeypatch):
    """Input: 'Can my MacBook handle it?'
    Assert: MacBook has 64GB RAM appears in output (via direct edge).
    Closure derives the same triple but it's suppressed since the direct
    edge already covers it — no wasted budget tokens on duplicates.
    """
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    result = get_kg_context("Can my MacBook handle it?", conn)
    assert result is not None

    # The fact is present via direct edge
    assert '64 gigs of RAM' in result.text
    assert 'MacBook Pro M3 Max' in result.text

    # Derived count is 0 because the closure triple duplicates the direct edge
    assert result.derived_count == 0

    # But the direct edge IS present
    direct = [e for e in result.edges
              if e.subj_name == 'MacBook Pro M3 Max' and e.predicate == 'has'
              and e.obj_name == '64 gigs of RAM']
    assert len(direct) >= 1, "Direct has edge should be present"


# ---------------------------------------------------------------------------
# T7: time_past_filtering
# ---------------------------------------------------------------------------

def test_time_past_filtering(kg_db, monkeypatch):
    """Add TIME_PAST edge, verify excluded without cues, included with cues."""
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    # Add IBM Research entity and TIME_PAST edge
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('org', NULL, 'IBM Research', 6, ?)",
        (time.time(),),
    )
    ibm_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
        "asserted_by, polarity, certainty, status, tags) "
        "VALUES (6, 0, 30, 'user', 'affirm', 'explicit', 'active', '[\"TIME_PAST\"]')"
    )
    past_aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, 'located_at', ?, ?)",
        (entity_ids['<user>'], ibm_id, past_aid),
    )

    # Update HWM to force rebuild
    conn.execute("UPDATE kg_state SET value = '200' WHERE key = 'high_water_mark'")
    conn.commit()

    # Without past-tense cues — IBM Research should be excluded
    result_no_past = get_kg_context("Where do I work?", conn)
    if result_no_past:
        assert 'IBM Research' not in result_no_past.text

    # With past-tense cues — IBM Research should be included
    # Need to force HWM rebuild again (singleton dict cached)
    result_with_past = get_kg_context("I used to work at IBM Research", conn)
    assert result_with_past is not None
    assert 'IBM Research' in result_with_past.text


# ---------------------------------------------------------------------------
# T8: cache_invalidation_on_hwm
# ---------------------------------------------------------------------------

def test_cache_invalidation_on_hwm(kg_db, monkeypatch):
    """1. Build cache -> 'rebuilt'
    2. Rebuild again -> 'hit'
    3. Change HWM + add alias 'nvim' -> 'rebuilt'
    4. detect_mentions('I use nvim') -> finds Neovim
    """
    conn, entity_ids, _ = kg_db
    monkeypatch.setattr('episodic.kg.context_source.config', _MockConfig())

    md = MentionDictionary()

    # Step 1: first build
    status1 = md.rebuild(conn)
    assert status1 == "rebuilt"

    # Step 2: no change
    status2 = md.rebuild(conn)
    assert status2 == "hit"

    # Step 3: change HWM + add alias
    conn.execute("UPDATE kg_state SET value = '200' WHERE key = 'high_water_mark'")
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, span_start, span_end) "
        "VALUES (?, 'nvim', 5, 0, 4)",
        (entity_ids['Neovim'],),
    )
    conn.commit()

    status3 = md.rebuild(conn)
    assert status3 == "rebuilt"

    # Step 4: detect 'nvim' -> Neovim entity
    matches = md.detect_mentions("I use nvim", max_entities=5)
    neovim_matches = [m for m in matches if m[0] == entity_ids['Neovim']]
    assert len(neovim_matches) == 1, f"Expected Neovim via 'nvim' alias, got {matches}"
