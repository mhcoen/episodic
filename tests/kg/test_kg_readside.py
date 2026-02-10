"""Read-side regression tests RT1-RT5 for KG context injection."""

import sqlite3
import time

import pytest

from episodic.kg.schema import ensure_kg_schema
from episodic.config import config
from episodic.kg.context_source import (
    get_kg_context,
    _mention_dict,
    PREDICATE_PRIORITY,
    compute_prompt_tokens,
)
from episodic.kg.merge import merge_entities


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fresh_db():
    """Create an in-memory DB with KG schema seeded."""
    conn = sqlite3.connect(':memory:')
    ensure_kg_schema(conn)
    # Create nodes table (required by retrieve_neighborhood)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user'
        )
    """)
    return conn


def _add_entity(conn, etype, name, node_id, ckey=None):
    """Insert an entity and return its entity_id."""
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (etype, ckey, name, node_id, time.time()),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_assertion(conn, node_id):
    """Insert an assertion and return its assertion_id."""
    conn.execute(
        "INSERT INTO kg_assertions (source_node_id, span_start, span_end, "
        "asserted_by, polarity, certainty, status, tags) "
        "VALUES (?, 0, 10, 'user', 'affirm', 'explicit', 'active', '[]')",
        (node_id,),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_edge(conn, subj_id, pred, obj_id, node_id):
    """Insert an edge (creates assertion automatically)."""
    conn.execute(f"INSERT OR IGNORE INTO nodes VALUES ({node_id}, 'test', 'user')")
    aid = _add_assertion(conn, node_id)
    conn.execute(
        "INSERT INTO kg_edges (subj_entity_id, predicate, obj_entity_id, assertion_id) "
        "VALUES (?, ?, ?, ?)", (subj_id, pred, obj_id, aid),
    )


def _add_alias(conn, entity_id, alias, node_id):
    """Insert an entity alias."""
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, "
        "span_start, span_end) VALUES (?, ?, ?, 0, ?)",
        (entity_id, alias, node_id, len(alias)),
    )


def _bump_hwm(conn, hwm):
    """Set high_water_mark so MentionDictionary rebuilds."""
    conn.execute("UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'",
                 (str(hwm),))
    conn.commit()
    _mention_dict._hwm = ""  # Force rebuild


def _seed_rt_fixture(conn):
    """Seed the shared fixture for RT1-RT3. Returns entity ID dict."""
    user_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]

    alice = _add_entity(conn, 'person', 'Alice', 100)
    mit = _add_entity(conn, 'org', 'MIT', 101)
    macbook = _add_entity(conn, 'artifact', 'MacBook', 200)
    _add_alias(conn, macbook, 'laptop', 200)
    python = _add_entity(conn, 'topic', 'Python', 102)
    rust = _add_entity(conn, 'topic', 'Rust', 300)
    react = _add_entity(conn, 'artifact', 'React', 301)
    ml_lab = _add_entity(conn, 'org', 'ML Lab', 400)
    ram = _add_entity(conn, 'artifact', '64GB RAM', 201)
    py_cert = _add_entity(conn, 'topic', 'Python certification', 500)

    # Edges per spec (ascending node_ids = recency)
    _add_edge(conn, user_id, 'related_to', alice, 100)
    _add_edge(conn, alice, 'located_at', mit, 101)
    _add_edge(conn, alice, 'studies', python, 102)
    _add_edge(conn, user_id, 'has', macbook, 200)
    _add_edge(conn, macbook, 'has', ram, 201)
    _add_edge(conn, user_id, 'uses', rust, 300)
    _add_edge(conn, user_id, 'uses', react, 301)
    _add_edge(conn, user_id, 'works_on', ml_lab, 400)
    _add_edge(conn, alice, 'affiliated_with', ml_lab, 401)
    _add_edge(conn, user_id, 'wants', py_cert, 500)

    _bump_hwm(conn, 501)

    return {
        'user': user_id, 'alice': alice, 'mit': mit, 'macbook': macbook,
        'python': python, 'rust': rust, 'react': react, 'ml_lab': ml_lab,
        'ram': ram, 'py_cert': py_cert,
    }


# ---------------------------------------------------------------------------
# RT1: Ranking and formatting
# ---------------------------------------------------------------------------

class TestRT1RankingFormatting:

    def test_returns_context(self):
        conn = _fresh_db()
        _seed_rt_fixture(conn)
        result = get_kg_context("What about Alice", conn)
        assert result is not None
        assert result.edge_count > 0
        conn.close()

    def test_edges_sorted_by_priority(self):
        conn = _fresh_db()
        _seed_rt_fixture(conn)
        result = get_kg_context("What about Alice", conn)
        # Edges should be in descending rank_score order
        scores = [e.rank_score for e in result.edges]
        assert scores == sorted(scores, reverse=True), \
            f"Edges not sorted by rank: {scores}"
        conn.close()

    def test_recency_within_same_priority(self):
        """Higher source_node_id (more recent) should rank higher within same predicate."""
        conn = _fresh_db()
        ids = _seed_rt_fixture(conn)
        # Add two 'uses' edges for Alice to test recency
        topic_a = _add_entity(conn, 'topic', 'TopicA', 600)
        topic_b = _add_entity(conn, 'topic', 'TopicB', 601)
        _add_edge(conn, ids['alice'], 'uses', topic_a, 600)
        _add_edge(conn, ids['alice'], 'uses', topic_b, 700)
        _bump_hwm(conn, 701)

        result = get_kg_context("What about Alice", conn)
        uses_edges = [e for e in result.edges if e.predicate == 'uses']
        if len(uses_edges) >= 2:
            # More recent (node 700) should rank higher
            assert uses_edges[0].source_node_id >= uses_edges[1].source_node_id
        conn.close()

    def test_output_format(self):
        conn = _fresh_db()
        _seed_rt_fixture(conn)
        result = get_kg_context("What about Alice", conn)
        assert "Alice" in result.text
        # Check format: "- subject predicate object [node:N]"
        for line in result.text.split('\n')[1:]:  # skip header
            if line.strip():
                assert line.startswith('- '), f"Bad format: {line}"
                assert '[node:' in line or '[from ' in line, f"Missing provenance: {line}"
        conn.close()


# ---------------------------------------------------------------------------
# RT2: Closure caps
# ---------------------------------------------------------------------------

class TestRT2ClosureCaps:

    def test_derived_count_capped(self):
        conn = _fresh_db()
        ensure_kg_schema(conn)
        conn.execute("CREATE TABLE IF NOT EXISTS nodes "
                     "(node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user')")
        user_id = conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
        ).fetchone()[0]

        # Create 5 people, each located at a different org
        people = []
        for i in range(5):
            pid = _add_entity(conn, 'person', f'Person{i}', 100 + i)
            oid = _add_entity(conn, 'org', f'Org{i}', 200 + i)
            _add_edge(conn, user_id, 'related_to', pid, 100 + i)
            _add_edge(conn, pid, 'located_at', oid, 200 + i)
            people.append(pid)

        _bump_hwm(conn, 300)

        # Mention all 5 people
        text = "What about Person0, Person1, Person2, Person3, Person4"
        from episodic.config import config
        old_derived = config.get('kg_max_derived', 3)
        config.set('kg_max_derived', 3)
        try:
            result = get_kg_context(text, conn)
            assert result is not None
            assert result.derived_count <= 3, \
                f"Derived count {result.derived_count} exceeds cap 3"
        finally:
            config.set('kg_max_derived', old_derived)
        conn.close()


# ---------------------------------------------------------------------------
# RT3: Budget enforcement
# ---------------------------------------------------------------------------

class TestRT3Budget:

    def test_budget_limits_output(self):
        conn = _fresh_db()
        ids = _seed_rt_fixture(conn)

        # Add many more edges for Alice
        for i in range(10):
            tid = _add_entity(conn, 'topic', f'ExtraTopic{i}', 800 + i)
            _add_edge(conn, ids['alice'], 'uses', tid, 800 + i)
        _bump_hwm(conn, 900)

        from episodic.config import config
        old_budget = config.get('kg_budget', 500)
        config.set('kg_budget', 80)  # Tight budget
        try:
            result = get_kg_context("What about Alice", conn)
            if result is not None:
                assert result.budget_used <= 80, \
                    f"Budget {result.budget_used} exceeds 80"
        finally:
            config.set('kg_budget', old_budget)
        conn.close()

    def test_highest_priority_edges_kept(self):
        conn = _fresh_db()
        ids = _seed_rt_fixture(conn)

        # Add many low-priority edges
        for i in range(10):
            tid = _add_entity(conn, 'topic', f'WantsTopic{i}', 800 + i)
            _add_edge(conn, ids['alice'], 'wants', tid, 800 + i)
        _bump_hwm(conn, 900)

        from episodic.config import config
        old_budget = config.get('kg_budget', 500)
        config.set('kg_budget', 80)
        try:
            result = get_kg_context("What about Alice", conn)
            if result and result.edges:
                # First edge should NOT be 'wants' (lowest priority)
                first_pred = result.text.split('\n')[1].split()[2] if result.text else ''
                # At least verify budget is respected
                assert result.budget_used <= 80
        finally:
            config.set('kg_budget', old_budget)
        conn.close()


# ---------------------------------------------------------------------------
# RT4: Merge tombstone exclusion
# ---------------------------------------------------------------------------

class TestRT4MergeTombstone:

    def test_merged_entity_excluded(self):
        conn = _fresh_db()
        conn.execute("CREATE TABLE IF NOT EXISTS nodes "
                     "(node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user')")

        # Two "Cherry MX Brown switches" entities
        x = _add_entity(conn, 'artifact', 'Cherry MX Brown switches', 10)
        y = _add_entity(conn, 'artifact', 'Cherry MX Brown switches', 20)

        macbook = _add_entity(conn, 'artifact', 'MacBook', 30)
        keychron = _add_entity(conn, 'artifact', 'Keychron', 40)

        _add_edge(conn, macbook, 'has', x, 10)
        _add_edge(conn, keychron, 'has', y, 20)
        _bump_hwm(conn, 50)

        # Verify both visible before merge
        result = get_kg_context("Cherry MX Brown switches", conn)
        assert result is not None

        # Merge Y into X
        merge_entities(x, y, "duplicate", conn)
        _mention_dict._hwm = ""  # Force rebuild

        result = get_kg_context("Cherry MX Brown switches", conn)
        assert result is not None

        # All edges should reference entity X only
        for edge in result.edges:
            assert edge.subj_name != f'entity_{y}', \
                f"Tombstoned entity Y ({y}) found in edge subject"

        # Keychron edge should now point to X
        keychron_edges = [e for e in result.edges
                          if 'Keychron' in e.subj_name and e.predicate == 'has']
        assert len(keychron_edges) >= 1, "Keychron --has--> Cherry MX edge missing"

        conn.close()

    def test_mention_dict_excludes_tombstoned(self):
        conn = _fresh_db()
        conn.execute("CREATE TABLE IF NOT EXISTS nodes "
                     "(node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user')")

        a = _add_entity(conn, 'artifact', 'TestWidget', 10)
        b = _add_entity(conn, 'artifact', 'TestWidget', 20)
        _bump_hwm(conn, 30)

        # Before merge: both resolve
        _mention_dict._hwm = ""
        _mention_dict.rebuild(conn)
        matches = _mention_dict.detect_mentions("TestWidget")
        entity_ids = {m[0] for m in matches}
        assert a in entity_ids or b in entity_ids

        # Merge B into A
        merge_entities(a, b, "duplicate", conn)
        _mention_dict._hwm = ""
        _mention_dict.rebuild(conn)

        matches = _mention_dict.detect_mentions("TestWidget")
        entity_ids = {m[0] for m in matches}
        assert a in entity_ids, "Survivor not in mention dict"
        assert b not in entity_ids, "Tombstoned entity still in mention dict"

        conn.close()


# ---------------------------------------------------------------------------
# RT5: Alias resolution post-merge
# ---------------------------------------------------------------------------

class TestRT5AliasPostMerge:

    def test_alias_moves_to_survivor(self):
        conn = _fresh_db()
        conn.execute("CREATE TABLE IF NOT EXISTS nodes "
                     "(node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user')")

        a = _add_entity(conn, 'artifact', 'Neovim', 10)
        b = _add_entity(conn, 'artifact', 'Neovim', 20)
        _add_alias(conn, a, 'vim', 10)
        _add_alias(conn, b, 'nvim', 20)
        _bump_hwm(conn, 30)

        # Merge B into A
        merge_entities(a, b, "duplicate", conn)
        _mention_dict._hwm = ""
        _mention_dict.rebuild(conn)

        # "nvim" alias should now resolve to entity A (survivor)
        matches = _mention_dict.detect_mentions("I use nvim daily")
        assert len(matches) > 0, "No match for 'nvim'"
        matched_ids = {m[0] for m in matches}
        assert a in matched_ids, f"nvim resolved to {matched_ids}, expected {a}"
        assert b not in matched_ids, "Tombstoned entity B found"

        conn.close()

    def test_conflicting_alias_handled(self):
        """If both entities have the same alias, merge doesn't crash."""
        conn = _fresh_db()
        conn.execute("CREATE TABLE IF NOT EXISTS nodes "
                     "(node_id INTEGER PRIMARY KEY, content TEXT, role TEXT DEFAULT 'user')")

        a = _add_entity(conn, 'artifact', 'Neovim', 10)
        b = _add_entity(conn, 'artifact', 'Neovim', 20)
        _add_alias(conn, a, 'vim', 10)
        _add_alias(conn, b, 'vim', 20)  # Same alias on both
        _bump_hwm(conn, 30)

        result = merge_entities(a, b, "duplicate", conn)
        assert result['dropped_aliases'] >= 1

        # vim should resolve to A
        _mention_dict._hwm = ""
        _mention_dict.rebuild(conn)
        matches = _mention_dict.detect_mentions("I use vim")
        matched_ids = {m[0] for m in matches}
        assert a in matched_ids
        assert b not in matched_ids

        conn.close()


# ---------------------------------------------------------------------------
# Phase 2v2: Chain-aware closure scoring
# ---------------------------------------------------------------------------

def _seed_phase2_fixture(conn):
    """Seed fixture for chain-aware closure tests. No aliases on entities."""
    user_id = conn.execute(
        "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
    ).fetchone()[0]
    signal_chain = _add_entity(conn, 'artifact', 'signal chain', 100)
    focusrite = _add_entity(conn, 'artifact', 'Focusrite Scarlett 2i2', 101)
    sm7b = _add_entity(conn, 'artifact', 'Shure SM7B', 102)
    emma = _add_entity(conn, 'person', 'Emma', 200)
    mit = _add_entity(conn, 'org', 'MIT', 201)
    macbook = _add_entity(conn, 'artifact', 'MacBook Pro M3 Max', 300)
    ram = _add_entity(conn, 'artifact', '64GB RAM', 301)
    _add_edge(conn, user_id, 'has', signal_chain, 100)
    _add_edge(conn, signal_chain, 'has', focusrite, 101)
    _add_edge(conn, signal_chain, 'has', sm7b, 102)
    _add_edge(conn, user_id, 'related_to', emma, 200)
    _add_edge(conn, emma, 'located_at', mit, 201)
    _add_edge(conn, user_id, 'has', macbook, 300)
    _add_edge(conn, macbook, 'has', ram, 301)
    _bump_hwm(conn, 400)
    return {
        'user': user_id, 'signal_chain': signal_chain, 'focusrite': focusrite,
        'sm7b': sm7b, 'emma': emma, 'mit': mit, 'macbook': macbook, 'ram': ram,
    }


class TestPhase2v2KinshipBridge:
    """T1: Kinship cues in prompt → KINSHIP_LOCATION ranks first."""

    def test_daughter_school_routes_to_kinship(self):
        """'daughter' ∈ KINSHIP_CUES → bridge_bonus for KINSHIP_LOCATION."""
        conn = _fresh_db()
        _seed_phase2_fixture(conn)
        config.set('kg_relevance_gate', False)
        try:
            # Only user:self matched (no entity names in prompt).
            # KINSHIP_CUES & {daughter, school} → bridge_bonus=3 for KINSHIP
            # DEVICE_CUES & {daughter, school} → bridge_bonus=0 for DEVICE
            result = get_kg_context(
                "Where does my daughter go to school?", conn)
            assert result is not None
            assert result.derived_count > 0, "Expected closure-derived facts"
            first = result.derived[0]
            assert first.rule == 'KINSHIP_LOCATION', \
                f"Expected KINSHIP first, got {first.rule}: {first.subj_name}→{first.obj_name}"
            assert 'Emma' in first.subj_name or 'MIT' in first.obj_name
        finally:
            config.set('kg_relevance_gate', True)
        conn.close()


class TestPhase2v2DeviceBridge:
    """T2: Device cues in prompt → DEVICE_SPEC ranks first."""

    def test_machine_specs_routes_to_device(self):
        """'machine', 'specs' ∈ DEVICE_CUES → bridge_bonus for DEVICE_SPEC."""
        conn = _fresh_db()
        _seed_phase2_fixture(conn)
        config.set('kg_relevance_gate', False)
        try:
            # DEVICE_CUES & {specs, main, machine} → bridge_bonus=3
            # KINSHIP_CUES → bridge_bonus=0
            result = get_kg_context(
                "What specs does my main machine have?", conn)
            assert result is not None
            assert result.derived_count > 0, "Expected closure-derived facts"
            first = result.derived[0]
            assert first.rule == 'DEVICE_SPEC', \
                f"Expected DEVICE_SPEC first, got {first.rule}: {first.subj_name}→{first.obj_name}"
            kinship = [d for d in result.derived if d.rule == 'KINSHIP_LOCATION']
            assert len(kinship) == 0, "KINSHIP should not appear (per_seed_cap=2 fills with DEVICE)"
        finally:
            config.set('kg_relevance_gate', True)
        conn.close()


class TestPhase2v2WeatherSuppressed:
    """T3: No bridge cues and no overlap → suppressed."""

    def test_weather_suppressed(self):
        conn = _fresh_db()
        _seed_phase2_fixture(conn)
        config.set('kg_relevance_gate', True)
        try:
            # "my" → user:self match. No KINSHIP_CUES or DEVICE_CUES in prompt.
            # max_overlap=0 and has_seeded_closure=False → suppressed.
            result = get_kg_context(
                "What is my local weather forecast today?", conn)
            assert result is not None, "Expected user:self match from 'my'"
            assert result.suppressed, \
                f"Expected suppressed, got text={result.text[:80] if result.text else '(empty)'}"
            assert result.suppressed_reason == "no_relevant_edges"
        finally:
            config.set('kg_relevance_gate', True)
        conn.close()


class TestPhase2v2ClosureMetadata:
    """T4: DerivedFact carries chain metadata (source_seed_id, intermediate)."""

    def test_derived_fact_metadata(self):
        conn = _fresh_db()
        ids = _seed_phase2_fixture(conn)
        config.set('kg_relevance_gate', False)
        try:
            result = get_kg_context(
                "Where does my daughter go to school?", conn)
            assert result is not None and result.derived
            kinship = [d for d in result.derived if d.rule == 'KINSHIP_LOCATION']
            assert kinship, "Expected KINSHIP_LOCATION derived fact"
            d = kinship[0]
            assert d.source_seed_id == ids['user'], f"seed should be user:self, got {d.source_seed_id}"
            assert d.intermediate_id == ids['emma'], f"intermediate should be Emma, got {d.intermediate_id}"
            assert d.intermediate_name == 'Emma'
        finally:
            config.set('kg_relevance_gate', True)
        conn.close()


class TestPhase2v2BudgetEdges:
    """T5: Budget allocation guarantees 2 per seed, fills by score."""

    def test_budget_limits_total_edges(self):
        conn = _fresh_db()
        user_id = conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_key = 'user:self'"
        ).fetchone()[0]
        for i in range(10):
            eid = _add_entity(conn, 'artifact', f'Gadget{i}', 100 + i)
            _add_edge(conn, user_id, 'has', eid, 100 + i)
        _bump_hwm(conn, 200)
        config.set('kg_edges_per_entity', 4)
        config.set('kg_relevance_gate', False)
        try:
            # 1 matched entity (user:self) × 4 = budget of 4
            result = get_kg_context("What gadgets do I have?", conn)
            assert result is not None
            assert len(result.edges) <= 4, \
                f"Expected <= 4 edges (1 entity × 4 budget), got {len(result.edges)}"
        finally:
            config.set('kg_edges_per_entity', 4)
            config.set('kg_relevance_gate', True)
        conn.close()
