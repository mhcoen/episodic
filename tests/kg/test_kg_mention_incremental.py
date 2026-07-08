"""
Tests for MentionDictionary incremental refresh.

The mention dictionary must stay correct while only doing incremental work on
the common per-turn path (high-water mark advances as new entities/aliases are
added), and fall back to a full rebuild whenever membership can shrink or shift
(merge, rollback/reset, or a new curation).
"""

import time
import sqlite3

import pytest

from episodic.kg.schema import ensure_kg_schema
from episodic.kg.context_source import MentionDictionary


@pytest.fixture
def kg_conn():
    conn = sqlite3.connect(":memory:")
    ensure_kg_schema(conn)
    conn.commit()
    yield conn
    conn.close()


def _add_entity(conn, name, node_id=1):
    conn.execute(
        "INSERT INTO kg_entities (entity_type, canonical_key, canonical_name, "
        "created_node_id, created_at) VALUES ('artifact', NULL, ?, ?, ?)",
        (name, node_id, time.time()),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_alias(conn, entity_id, alias, node_id=1):
    conn.execute(
        "INSERT INTO kg_entity_aliases (entity_id, alias, source_node_id, "
        "span_start, span_end) VALUES (?, ?, ?, 0, 1)",
        (entity_id, alias, node_id),
    )
    conn.commit()


def _set_hwm(conn, value):
    conn.execute("UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'", (str(value),))
    conn.commit()


def _set_merge_epoch(conn, value):
    conn.execute(
        "INSERT OR REPLACE INTO kg_state (key, value) VALUES ('merge_epoch', ?)",
        (str(value),),
    )
    conn.commit()


def _ids(md, text):
    return {eid for eid, _, _ in md.detect_mentions(text, max_entities=10)}


class TestIncrementalGrowth:
    def test_new_entity_picked_up_after_hwm_advance(self, kg_conn):
        _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)

        md = MentionDictionary()
        assert md.rebuild(kg_conn) == "rebuilt"
        assert _ids(md, "I love Neovim")  # found

        # A new entity is added and the HWM advances (the realtime path).
        emacs_id = _add_entity(kg_conn, "Emacs", node_id=2)
        _set_hwm(kg_conn, 2)

        assert md.rebuild(kg_conn) == "rebuilt"
        assert emacs_id in _ids(md, "switching to Emacs")
        # The previously loaded entity is still present.
        assert _ids(md, "I love Neovim")

    def test_incremental_only_queries_new_rows(self, kg_conn):
        _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)
        md = MentionDictionary()
        md.rebuild(kg_conn)

        _add_entity(kg_conn, "Emacs", node_id=2)
        _set_hwm(kg_conn, 2)

        executed = []

        class RecordingConn:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args):
                executed.append(sql)
                return self._real.execute(sql, *args)

        md.rebuild(RecordingConn(kg_conn))

        # The incremental entity/alias fetches are bounded by id, not full scans.
        assert any("entity_id >" in s for s in executed)
        assert any("alias_id >" in s for s in executed)
        # No unbounded entity/alias scan ran on the incremental path.
        assert not any(
            s.strip().endswith("FROM kg_entities WHERE merged_into_entity_id IS NULL")
            for s in executed
        )

    def test_new_alias_picked_up_incrementally(self, kg_conn):
        nid = _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)
        md = MentionDictionary()
        md.rebuild(kg_conn)
        assert nid not in _ids(md, "using vim today")

        _add_alias(kg_conn, nid, "vim", node_id=2)
        _set_hwm(kg_conn, 2)
        md.rebuild(kg_conn)
        assert nid in _ids(md, "using vim today")


class TestFullRebuildTriggers:
    def test_merge_drops_entity(self, kg_conn):
        a = _add_entity(kg_conn, "Neovim")
        b = _add_entity(kg_conn, "NVIM", node_id=2)
        _set_hwm(kg_conn, 2)
        md = MentionDictionary()
        md.rebuild(kg_conn)
        assert b in _ids(md, "NVIM rocks")

        # Merge b into a and bump merge_epoch (as merge.py does).
        kg_conn.execute(
            "UPDATE kg_entities SET merged_into_entity_id = ? WHERE entity_id = ?",
            (a, b),
        )
        _set_merge_epoch(kg_conn, time.time())

        assert md.rebuild(kg_conn) == "rebuilt"
        # The merged-away entity is gone; the survivor remains.
        assert b not in _ids(md, "NVIM rocks")
        assert a in _ids(md, "Neovim rocks")

    def test_rollback_hwm_decrease_drops_deleted(self, kg_conn):
        a = _add_entity(kg_conn, "Neovim")
        b = _add_entity(kg_conn, "Emacs", node_id=5)
        _set_hwm(kg_conn, 5)
        md = MentionDictionary()
        md.rebuild(kg_conn)
        assert b in _ids(md, "Emacs is fun")

        # Rollback: delete the recently-created entity and lower the HWM.
        kg_conn.execute("DELETE FROM kg_entities WHERE entity_id = ?", (b,))
        _set_hwm(kg_conn, 2)
        kg_conn.commit()

        assert md.rebuild(kg_conn) == "rebuilt"
        assert b not in _ids(md, "Emacs is fun")
        assert a in _ids(md, "Neovim is fun")

    def test_new_curation_with_hwm_advance_forces_full(self, kg_conn):
        a = _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)
        md = MentionDictionary()
        md.rebuild(kg_conn)

        # Add a curated alias and advance the HWM in the same turn.
        kg_conn.execute(
            "INSERT INTO kg_curations (entity_id, curation_type, value, created_at) "
            "VALUES (?, 'alias_add', 'editor', ?)",
            (a, time.time()),
        )
        _set_hwm(kg_conn, 2)
        kg_conn.commit()

        md.rebuild(kg_conn)
        assert a in _ids(md, "my favorite editor")


class TestCacheAndSortedForms:
    def test_no_change_is_hit(self, kg_conn):
        _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)
        md = MentionDictionary()
        assert md.rebuild(kg_conn) == "rebuilt"
        assert md.rebuild(kg_conn) == "hit"

    def test_sorted_forms_cached_until_rebuild(self, kg_conn):
        _add_entity(kg_conn, "Neovim")
        _set_hwm(kg_conn, 1)
        md = MentionDictionary()
        md.rebuild(kg_conn)
        assert md._sorted_forms is None  # not built yet

        md.detect_mentions("Neovim")
        assert md._sorted_forms is not None  # cached after first detect

        _add_entity(kg_conn, "Emacs", node_id=2)
        _set_hwm(kg_conn, 2)
        md.rebuild(kg_conn)
        assert md._sorted_forms is None  # invalidated by rebuild


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
