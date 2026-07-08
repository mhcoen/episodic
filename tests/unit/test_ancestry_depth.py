"""Regression test: context_depth=N must yield N prior exchanges, not N-1.

The old ancestry loop counted an exchange on the assistant->user boundary and
broke before appending the completing user message, then an even-count trim +
start-with-user pop dropped it — so depth=1 produced zero prior history.
"""

import os
import sqlite3
import tempfile

import pytest


@pytest.fixture
def chain_db(monkeypatch):
    db_path = os.path.join(tempfile.mkdtemp(), "chain.db")
    monkeypatch.setenv("EPISODIC_DB_PATH", db_path)

    from episodic import db_connection
    db_connection.close_pool()
    db_connection._resolved_db_path = None

    from episodic.db_migrations import initialize_db
    from episodic.db_nodes import insert_node
    initialize_db(create_root_node=False)

    ids = []
    parent = None
    for role, txt in [("user", "qA"), ("assistant", "ansA"),
                      ("user", "qB"), ("assistant", "ansB"),
                      ("user", "qC")]:
        nid, _ = insert_node(txt, parent_id=parent, role=role)
        parent = nid
        ids.append(nid)

    yield ids

    db_connection.close_pool()
    db_connection._resolved_db_path = None


def _assemble(user_node_id, depth):
    from episodic.context_recovery.ancestry import AncestryStrategy
    from episodic.config import config
    config.set("context_depth", depth)
    result = AncestryStrategy().assemble(
        user_turn_text="qC", user_node_id=user_node_id,
        active_topic_start_node_id=None, user_embedding=None, token_budget=4000,
    )
    return [(m["role"], m["content"]) for m in result.messages]


def test_depth_one_includes_one_prior_exchange(chain_db):
    seq = _assemble(chain_db[-1], depth=1)
    assert seq == [("user", "qB"), ("assistant", "ansB"), ("user", "qC")]


def test_depth_two_includes_two_prior_exchanges(chain_db):
    seq = _assemble(chain_db[-1], depth=2)
    assert seq == [
        ("user", "qA"), ("assistant", "ansA"),
        ("user", "qB"), ("assistant", "ansB"),
        ("user", "qC"),
    ]


def test_depth_capped_by_history(chain_db):
    # Only two prior exchanges exist; asking for more returns all of them.
    seq = _assemble(chain_db[-1], depth=10)
    assert seq[0] == ("user", "qA")
    assert seq[-1] == ("user", "qC")
    assert len(seq) == 5


def test_starts_with_user_and_alternates(chain_db):
    seq = _assemble(chain_db[-1], depth=2)
    assert seq[0][0] == "user"
    roles = [r for r, _ in seq]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
