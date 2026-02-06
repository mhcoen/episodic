"""Tests for episodic.mcp.stateful module — stateful LLM conversations."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from episodic.mcp.stateful import (
    _build_context_messages,
    _get_thread_ancestry,
    _get_thread_head,
    _insert_thread_node,
    ask_llm_stateful,
)


@pytest.fixture(autouse=True)
def _mock_short_id():
    """Mock generate_short_id to avoid needing the global DB connection."""
    counter = [0]
    def fake_short_id(*args, **kwargs):
        counter[0] += 1
        return f"s{counter[0]:03d}"
    with patch("episodic.db_ids.generate_short_id", side_effect=fake_short_id):
        yield


@pytest.fixture
def db():
    """Create an in-memory DB with nodes, state, and conversations tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            short_id TEXT UNIQUE,
            parent_id TEXT,
            content TEXT,
            role TEXT,
            provider TEXT,
            model TEXT,
            is_meta_query INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES nodes(id)
        )
    """)
    conn.execute("""
        CREATE TABLE state (
            name TEXT PRIMARY KEY,
            head_id TEXT,
            FOREIGN KEY (head_id) REFERENCES nodes(id)
        )
    """)
    conn.execute("INSERT INTO state (name, head_id) VALUES ('head', NULL)")
    conn.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT UNIQUE NOT NULL,
            root_node_id TEXT,
            current_head_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    """)
    yield conn
    conn.close()


def _create_conversation(conn, conv_id="test-conv"):
    """Create a conversation and return its ID."""
    conn.execute(
        "INSERT INTO conversations (conversation_id) VALUES (?)",
        (conv_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM conversations WHERE conversation_id = ?",
        (conv_id,),
    ).fetchone()
    return row[0]


# ===================================================================
# _insert_thread_node
# ===================================================================

class TestInsertThreadNode:
    def test_inserts_node(self, db):
        tid = _create_conversation(db)
        node_id, short_id = _insert_thread_node(
            db, tid, "Hello", role="user"
        )
        assert node_id is not None
        assert short_id is not None

        row = db.execute(
            "SELECT content, role FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        assert row == ("Hello", "user")

    def test_updates_thread_head(self, db):
        tid = _create_conversation(db)
        node_id, _ = _insert_thread_node(
            db, tid, "Hi", role="user"
        )
        row = db.execute(
            "SELECT current_head_id FROM conversations WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row[0] == node_id

    def test_does_not_update_global_head(self, db):
        tid = _create_conversation(db)
        _insert_thread_node(db, tid, "Hi", role="user")
        row = db.execute(
            "SELECT head_id FROM state WHERE name = 'head'"
        ).fetchone()
        assert row[0] is None  # Global head unchanged

    def test_parent_id_set(self, db):
        tid = _create_conversation(db)
        n1, _ = _insert_thread_node(db, tid, "Q", role="user")
        n2, _ = _insert_thread_node(
            db, tid, "A", parent_id=n1, role="assistant"
        )
        row = db.execute(
            "SELECT parent_id FROM nodes WHERE id = ?",
            (n2,),
        ).fetchone()
        assert row[0] == n1

    def test_provider_and_model_stored(self, db):
        tid = _create_conversation(db)
        node_id, _ = _insert_thread_node(
            db, tid, "response", role="assistant",
            provider="openai", model="gpt-4o-mini"
        )
        row = db.execute(
            "SELECT provider, model FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        assert row == ("openai", "gpt-4o-mini")

    def test_chain_of_nodes(self, db):
        tid = _create_conversation(db)
        n1, _ = _insert_thread_node(db, tid, "Q1", role="user")
        n2, _ = _insert_thread_node(
            db, tid, "A1", parent_id=n1, role="assistant"
        )
        n3, _ = _insert_thread_node(
            db, tid, "Q2", parent_id=n2, role="user"
        )
        # Thread head should be n3
        assert _get_thread_head(db, tid) == n3


# ===================================================================
# _get_thread_head
# ===================================================================

class TestGetThreadHead:
    def test_none_for_new_thread(self, db):
        tid = _create_conversation(db)
        assert _get_thread_head(db, tid) is None

    def test_returns_head_after_insert(self, db):
        tid = _create_conversation(db)
        node_id, _ = _insert_thread_node(db, tid, "Hi", role="user")
        assert _get_thread_head(db, tid) == node_id

    def test_nonexistent_thread(self, db):
        assert _get_thread_head(db, 999) is None


# ===================================================================
# _get_thread_ancestry
# ===================================================================

class TestGetThreadAncestry:
    def test_single_node(self, db):
        tid = _create_conversation(db)
        n1, _ = _insert_thread_node(db, tid, "Hello", role="user")
        ancestry = _get_thread_ancestry(db, n1)
        assert len(ancestry) == 1
        assert ancestry[0]["content"] == "Hello"

    def test_chain_oldest_first(self, db):
        tid = _create_conversation(db)
        n1, _ = _insert_thread_node(db, tid, "Q1", role="user")
        n2, _ = _insert_thread_node(
            db, tid, "A1", parent_id=n1, role="assistant"
        )
        n3, _ = _insert_thread_node(
            db, tid, "Q2", parent_id=n2, role="user"
        )
        ancestry = _get_thread_ancestry(db, n3)
        assert len(ancestry) == 3
        assert ancestry[0]["content"] == "Q1"
        assert ancestry[1]["content"] == "A1"
        assert ancestry[2]["content"] == "Q2"

    def test_nonexistent_node(self, db):
        ancestry = _get_thread_ancestry(db, "nonexistent")
        assert ancestry == []


# ===================================================================
# _build_context_messages
# ===================================================================

class TestBuildContextMessages:
    def test_empty_ancestry(self):
        messages = _build_context_messages([], "Hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_includes_ancestry(self):
        ancestry = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        messages = _build_context_messages(ancestry, "Q2")
        assert len(messages) == 4  # system + Q1 + A1 + Q2
        assert messages[1] == {"role": "user", "content": "Q1"}
        assert messages[2] == {"role": "assistant", "content": "A1"}
        assert messages[3] == {"role": "user", "content": "Q2"}

    def test_context_depth_limits(self):
        ancestry = []
        for i in range(10):
            ancestry.append({"role": "user", "content": f"Q{i}"})
            ancestry.append({"role": "assistant", "content": f"A{i}"})

        messages = _build_context_messages(ancestry, "new Q", context_depth=2)
        # system + 2 exchanges (4 messages) + new user message = 6
        # But context_depth counts exchanges from newest backwards
        user_messages = [m for m in messages if m["role"] == "user"]
        # Should have at most 3 user messages: 2 from history + 1 new
        assert len(user_messages) <= 3

    def test_custom_system_message(self):
        messages = _build_context_messages(
            [], "Hi", system_message="Be concise."
        )
        assert messages[0]["content"] == "Be concise."

    def test_skips_non_user_assistant_roles(self):
        ancestry = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        messages = _build_context_messages(ancestry, "Q2")
        roles = [m["role"] for m in messages]
        # Only one system (from build_context), no duplicate from ancestry
        assert roles.count("system") == 1


# ===================================================================
# ask_llm_stateful
# ===================================================================

class TestAskLlmStateful:
    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_basic_conversation(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "model": "gpt-4o-mini",
        }.get(key, default))
        mock_llm.return_value = (
            "The capital of France is Paris.",
            {"input_tokens": 50, "output_tokens": 20, "cost_usd": 0.001},
        )

        tid = _create_conversation(db)
        result = ask_llm_stateful(
            db, thread_id=tid, client_id="test",
            message="What is the capital of France?"
        )

        assert result["response"] == "The capital of France is Paris."
        assert result["thread_id"] == tid
        assert result["tokens_in"] == 50
        assert result["tokens_out"] == 20
        assert result["model"] == "gpt-4o-mini"
        assert result["provider"] == "openai"
        assert result["node_id"] is not None

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_nodes_inserted(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        tid = _create_conversation(db)
        result = ask_llm_stateful(
            db, thread_id=tid, client_id="test", message="Question"
        )

        # Should have 2 nodes: user + assistant
        nodes = db.execute("SELECT role, content FROM nodes ORDER BY created_at").fetchall()
        assert len(nodes) == 2
        assert nodes[0] == ("user", "Question")
        assert nodes[1] == ("assistant", "Answer")

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_thread_head_updated(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        tid = _create_conversation(db)
        result = ask_llm_stateful(
            db, thread_id=tid, client_id="test", message="Question"
        )

        head = _get_thread_head(db, tid)
        assert head == result["node_id"]  # Head points to assistant node

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_global_head_unchanged(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        tid = _create_conversation(db)
        ask_llm_stateful(db, thread_id=tid, client_id="test", message="Q")

        global_head = db.execute(
            "SELECT head_id FROM state WHERE name = 'head'"
        ).fetchone()
        assert global_head[0] is None

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_multi_turn_context(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")

        tid = _create_conversation(db)

        # Turn 1
        mock_llm.return_value = ("Paris", {"input_tokens": 10, "output_tokens": 5})
        ask_llm_stateful(
            db, thread_id=tid, client_id="test",
            message="Capital of France?"
        )

        # Turn 2 — LLM should receive context from turn 1
        mock_llm.return_value = ("Madrid", {"input_tokens": 20, "output_tokens": 5})
        ask_llm_stateful(
            db, thread_id=tid, client_id="test",
            message="And Spain?"
        )

        # Check that turn 2's LLM call included context from turn 1
        call_args = mock_llm.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        # Should have: system + Q1 + A1 + Q2
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "Capital of France?"
        assert user_msgs[1]["content"] == "And Spain?"

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_parent_chain_correct(self, mock_config, mock_provider, mock_llm, db):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("A1", {"input_tokens": 10, "output_tokens": 5})

        tid = _create_conversation(db)
        r1 = ask_llm_stateful(
            db, thread_id=tid, client_id="test", message="Q1"
        )

        # Check parent chain: user → assistant
        assistant_node = db.execute(
            "SELECT parent_id FROM nodes WHERE id = ?",
            (r1["node_id"],),
        ).fetchone()
        user_node_id = assistant_node[0]

        user_node = db.execute(
            "SELECT content, parent_id FROM nodes WHERE id = ?",
            (user_node_id,),
        ).fetchone()
        assert user_node[0] == "Q1"
        assert user_node[1] is None  # First node has no parent

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_cost_info_missing(self, mock_config, mock_provider, mock_llm, db):
        """Token counts default to 0 when cost_info has no token keys."""
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {})  # No token info

        tid = _create_conversation(db)
        result = ask_llm_stateful(
            db, thread_id=tid, client_id="test", message="Q"
        )
        assert result["tokens_in"] == 0
        assert result["tokens_out"] == 0
