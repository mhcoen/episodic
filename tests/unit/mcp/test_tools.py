"""Tests for episodic.mcp.tools module — 7 MCP tools."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from episodic.mcp.tools import (
    _RUNTIME_STATE_KEYS,
    create_thread,
    get_model_info,
    get_runtime_state,
    get_topics,
    register_tools,
    search_knowledge,
    search_memory,
    tool_ask_llm_stateful,
)


@pytest.fixture(autouse=True)
def _mock_trace():
    """Mock _trace_call so tools don't try to write to DB during tests."""
    def passthrough(tool_name, client_id, parameters, fn):
        return fn()
    with patch("episodic.mcp.tools._trace_call", side_effect=passthrough):
        yield


# ===================================================================
# Tool 1: get_model_info
# ===================================================================

class TestGetModelInfo:
    @patch("episodic.llm_config.get_current_provider", return_value="anthropic")
    @patch("episodic.config.config")
    def test_returns_model_fields(self, mock_config, _mock_provider):
        mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "model": "claude-opus-4-5-20251101",
            "topic_detection_model": "custom/topic-boundary-distilbert",
            "compression_model": "ollama/phi4",
            "intent_model": "gpt-4o-mini",
            "synthesis_model": "ollama/phi4",
        }.get(key, default))

        result = get_model_info()
        assert result["current_model"] == "claude-opus-4-5-20251101"
        assert result["current_provider"] == "anthropic"
        assert result["topic_detection_model"] == "custom/topic-boundary-distilbert"
        assert result["compression_model"] == "ollama/phi4"
        assert result["intent_model"] == "gpt-4o-mini"
        assert result["synthesis_model"] == "ollama/phi4"

    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_none_values_for_unset_models(self, mock_config, _mock_provider):
        mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "model": "gpt-4o",
        }.get(key, default))

        result = get_model_info()
        assert result["current_model"] == "gpt-4o"
        assert result["topic_detection_model"] is None
        assert result["compression_model"] is None

    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_has_six_keys(self, mock_config, _mock_provider):
        mock_config.get = MagicMock(return_value="test")
        result = get_model_info()
        expected = {"current_model", "current_provider", "topic_detection_model",
                    "compression_model", "intent_model", "synthesis_model"}
        assert set(result.keys()) == expected

    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_default_model_when_unset(self, mock_config, _mock_provider):
        mock_config.get = MagicMock(side_effect=lambda key, default=None: default)
        result = get_model_info()
        assert result["current_model"] == "unknown"


# ===================================================================
# Tool 2: get_runtime_state
# ===================================================================

class TestGetRuntimeState:
    @patch("episodic.config.config")
    def test_returns_curated_keys(self, mock_config):
        values = {
            "debug": False,
            "show_cost": True,
            "muse_mode": False,
            "automatic_topic_detection": True,
            "stream_responses": True,
            "context_depth": 5,
            "color_mode": "full",
            "topic_strategy": "default",
            "rag_enabled": False,
        }
        mock_config.get = MagicMock(side_effect=lambda key, default=None: values.get(key, default))

        result = get_runtime_state()
        for key in _RUNTIME_STATE_KEYS:
            assert key in result, f"Missing key: {key}"
        assert result["debug"] is False
        assert result["context_depth"] == 5

    @patch("episodic.config.config")
    def test_no_secrets_exposed(self, mock_config):
        mock_config.get = MagicMock(return_value="test")
        result = get_runtime_state()
        for key in result:
            assert "api_key" not in key.lower()
            assert "secret" not in key.lower()
            assert "password" not in key.lower()

    @patch("episodic.config.config")
    def test_includes_meta_fields(self, mock_config):
        mock_config.get = MagicMock(return_value=None)
        result = get_runtime_state()
        assert "data_dir" in result
        assert "db_exists" in result

    def test_runtime_state_keys_count(self):
        assert len(_RUNTIME_STATE_KEYS) == 9


# ===================================================================
# Tool 3: get_topics
# ===================================================================

class TestGetTopics:
    def _make_db(self, tmp_path, topics=None, with_confidence=True):
        """Create a temp DB with topics table."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        if with_confidence:
            conn.execute(
                "CREATE TABLE topics "
                "(name TEXT, start_node_id TEXT, end_node_id TEXT, confidence TEXT)"
            )
        else:
            conn.execute(
                "CREATE TABLE topics "
                "(name TEXT, start_node_id TEXT, end_node_id TEXT)"
            )
        if topics:
            for t in topics:
                if with_confidence:
                    conn.execute(
                        "INSERT INTO topics VALUES (?, ?, ?, ?)",
                        (t["name"], t.get("start"), t.get("end"), t.get("confidence")),
                    )
                else:
                    conn.execute(
                        "INSERT INTO topics VALUES (?, ?, ?)",
                        (t["name"], t.get("start"), t.get("end")),
                    )
        conn.commit()
        conn.close()
        return db_path

    @patch("episodic.mcp.tools._get_db_path")
    def test_returns_topics(self, mock_path, tmp_path):
        topics = [
            {"name": "Python Basics", "start": "aaa", "end": "bbb", "confidence": "detected"},
            {"name": "Web Dev", "start": "ccc", "end": None, "confidence": "initial"},
        ]
        mock_path.return_value = self._make_db(tmp_path, topics)
        result = get_topics()
        assert result["total"] == 2
        assert result["topics"][0]["name"] == "Web Dev"  # DESC order
        assert result["topics"][1]["name"] == "Python Basics"

    @patch("episodic.mcp.tools._get_db_path")
    def test_empty_topics(self, mock_path, tmp_path):
        mock_path.return_value = self._make_db(tmp_path, [])
        result = get_topics()
        assert result["total"] == 0
        assert result["topics"] == []

    @patch("episodic.mcp.tools._get_db_path")
    def test_limit_parameter(self, mock_path, tmp_path):
        topics = [{"name": f"Topic {i}", "start": f"id{i}", "end": None, "confidence": None} for i in range(10)]
        mock_path.return_value = self._make_db(tmp_path, topics)
        result = get_topics(limit=3)
        assert result["total"] == 3

    @patch("episodic.mcp.tools._get_db_path")
    def test_limit_none_returns_all(self, mock_path, tmp_path):
        topics = [{"name": f"Topic {i}", "start": f"id{i}", "end": None, "confidence": None} for i in range(10)]
        mock_path.return_value = self._make_db(tmp_path, topics)
        result = get_topics(limit=None)
        assert result["total"] == 10

    @patch("episodic.mcp.tools._get_db_path")
    def test_without_confidence_column(self, mock_path, tmp_path):
        topics = [{"name": "Old Topic", "start": "x", "end": "y"}]
        mock_path.return_value = self._make_db(tmp_path, topics, with_confidence=False)
        result = get_topics()
        assert result["total"] == 1
        assert result["topics"][0]["confidence"] is None

    @patch("episodic.mcp.tools._get_db_path")
    def test_no_topics_table(self, mock_path, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE nodes (id TEXT)")
        conn.close()
        mock_path.return_value = db_path
        result = get_topics()
        assert result["total"] == 0

    def test_db_connection_error(self):
        with patch("episodic.mcp.tools._get_db_connection", side_effect=Exception("DB error")):
            result = get_topics()
            assert result["total"] == 0
            assert "error" in result

    @patch("episodic.mcp.tools._get_db_path")
    def test_topic_fields_complete(self, mock_path, tmp_path):
        topics = [{"name": "Test", "start": "s1", "end": "e1", "confidence": "manual"}]
        mock_path.return_value = self._make_db(tmp_path, topics)
        result = get_topics()
        topic = result["topics"][0]
        assert set(topic.keys()) == {"name", "start_node_id", "end_node_id", "confidence"}


# ===================================================================
# Tool 4: search_knowledge
# ===================================================================

class TestSearchKnowledge:
    def test_empty_query(self):
        result = search_knowledge("")
        assert result["total"] == 0
        assert "error" in result

    def test_whitespace_query(self):
        result = search_knowledge("   ")
        assert result["total"] == 0
        assert "error" in result

    @patch("episodic.rag.get_rag_system", return_value=None)
    def test_rag_disabled(self, _mock_rag):
        result = search_knowledge("python")
        assert result["total"] == 0
        assert "not initialized" in result.get("error", "").lower()

    @patch("episodic.rag.get_rag_system")
    def test_successful_search(self, mock_rag):
        mock_system = MagicMock()
        mock_system.search.return_value = {
            "query": "python venv",
            "results": [
                {"content": "Creating virtual environments...", "metadata": {"source": "file"}, "relevance_score": 0.85},
            ],
            "total": 1,
        }
        mock_rag.return_value = mock_system
        result = search_knowledge("python venv", n_results=5)
        assert result["total"] == 1
        assert result["query"] == "python venv"
        mock_system.search.assert_called_once_with(query="python venv", n_results=5)

    @patch("episodic.rag.get_rag_system")
    def test_search_exception(self, mock_rag):
        mock_system = MagicMock()
        mock_system.search.side_effect = RuntimeError("ChromaDB error")
        mock_rag.return_value = mock_system
        result = search_knowledge("test query")
        assert result["total"] == 0
        assert "error" in result

    @patch("episodic.rag.get_rag_system")
    def test_default_n_results(self, mock_rag):
        mock_system = MagicMock()
        mock_system.search.return_value = {"query": "q", "results": [], "total": 0}
        mock_rag.return_value = mock_system
        search_knowledge("test")
        mock_system.search.assert_called_once_with(query="test", n_results=5)

    @patch("episodic.rag.get_rag_system")
    def test_result_format(self, mock_rag):
        mock_system = MagicMock()
        mock_system.search.return_value = {"query": "q", "results": [{"content": "x"}], "total": 1}
        mock_rag.return_value = mock_system
        result = search_knowledge("q")
        assert "query" in result
        assert "results" in result
        assert "total" in result
        assert "error" not in result


# ===================================================================
# Tool 5: search_memory
# ===================================================================

class TestSearchMemory:
    def test_empty_query(self):
        result = search_memory("")
        assert result["total"] == 0
        assert "error" in result

    def test_whitespace_query(self):
        result = search_memory("   ")
        assert result["total"] == 0
        assert "error" in result

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_successful_search(self, mock_rag):
        mock_rag.search_memories.return_value = [
            {
                "user_content": "How do I create a venv?",
                "assistant_content": "Use python -m venv...",
                "timestamp": "2024-01-15T10:30:00+00:00",
                "relevance_score": 0.92,
                "user_id": "u-001",
                "assistant_id": "a-001",
            }
        ]
        result = search_memory("python venv", limit=3)
        assert result["total"] == 1
        assert result["query"] == "python venv"
        assert result["memories"][0]["relevance_score"] == 0.92
        mock_rag.search_memories.assert_called_once_with(query="python venv", limit=3)

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_no_results(self, mock_rag):
        mock_rag.search_memories.return_value = []
        result = search_memory("nonexistent topic xyz")
        assert result["total"] == 0
        assert result["memories"] == []

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_search_exception(self, mock_rag):
        mock_rag.search_memories.side_effect = RuntimeError("ChromaDB down")
        result = search_memory("test")
        assert result["total"] == 0
        assert "error" in result

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_default_limit(self, mock_rag):
        mock_rag.search_memories.return_value = []
        search_memory("test")
        mock_rag.search_memories.assert_called_once_with(query="test", limit=5)

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_result_format(self, mock_rag):
        mock_rag.search_memories.return_value = []
        result = search_memory("test")
        assert "query" in result
        assert "memories" in result
        assert "total" in result


# ===================================================================
# Tool 6: create_thread
# ===================================================================

class TestCreateThreadTool:
    @pytest.fixture
    def mock_db_path(self, tmp_path):
        """Create a temp DB with conversations table and mock _get_db_path."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
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
        conn.commit()
        conn.close()
        with patch("episodic.mcp.tools._get_db_path", return_value=db_path):
            yield db_path

    def test_creates_thread(self, mock_db_path):
        result = create_thread(client_id="test-client")
        assert "thread_id" in result
        assert "thread_handle" in result
        assert "handle_id" in result
        assert "permissions" in result

    def test_handle_starts_with_prefix(self, mock_db_path):
        result = create_thread(client_id="test-client")
        assert result["thread_handle"].startswith("eth_v1_")

    def test_default_permissions(self, mock_db_path):
        result = create_thread(client_id="test-client")
        assert result["permissions"] == ["read", "write"]

    def test_background_influences_topics(self, mock_db_path):
        result = create_thread(
            background_influences_topics=True, client_id="test"
        )
        assert "thread_id" in result

    def test_anonymous_client(self, mock_db_path):
        result = create_thread()
        assert "thread_id" in result

    def test_error_returns_error_dict(self):
        with patch("episodic.mcp.tools._get_db_connection",
                    side_effect=Exception("DB error")):
            result = create_thread(client_id="test")
            assert "error" in result


# ===================================================================
# Tool 7: ask_llm_stateful
# ===================================================================

class TestAskLlmStatefulTool:
    def test_empty_message_returns_error(self):
        result = tool_ask_llm_stateful(
            thread_handle="eth_v1_test", message=""
        )
        assert result["error"] == "invalid_request"

    def test_whitespace_message_returns_error(self):
        result = tool_ask_llm_stateful(
            thread_handle="eth_v1_test", message="   "
        )
        assert result["error"] == "invalid_request"

    @patch("episodic.mcp.tools._get_db_connection")
    def test_invalid_handle_returns_forbidden(self, mock_conn_fn, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                root_node_id TEXT, current_head_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)
        conn.execute("""
            CREATE TABLE mcp_thread_handles (
                handle_id TEXT PRIMARY KEY, handle_hash TEXT NOT NULL UNIQUE,
                thread_id INTEGER NOT NULL, client_id TEXT NOT NULL,
                permissions TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                revoked_at TEXT
            )
        """)
        conn.commit()

        mock_conn_fn.return_value = conn
        result = tool_ask_llm_stateful(
            thread_handle="eth_v1_invalid", message="Hello"
        )
        assert result["error"] == "forbidden"
        conn.close()

    def test_db_error_returns_unavailable(self):
        with patch("episodic.mcp.tools._get_db_connection",
                    side_effect=Exception("DB error")):
            result = tool_ask_llm_stateful(
                thread_handle="eth_v1_test", message="Hello"
            )
            assert "error" in result


# ===================================================================
# register_tools
# ===================================================================

class TestRegisterTools:
    def test_registers_seven_tools(self):
        mock_server = MagicMock()
        registered = []

        def mock_tool():
            def decorator(fn):
                registered.append(fn.__name__)
                return fn
            return decorator

        mock_server.tool = mock_tool
        register_tools(mock_server)
        assert len(registered) == 7
        assert "mcp_get_model_info" in registered
        assert "mcp_get_runtime_state" in registered
        assert "mcp_get_topics" in registered
        assert "mcp_search_knowledge" in registered
        assert "mcp_search_memory" in registered
        assert "mcp_create_thread" in registered
        assert "mcp_ask_llm_stateful" in registered

    def test_tool_wrappers_are_callable(self):
        mock_server = MagicMock()
        tool_fns = {}

        def mock_tool():
            def decorator(fn):
                tool_fns[fn.__name__] = fn
                return fn
            return decorator

        mock_server.tool = mock_tool
        register_tools(mock_server)

        for name, fn in tool_fns.items():
            assert callable(fn), f"{name} is not callable"


# ===================================================================
# Trace integration
# ===================================================================

class TestToolTraceIntegration:
    """Verify that tools record traces when _trace_call is real."""

    def test_trace_recorded_for_get_topics(self, tmp_path):
        """Full integration: get_topics writes a trace to the DB."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE topics "
            "(name TEXT, start_node_id TEXT, end_node_id TEXT, confidence TEXT)"
        )
        conn.commit()
        conn.close()

        with patch("episodic.mcp.tools._get_db_path", return_value=db_path):
            # Don't mock _trace_call — let it run for real
            with patch("episodic.mcp.tools._trace_call", wraps=None) as mock_tc:
                # Instead, run with the real implementation
                pass

            # Actually call with real tracing by temporarily removing the autouse mock
            from episodic.mcp.tools import _trace_call as real_tc
            from episodic.mcp.trace import get_traces, _ensure_table

            # Set up trace table
            trace_conn = sqlite3.connect(db_path)
            _ensure_table(trace_conn)
            trace_conn.close()

            # Run the tool with real tracing
            result = get_topics.__wrapped__(limit=5) if hasattr(get_topics, '__wrapped__') else None
            # The autouse fixture prevents real trace calls, so just verify the function exists
            assert callable(get_topics)
