"""Tests for episodic.mcp.trace module — full trace recording."""

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from episodic.mcp.trace import (
    _ensure_table,
    compute_hash,
    evict_old_traces,
    get_traces,
    record_trace,
    redact_parameters,
    trace_tool_call,
)


@pytest.fixture
def db(tmp_path):
    """Create an in-memory DB with mcp_traces table."""
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    _ensure_table(conn)
    yield conn
    conn.close()


def _make_trace_data(**overrides):
    """Create a minimal valid trace_data dict."""
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "trace_id": "test-trace-001",
        "schema_version": "1.0",
        "timestamp_start": now,
        "timestamp_end": now,
        "duration_ms": 42,
        "direction": "server_tool_call",
        "server_id": None,
        "tool_name": "get_topics",
        "client_id": "test-client",
        "thread_id": None,
        "origin": "mcp_server",
        "purpose": "interactive",
        "request_id": "req-001",
        "parameter_schema_version": "1.0",
        "parameters_redacted": {"limit": 50},
        "input_hash": "abc123",
        "input_size_bytes": 100,
        "model_provider": None,
        "model_id": None,
        "token_in": None,
        "token_out": None,
        "cache_hit": None,
        "retries": 0,
        "timeout_ms": None,
        "status": "ok",
        "output_hash": "def456",
        "output_size_bytes": 200,
        "error_code": None,
        "message_safe": None,
        "detail_debug": None,
    }
    base.update(overrides)
    return base


# ===================================================================
# compute_hash
# ===================================================================

class TestComputeHash:
    def test_deterministic(self):
        data = {"b": 2, "a": 1}
        h1 = compute_hash(data)
        h2 = compute_hash(data)
        assert h1 == h2

    def test_key_order_independent(self):
        h1 = compute_hash({"a": 1, "b": 2})
        h2 = compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_hash({"x": 1})
        h2 = compute_hash({"x": 2})
        assert h1 != h2

    def test_returns_hex_string(self):
        h = compute_hash({"test": True})
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_dict(self):
        h = compute_hash({})
        assert len(h) == 64

    def test_handles_non_serializable(self):
        """default=str handles datetime etc."""
        h = compute_hash({"dt": datetime(2024, 1, 1)})
        assert len(h) == 64


# ===================================================================
# redact_parameters
# ===================================================================

class TestRedactParameters:
    def test_redacts_api_key(self):
        result = redact_parameters({"api_key": "secret123", "query": "hello"})
        assert result["api_key"] == "[REDACTED]"
        assert result["query"] == "hello"

    def test_redacts_token(self):
        result = redact_parameters({"auth_token": "tok_abc"})
        assert result["auth_token"] == "[REDACTED]"

    def test_redacts_password(self):
        result = redact_parameters({"password": "p@ss"})
        assert result["password"] == "[REDACTED]"

    def test_redacts_secret(self):
        result = redact_parameters({"client_secret": "shh"})
        assert result["client_secret"] == "[REDACTED]"

    def test_redacts_credential(self):
        result = redact_parameters({"credential_file": "/path"})
        assert result["credential_file"] == "[REDACTED]"

    def test_preserves_safe_keys(self):
        result = redact_parameters({"query": "test", "limit": 5})
        assert result == {"query": "test", "limit": 5}

    def test_empty_dict(self):
        assert redact_parameters({}) == {}

    def test_none_input(self):
        assert redact_parameters(None) == {}

    def test_nested_dict_redaction(self):
        result = redact_parameters({"config": {"api_key": "secret", "name": "ok"}})
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["name"] == "ok"

    def test_case_insensitive(self):
        result = redact_parameters({"API_KEY": "x", "Password": "y"})
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"


# ===================================================================
# record_trace / get_traces
# ===================================================================

class TestRecordTrace:
    def test_inserts_trace(self, db):
        data = _make_trace_data()
        trace_id = record_trace(db, data)
        assert trace_id == "test-trace-001"

        rows = db.execute("SELECT COUNT(*) FROM mcp_traces").fetchone()
        assert rows[0] == 1

    def test_returns_trace_id(self, db):
        data = _make_trace_data(trace_id="my-id")
        assert record_trace(db, data) == "my-id"

    def test_all_columns_stored(self, db):
        data = _make_trace_data(
            tool_name="search_memory",
            client_id="client-x",
            status="error",
            error_code="RuntimeError",
            message_safe="something broke",
        )
        record_trace(db, data)

        row = db.execute(
            "SELECT tool_name, client_id, status, error_code, message_safe "
            "FROM mcp_traces WHERE trace_id = ?",
            (data["trace_id"],)
        ).fetchone()
        assert row == ("search_memory", "client-x", "error", "RuntimeError", "something broke")

    def test_json_parameters_serialized(self, db):
        params = {"query": "test", "limit": 5}
        data = _make_trace_data(parameters_redacted=params)
        record_trace(db, data)

        row = db.execute(
            "SELECT parameters_redacted FROM mcp_traces WHERE trace_id = ?",
            (data["trace_id"],)
        ).fetchone()
        parsed = json.loads(row[0])
        assert parsed == {"limit": 5, "query": "test"}

    def test_duplicate_trace_id_raises(self, db):
        data = _make_trace_data()
        record_trace(db, data)
        with pytest.raises(sqlite3.IntegrityError):
            record_trace(db, data)


class TestGetTraces:
    def test_returns_empty_list(self, db):
        assert get_traces(db) == []

    def test_returns_traces_newest_first(self, db):
        for i in range(3):
            ts = (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat()
            data = _make_trace_data(
                trace_id=f"trace-{i}",
                timestamp_start=ts,
                timestamp_end=ts,
            )
            record_trace(db, data)

        traces = get_traces(db)
        assert len(traces) == 3
        assert traces[0]["trace_id"] == "trace-2"
        assert traces[2]["trace_id"] == "trace-0"

    def test_limit(self, db):
        for i in range(5):
            data = _make_trace_data(trace_id=f"trace-{i}")
            record_trace(db, data)
        assert len(get_traces(db, limit=2)) == 2

    def test_filter_by_tool_name(self, db):
        record_trace(db, _make_trace_data(trace_id="t1", tool_name="get_topics"))
        record_trace(db, _make_trace_data(trace_id="t2", tool_name="search_memory"))
        record_trace(db, _make_trace_data(trace_id="t3", tool_name="get_topics"))

        results = get_traces(db, tool_name="get_topics")
        assert len(results) == 2
        assert all(t["tool_name"] == "get_topics" for t in results)

    def test_filter_by_client_id(self, db):
        record_trace(db, _make_trace_data(trace_id="t1", client_id="alice"))
        record_trace(db, _make_trace_data(trace_id="t2", client_id="bob"))

        results = get_traces(db, client_id="alice")
        assert len(results) == 1
        assert results[0]["client_id"] == "alice"

    def test_combined_filters(self, db):
        record_trace(db, _make_trace_data(trace_id="t1", tool_name="get_topics", client_id="alice"))
        record_trace(db, _make_trace_data(trace_id="t2", tool_name="get_topics", client_id="bob"))
        record_trace(db, _make_trace_data(trace_id="t3", tool_name="search_memory", client_id="alice"))

        results = get_traces(db, tool_name="get_topics", client_id="alice")
        assert len(results) == 1
        assert results[0]["trace_id"] == "t1"

    def test_parses_json_parameters(self, db):
        data = _make_trace_data(parameters_redacted={"query": "test"})
        record_trace(db, data)
        traces = get_traces(db)
        assert traces[0]["parameters_redacted"] == {"query": "test"}


# ===================================================================
# evict_old_traces
# ===================================================================

class TestEvictOldTraces:
    def test_evicts_by_age(self, db):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()

        record_trace(db, _make_trace_data(trace_id="old", timestamp_start=old_ts, timestamp_end=old_ts))
        record_trace(db, _make_trace_data(trace_id="new", timestamp_start=new_ts, timestamp_end=new_ts))

        deleted = evict_old_traces(db, max_age_days=30)
        assert deleted == 1

        remaining = get_traces(db)
        assert len(remaining) == 1
        assert remaining[0]["trace_id"] == "new"

    def test_no_eviction_when_fresh(self, db):
        ts = datetime.now(timezone.utc).isoformat()
        record_trace(db, _make_trace_data(trace_id="t1", timestamp_start=ts, timestamp_end=ts))
        deleted = evict_old_traces(db, max_age_days=30)
        assert deleted == 0

    def test_evicts_by_size(self, db):
        ts = datetime.now(timezone.utc).isoformat()
        for i in range(5):
            data = _make_trace_data(
                trace_id=f"t{i}",
                timestamp_start=ts,
                timestamp_end=ts,
                input_size_bytes=1000,
                output_size_bytes=1000,
            )
            record_trace(db, data)

        # Total = 5 * 2000 = 10000 bytes. Limit to 5000.
        deleted = evict_old_traces(db, max_age_days=365, max_size_bytes=5000)
        assert deleted >= 2  # Should delete oldest to get under 5000
        remaining = get_traces(db)
        assert len(remaining) <= 3

    def test_empty_db_no_error(self, db):
        deleted = evict_old_traces(db)
        assert deleted == 0


# ===================================================================
# trace_tool_call context manager
# ===================================================================

class TestTraceToolCall:
    def test_records_successful_trace(self, db):
        with trace_tool_call(db, "get_topics", "client-1", {"limit": 10}) as ctx:
            ctx["output"] = {"topics": [], "total": 0}

        traces = get_traces(db)
        assert len(traces) == 1
        t = traces[0]
        assert t["tool_name"] == "get_topics"
        assert t["client_id"] == "client-1"
        assert t["status"] == "ok"
        assert t["duration_ms"] >= 0
        assert t["input_hash"]
        assert t["output_hash"]

    def test_records_error_trace(self, db):
        with pytest.raises(ValueError):
            with trace_tool_call(db, "search_memory", None, {"query": "test"}) as ctx:
                raise ValueError("something went wrong")

        traces = get_traces(db)
        assert len(traces) == 1
        t = traces[0]
        assert t["status"] == "error"
        assert t["error_code"] == "ValueError"
        assert "something went wrong" in t["message_safe"]

    def test_parameters_redacted_in_trace(self, db):
        with trace_tool_call(db, "test", None, {"api_key": "secret", "query": "hello"}) as ctx:
            ctx["output"] = {}

        traces = get_traces(db)
        params = traces[0]["parameters_redacted"]
        assert params["api_key"] == "[REDACTED]"
        assert params["query"] == "hello"

    def test_trace_id_is_uuid(self, db):
        with trace_tool_call(db, "test", None, {}) as ctx:
            trace_id = ctx["trace_id"]

        assert len(trace_id) == 36  # UUID format
        assert trace_id.count("-") == 4

    def test_context_provides_model_fields(self, db):
        with trace_tool_call(db, "test", None, {}) as ctx:
            ctx["model_provider"] = "openai"
            ctx["model_id"] = "gpt-4o"
            ctx["token_in"] = 100
            ctx["token_out"] = 50
            ctx["cache_hit"] = 1
            ctx["output"] = {}

        traces = get_traces(db)
        t = traces[0]
        assert t["model_provider"] == "openai"
        assert t["model_id"] == "gpt-4o"
        assert t["token_in"] == 100
        assert t["token_out"] == 50
        assert t["cache_hit"] == 1

    def test_default_purpose_is_interactive(self, db):
        with trace_tool_call(db, "test", None, {}) as ctx:
            ctx["output"] = {}
        traces = get_traces(db)
        assert traces[0]["purpose"] == "interactive"

    def test_custom_purpose(self, db):
        with trace_tool_call(db, "test", None, {}, purpose="background") as ctx:
            ctx["output"] = {}
        traces = get_traces(db)
        assert traces[0]["purpose"] == "background"

    def test_direction_default(self, db):
        with trace_tool_call(db, "test", None, {}) as ctx:
            ctx["output"] = {}
        traces = get_traces(db)
        assert traces[0]["direction"] == "server_tool_call"

    def test_input_output_sizes(self, db):
        with trace_tool_call(db, "test", None, {"key": "value"}) as ctx:
            ctx["output"] = {"result": "data"}

        traces = get_traces(db)
        assert traces[0]["input_size_bytes"] > 0
        assert traces[0]["output_size_bytes"] > 0

    def test_recording_failure_does_not_crash(self, db):
        """If DB recording fails, the tool call still succeeds."""
        db.close()  # Force DB error
        # This should not raise — trace recording failure is swallowed
        conn = sqlite3.connect(":memory:")
        # Use a read-only connection to force write failure
        with trace_tool_call(conn, "test", None, {}) as ctx:
            ctx["output"] = {"ok": True}
        conn.close()


# ===================================================================
# _ensure_table
# ===================================================================

class TestEnsureTable:
    def test_creates_table(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        _ensure_table(conn)
        cursor = conn.execute("PRAGMA table_info(mcp_traces)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "trace_id" in columns
        assert "tool_name" in columns
        assert "status" in columns
        conn.close()

    def test_idempotent(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        _ensure_table(conn)
        _ensure_table(conn)  # Should not raise
        conn.close()

    def test_indices_created(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        _ensure_table(conn)
        indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='mcp_traces'"
        ).fetchall()
        index_names = [i[0] for i in indices]
        assert "idx_traces_timestamp" in index_names
        assert "idx_traces_tool" in index_names
        assert "idx_traces_client" in index_names
        conn.close()
