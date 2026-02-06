"""Tests for episodic.mcp.threads module — thread handle management."""

import json
import sqlite3
from unittest.mock import patch

import pytest

from episodic.mcp.threads import (
    HANDLE_PREFIX,
    HANDLE_BYTE_LENGTH,
    create_thread,
    generate_thread_handle,
    get_thread_handles,
    hash_handle,
    revoke_thread_handle,
    validate_thread_handle,
)


@pytest.fixture
def db():
    """Create an in-memory DB with conversations table."""
    conn = sqlite3.connect(":memory:")
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


# ===================================================================
# generate_thread_handle
# ===================================================================

class TestGenerateThreadHandle:
    def test_returns_tuple(self):
        handle, handle_id = generate_thread_handle()
        assert isinstance(handle, str)
        assert isinstance(handle_id, str)

    def test_handle_has_prefix(self):
        handle, _ = generate_thread_handle()
        assert handle.startswith(HANDLE_PREFIX)

    def test_handle_id_is_uuid(self):
        _, handle_id = generate_thread_handle()
        assert len(handle_id) == 36
        assert handle_id.count("-") == 4

    def test_handles_are_unique(self):
        h1, id1 = generate_thread_handle()
        h2, id2 = generate_thread_handle()
        assert h1 != h2
        assert id1 != id2

    def test_handle_length(self):
        handle, _ = generate_thread_handle()
        # eth_v1_ prefix + base64url(32 bytes) = 7 + 43 = 50 chars
        assert len(handle) >= 40


# ===================================================================
# hash_handle
# ===================================================================

class TestHashHandle:
    def test_deterministic(self):
        assert hash_handle("test") == hash_handle("test")

    def test_different_inputs_different_hashes(self):
        assert hash_handle("handle_a") != hash_handle("handle_b")

    def test_returns_hex_string(self):
        h = hash_handle("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_token_hash_algorithm(self):
        """Verify same algorithm as auth.hash_token."""
        import hashlib
        plaintext = "eth_v1_test123"
        expected = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        assert hash_handle(plaintext) == expected


# ===================================================================
# create_thread
# ===================================================================

class TestCreateThread:
    def test_creates_conversation_and_handle(self, db):
        result = create_thread(db, client_id="test-client")
        assert "thread_id" in result
        assert "thread_handle" in result
        assert "handle_id" in result
        assert "permissions" in result

    def test_handle_has_prefix(self, db):
        result = create_thread(db, client_id="test-client")
        assert result["thread_handle"].startswith(HANDLE_PREFIX)

    def test_default_permissions(self, db):
        result = create_thread(db, client_id="test-client")
        assert result["permissions"] == ["read", "write"]

    def test_custom_permissions(self, db):
        result = create_thread(
            db, client_id="admin", permissions=["read", "write", "admin"]
        )
        assert result["permissions"] == ["read", "write", "admin"]

    def test_conversation_row_created(self, db):
        result = create_thread(db, client_id="test-client")
        row = db.execute(
            "SELECT conversation_id, metadata FROM conversations WHERE id = ?",
            (result["thread_id"],),
        ).fetchone()
        assert row is not None
        metadata = json.loads(row[1])
        assert metadata["created_by"] == "test-client"
        assert metadata["background_influences_topics"] is False

    def test_background_influences_topics(self, db):
        result = create_thread(
            db, client_id="test-client", background_influences_topics=True
        )
        row = db.execute(
            "SELECT metadata FROM conversations WHERE id = ?",
            (result["thread_id"],),
        ).fetchone()
        metadata = json.loads(row[0])
        assert metadata["background_influences_topics"] is True

    def test_handle_stored_as_hash(self, db):
        result = create_thread(db, client_id="test-client")
        plaintext = result["thread_handle"]
        expected_hash = hash_handle(plaintext)

        row = db.execute(
            "SELECT handle_hash FROM mcp_thread_handles WHERE handle_id = ?",
            (result["handle_id"],),
        ).fetchone()
        assert row[0] == expected_hash

    def test_multiple_threads(self, db):
        r1 = create_thread(db, client_id="c1")
        r2 = create_thread(db, client_id="c2")
        assert r1["thread_id"] != r2["thread_id"]
        assert r1["thread_handle"] != r2["thread_handle"]

    def test_thread_id_is_integer(self, db):
        result = create_thread(db, client_id="test-client")
        assert isinstance(result["thread_id"], int)


# ===================================================================
# validate_thread_handle
# ===================================================================

class TestValidateThreadHandle:
    def test_valid_handle(self, db):
        created = create_thread(db, client_id="test-client")
        result = validate_thread_handle(db, created["thread_handle"])
        assert result is not None
        assert result["handle_id"] == created["handle_id"]
        assert result["thread_id"] == created["thread_id"]
        assert result["client_id"] == "test-client"
        assert result["permissions"] == ["read", "write"]

    def test_invalid_handle_returns_none(self, db):
        result = validate_thread_handle(db, "eth_v1_nonexistent")
        assert result is None

    def test_revoked_handle_returns_none(self, db):
        created = create_thread(db, client_id="test-client")
        revoke_thread_handle(db, created["handle_id"])
        result = validate_thread_handle(db, created["thread_handle"])
        assert result is None

    def test_permission_check_passes(self, db):
        created = create_thread(db, client_id="test-client")
        result = validate_thread_handle(
            db, created["thread_handle"], required_permission="read"
        )
        assert result is not None

    def test_permission_check_fails(self, db):
        created = create_thread(db, client_id="test-client")
        result = validate_thread_handle(
            db, created["thread_handle"], required_permission="admin"
        )
        assert result is None

    def test_admin_permission_granted(self, db):
        created = create_thread(
            db, client_id="admin", permissions=["read", "write", "admin"]
        )
        result = validate_thread_handle(
            db, created["thread_handle"], required_permission="admin"
        )
        assert result is not None

    def test_no_required_permission_always_valid(self, db):
        created = create_thread(db, client_id="test-client")
        result = validate_thread_handle(
            db, created["thread_handle"], required_permission=None
        )
        assert result is not None


# ===================================================================
# get_thread_handles
# ===================================================================

class TestGetThreadHandles:
    def test_empty_list(self, db):
        assert get_thread_handles(db) == []

    def test_returns_active_handles(self, db):
        create_thread(db, client_id="c1")
        create_thread(db, client_id="c2")
        handles = get_thread_handles(db)
        assert len(handles) == 2

    def test_excludes_revoked(self, db):
        r1 = create_thread(db, client_id="c1")
        create_thread(db, client_id="c2")
        revoke_thread_handle(db, r1["handle_id"])
        handles = get_thread_handles(db)
        assert len(handles) == 1
        assert handles[0]["client_id"] == "c2"

    def test_filter_by_client_id(self, db):
        create_thread(db, client_id="alice")
        create_thread(db, client_id="bob")
        create_thread(db, client_id="alice")
        handles = get_thread_handles(db, client_id="alice")
        assert len(handles) == 2
        assert all(h["client_id"] == "alice" for h in handles)

    def test_filter_by_thread_id(self, db):
        r1 = create_thread(db, client_id="c1")
        create_thread(db, client_id="c2")
        handles = get_thread_handles(db, thread_id=r1["thread_id"])
        assert len(handles) == 1
        assert handles[0]["thread_id"] == r1["thread_id"]

    def test_combined_filters(self, db):
        r1 = create_thread(db, client_id="alice")
        create_thread(db, client_id="bob")
        create_thread(db, client_id="alice")
        handles = get_thread_handles(
            db, client_id="alice", thread_id=r1["thread_id"]
        )
        assert len(handles) == 1

    def test_handle_fields_complete(self, db):
        create_thread(db, client_id="test")
        handles = get_thread_handles(db)
        h = handles[0]
        assert set(h.keys()) == {
            "handle_id", "thread_id", "client_id", "permissions", "created_at"
        }

    def test_ordered_by_created_at_desc(self, db):
        # Insert with explicit timestamps to avoid same-second ordering issues
        r1 = create_thread(db, client_id="first")
        # Manually update created_at to ensure ordering
        db.execute(
            "UPDATE mcp_thread_handles SET created_at = '2024-01-01T00:00:00' "
            "WHERE handle_id = ?",
            (r1["handle_id"],),
        )
        r2 = create_thread(db, client_id="second")
        db.execute(
            "UPDATE mcp_thread_handles SET created_at = '2024-01-02T00:00:00' "
            "WHERE handle_id = ?",
            (r2["handle_id"],),
        )
        db.commit()
        handles = get_thread_handles(db)
        # Second created (newer) should be first in list (DESC order)
        assert handles[0]["handle_id"] == r2["handle_id"]
        assert handles[1]["handle_id"] == r1["handle_id"]


# ===================================================================
# revoke_thread_handle
# ===================================================================

class TestRevokeThreadHandle:
    def test_revoke_existing(self, db):
        created = create_thread(db, client_id="test-client")
        assert revoke_thread_handle(db, created["handle_id"]) is True

    def test_revoke_nonexistent(self, db):
        assert revoke_thread_handle(db, "nonexistent-id") is False

    def test_double_revoke(self, db):
        created = create_thread(db, client_id="test-client")
        assert revoke_thread_handle(db, created["handle_id"]) is True
        assert revoke_thread_handle(db, created["handle_id"]) is False

    def test_revoke_sets_timestamp(self, db):
        created = create_thread(db, client_id="test-client")
        revoke_thread_handle(db, created["handle_id"])
        row = db.execute(
            "SELECT revoked_at FROM mcp_thread_handles WHERE handle_id = ?",
            (created["handle_id"],),
        ).fetchone()
        assert row[0] is not None

    def test_revoked_handle_excluded_from_list(self, db):
        created = create_thread(db, client_id="test-client")
        revoke_thread_handle(db, created["handle_id"])
        handles = get_thread_handles(db)
        assert len(handles) == 0
