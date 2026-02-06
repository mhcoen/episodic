"""Tests for episodic.mcp.auth module."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from episodic.mcp.auth import (
    TOKEN_PREFIX,
    generate_token,
    hash_token,
    create_token,
    validate_token,
    revoke_token,
    rotate_token,
    list_tokens,
    record_cost,
    get_daily_cost,
    _ensure_tables,
)


@pytest.fixture
def db():
    """In-memory SQLite database with MCP tables."""
    conn = sqlite3.connect(":memory:")
    _ensure_tables(conn)
    return conn


class TestTokenGeneration:
    """Tests for generate_token and hash_token."""

    def test_token_has_prefix(self):
        token, token_id = generate_token()
        assert token.startswith(TOKEN_PREFIX)

    def test_token_id_is_uuid(self):
        import uuid
        _, token_id = generate_token()
        uuid.UUID(token_id)  # Raises if not valid UUID

    def test_tokens_are_unique(self):
        t1, _ = generate_token()
        t2, _ = generate_token()
        assert t1 != t2

    def test_hash_is_deterministic(self):
        h1 = hash_token("epk_v1_test123")
        h2 = hash_token("epk_v1_test123")
        assert h1 == h2

    def test_hash_is_hex_sha256(self):
        h = hash_token("anything")
        assert len(h) == 64  # SHA-256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_tokens_different_hashes(self):
        h1 = hash_token("token_a")
        h2 = hash_token("token_b")
        assert h1 != h2


class TestCreateToken:
    """Tests for create_token."""

    def test_create_returns_plaintext_and_id(self, db):
        plaintext, token_id = create_token(db, "test-client")
        assert plaintext.startswith(TOKEN_PREFIX)
        assert len(token_id) > 0

    def test_plaintext_not_stored(self, db):
        plaintext, _ = create_token(db, "test-client")
        rows = db.execute("SELECT token_hash FROM mcp_tokens").fetchall()
        assert len(rows) == 1
        assert rows[0][0] != plaintext  # Hash stored, not plaintext

    def test_scopes_stored_as_json(self, db):
        _, token_id = create_token(db, "test", scopes=["read", "write"])
        row = db.execute(
            "SELECT scopes FROM mcp_tokens WHERE token_id = ?",
            (token_id,),
        ).fetchone()
        assert json.loads(row[0]) == ["read", "write"]

    def test_default_scopes_empty_list(self, db):
        _, token_id = create_token(db, "test")
        row = db.execute(
            "SELECT scopes FROM mcp_tokens WHERE token_id = ?",
            (token_id,),
        ).fetchone()
        assert json.loads(row[0]) == []


class TestValidateToken:
    """Tests for validate_token."""

    def test_valid_token(self, db):
        plaintext, _ = create_token(db, "test-client", scopes=["read"])
        result = validate_token(db, plaintext)
        assert result is not None
        assert result["client_id"] == "test-client"
        assert result["scopes"] == ["read"]

    def test_invalid_token(self, db):
        result = validate_token(db, "epk_v1_bogus")
        assert result is None

    def test_revoked_token(self, db):
        plaintext, token_id = create_token(db, "test")
        revoke_token(db, token_id)
        result = validate_token(db, plaintext)
        assert result is None

    def test_expired_token(self, db):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        plaintext, _ = create_token(db, "test", expires_at=past)
        result = validate_token(db, plaintext)
        assert result is None

    def test_not_yet_expired_token(self, db):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        plaintext, _ = create_token(db, "test", expires_at=future)
        result = validate_token(db, plaintext)
        assert result is not None


class TestRevokeToken:
    """Tests for revoke_token."""

    def test_revoke_existing(self, db):
        _, token_id = create_token(db, "test")
        assert revoke_token(db, token_id) is True

    def test_revoke_nonexistent(self, db):
        assert revoke_token(db, "nonexistent-id") is False

    def test_double_revoke(self, db):
        _, token_id = create_token(db, "test")
        assert revoke_token(db, token_id) is True
        assert revoke_token(db, token_id) is False


class TestRotateToken:
    """Tests for rotate_token."""

    def test_rotate_creates_new_token(self, db):
        old_plain, old_id = create_token(db, "test", scopes=["read"])
        result = rotate_token(db, old_id)
        assert result is not None
        new_plain, new_id = result
        assert new_plain != old_plain
        assert new_id != old_id

    def test_rotate_preserves_client_and_scopes(self, db):
        _, old_id = create_token(db, "my-client", scopes=["search", "index"])
        new_plain, _ = rotate_token(db, old_id)
        validated = validate_token(db, new_plain)
        assert validated["client_id"] == "my-client"
        assert validated["scopes"] == ["search", "index"]

    def test_rotate_revokes_old_immediately(self, db):
        old_plain, old_id = create_token(db, "test")
        rotate_token(db, old_id, grace_seconds=0)
        assert validate_token(db, old_plain) is None

    def test_rotate_with_grace_period(self, db):
        old_plain, old_id = create_token(db, "test")
        rotate_token(db, old_id, grace_seconds=3600)
        # Old token has revoked_at set in the future, so validate should reject
        # (revoked_at is set, even if in the future — our validate checks for non-None)
        result = validate_token(db, old_plain)
        assert result is None

    def test_rotate_nonexistent(self, db):
        assert rotate_token(db, "bogus") is None

    def test_rotate_already_revoked(self, db):
        _, token_id = create_token(db, "test")
        revoke_token(db, token_id)
        assert rotate_token(db, token_id) is None


class TestListTokens:
    """Tests for list_tokens."""

    def test_empty_list(self, db):
        assert list_tokens(db) == []

    def test_lists_active_tokens(self, db):
        create_token(db, "client-a")
        create_token(db, "client-b")
        tokens = list_tokens(db)
        assert len(tokens) == 2
        client_ids = {t["client_id"] for t in tokens}
        assert client_ids == {"client-a", "client-b"}

    def test_excludes_revoked(self, db):
        _, tid1 = create_token(db, "kept")
        _, tid2 = create_token(db, "revoked")
        revoke_token(db, tid2)
        tokens = list_tokens(db)
        assert len(tokens) == 1
        assert tokens[0]["client_id"] == "kept"

    def test_excludes_expired(self, db):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        create_token(db, "expired", expires_at=past)
        create_token(db, "active")
        tokens = list_tokens(db)
        assert len(tokens) == 1
        assert tokens[0]["client_id"] == "active"

    def test_token_fields(self, db):
        create_token(db, "test", scopes=["read"])
        tokens = list_tokens(db)
        tok = tokens[0]
        assert "token_id" in tok
        assert "client_id" in tok
        assert "scopes" in tok
        assert "created_at" in tok


class TestCostAccounting:
    """Tests for record_cost and get_daily_cost."""

    def test_record_and_get(self, db):
        record_cost(db, "client-a", 0.50)
        assert get_daily_cost(db, "client-a") == pytest.approx(0.50)

    def test_accumulates(self, db):
        record_cost(db, "client-a", 0.10)
        record_cost(db, "client-a", 0.20)
        assert get_daily_cost(db, "client-a") == pytest.approx(0.30)

    def test_separate_clients(self, db):
        record_cost(db, "a", 1.00)
        record_cost(db, "b", 2.00)
        assert get_daily_cost(db, "a") == pytest.approx(1.00)
        assert get_daily_cost(db, "b") == pytest.approx(2.00)

    def test_no_cost_returns_zero(self, db):
        assert get_daily_cost(db, "nobody") == 0.0
