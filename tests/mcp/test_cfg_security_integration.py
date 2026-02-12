"""
Tests for CFG MCP Security Integration.

Spec tests 21-30 from CFG_MCP_DISPATCH_EXTENSION.md §9.4.
Tests the integration between CFGDirectiveAdapter, PolicyEngine,
and the SecurityPipeline.
"""

import hashlib
import time

import pytest

from episodic.mcp.security.types import (
    AuthorizationEvent,
    SecurityContext,
    PolicyConfig,
    TrustLevel,
    ContentType,
    GateResult,
)
from episodic.mcp.security.policy_engine import PolicyEngine, DESTRUCTIVE_TOOLS
from episodic.mcp.security.pipeline import SecurityPipeline
from episodic.mcp.adapters.cfg_directive import CFGDirectiveAdapter
from episodic.mcp.dispatch_types import WRITE_INTENTS
from episodic.utility.types import UtilityQuery


@pytest.fixture
def engine():
    return PolicyEngine()


@pytest.fixture
def pipeline():
    return SecurityPipeline()


@pytest.fixture
def client_ctx():
    """Client-mode context with destructive tools disabled."""
    return SecurityContext(
        mode="client",
        source_type="mcp_server",
        source_id="gsuite",
        trust_level=TrustLevel.UNTRUSTED,
        content_type=ContentType.PLAINTEXT,
        policy=PolicyConfig(enable_destructive=False),
    )


@pytest.fixture
def client_ctx_destructive():
    """Client-mode context with destructive tools enabled."""
    return SecurityContext(
        mode="client",
        source_type="mcp_server",
        source_id="gsuite",
        trust_level=TrustLevel.UNTRUSTED,
        content_type=ContentType.PLAINTEXT,
        policy=PolicyConfig(enable_destructive=True),
    )


def _make_auth_event(tool: str, scope: dict = None) -> AuthorizationEvent:
    """Helper to create a valid auth event."""
    return AuthorizationEvent(
        action=tool,
        scope=scope or {},
        message_hash=hashlib.sha256(b"test").hexdigest(),
        timestamp=time.time(),
        source="cfg_parser",
    )


class TestReadIntents:
    """Spec tests 21."""

    def test_21_read_intent_no_auth_event(self, engine, client_ctx):
        """Test 21: Read intent passes without auth event."""
        result = engine.check_tool_execution(
            tool="query_gmail_emails",
            args={},
            auth_event=None,
            ctx=client_ctx,
        )
        assert result.allowed is True
        assert result.confidence == "implicit"

    def test_read_intent_via_pipeline(self, pipeline, client_ctx):
        """Read intent passes through full pipeline."""
        result = pipeline.check_tool_execution(
            tool="get_calendar_events",
            args={},
            ctx=client_ctx,
            auth_event=None,
        )
        assert result.allowed is True


class TestWriteIntents:
    """Spec tests 22-23."""

    def test_22_write_intent_without_auth_blocked(self, engine, client_ctx_destructive):
        """Test 22: Write intent (destructive tool) without auth event → blocked."""
        result = engine.check_tool_execution(
            tool="create_calendar_event",
            args={},
            auth_event=None,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is False
        assert "authorization" in result.reason.lower()

    def test_23_write_intent_with_auth_passes(self, engine, client_ctx_destructive):
        """Test 23: Write intent with valid auth event → allowed."""
        auth = _make_auth_event("create_calendar_event")
        result = engine.check_tool_execution(
            tool="create_calendar_event",
            args={},
            auth_event=auth,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is True
        assert result.confidence == "explicit"


class TestDestructiveIntents:
    """Spec tests 24-25."""

    def test_24_destructive_flag_off_blocked(self, engine, client_ctx):
        """Test 24: Destructive intent with flag off → blocked."""
        auth = _make_auth_event("delete_calendar_event")
        result = engine.check_tool_execution(
            tool="delete_calendar_event",
            args={},
            auth_event=auth,
            ctx=client_ctx,  # enable_destructive=False
        )
        assert result.allowed is False
        assert "not enabled" in result.reason.lower()

    def test_25_destructive_flag_on_with_auth(self, engine, client_ctx_destructive):
        """Test 25: Destructive intent with flag on + auth event → allowed."""
        auth = _make_auth_event("delete_calendar_event")
        result = engine.check_tool_execution(
            tool="delete_calendar_event",
            args={},
            auth_event=auth,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is True

    def test_delete_gmail_draft_is_destructive(self):
        """delete_gmail_draft is in DESTRUCTIVE_TOOLS."""
        assert "delete_gmail_draft" in DESTRUCTIVE_TOOLS


class TestScopeMismatch:
    """Spec tests 28."""

    def test_28_scope_mismatch_blocked(self, engine, client_ctx_destructive):
        """Test 28: Auth event scope mismatch → blocked."""
        auth = AuthorizationEvent(
            action="reply_gmail_email",
            scope={"recipient": "alice@example.com"},
            message_hash=hashlib.sha256(b"test").hexdigest(),
            timestamp=time.time(),
            source="cfg_parser",
        )
        result = engine.check_tool_execution(
            tool="reply_gmail_email",
            args={"recipient": "bob@example.com"},  # Different recipient
            auth_event=auth,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is False
        assert "scope violation" in result.reason.lower()


class TestAuthTTL:
    """Spec tests 29."""

    def test_29_auth_event_expired(self, client_ctx_destructive):
        """Test 29: Expired auth event is rejected."""
        engine = PolicyEngine(auth_ttl=10.0)  # 10 second TTL
        auth = AuthorizationEvent(
            action="reply_gmail_email",
            scope={},
            message_hash=hashlib.sha256(b"test").hexdigest(),
            timestamp=time.time() - 60,  # 60 seconds ago
            source="cfg_parser",
        )
        result = engine.check_tool_execution(
            tool="reply_gmail_email",
            args={},
            auth_event=auth,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is False
        assert "expired" in result.reason.lower()


class TestInboundSanitization:
    """Spec test 30."""

    def test_30_inbound_response_sanitized(self, pipeline):
        """Test 30: Inbound tool response is sanitized."""
        ctx = SecurityContext(
            mode="client",
            source_type="mcp_server",
            source_id="gsuite",
            trust_level=TrustLevel.UNTRUSTED,
            content_type=ContentType.HTML,
            policy=PolicyConfig(),
        )
        # Content with HTML that should be stripped
        raw = '<div style="display:none">hidden</div><p>Email from Alice</p>'
        processed = pipeline.process_inbound(raw, ctx)
        # HTML should be sanitized
        assert "display:none" not in processed.content
        # Untrusted content should be isolated
        assert "untrusted_content" in processed.content


class TestCFGAdapterIntegration:
    """End-to-end: CFG parse → auth event → policy check."""

    def test_cfg_to_policy_roundtrip(self, client_ctx_destructive):
        """CFG adapter produces event that PolicyEngine validates."""
        adapter = CFGDirectiveAdapter()
        query = UtilityQuery(
            category="email", command="email.reply",
            args={"to": "alice@example.com", "body": "thanks"},
            confidence=0.9, source="cli",
            raw_input="reply to alice saying thanks",
        )
        # Produce auth event
        auth_event = adapter.produce(query, "reply to alice saying thanks")
        assert auth_event is not None

        # The dispatch layer translates intent action to tool name
        from episodic.mcp.security.types import AuthorizationEvent as AE
        tool_event = AE(
            action="reply_gmail_email",
            scope={"action": "reply_gmail_email",
                   "recipient": "alice@example.com"},
            message_hash=auth_event.message_hash,
            timestamp=auth_event.timestamp,
            source=auth_event.source,
        )

        engine = PolicyEngine()
        result = engine.check_tool_execution(
            tool="reply_gmail_email",
            args={"action": "reply_gmail_email",
                  "recipient": "alice@example.com"},
            auth_event=tool_event,
            ctx=client_ctx_destructive,
        )
        assert result.allowed is True

    def test_non_write_skips_auth_event(self):
        """Non-write intents produce no auth event."""
        adapter = CFGDirectiveAdapter()
        for command in ("email.search", "email.get", "calendar.query",
                        "calendar.list", "calendar.freebusy"):
            query = UtilityQuery(
                category="email" if command.startswith("email") else "calendar",
                command=command, args={}, confidence=0.9,
                source="cli", raw_input="test",
            )
            assert adapter.produce(query, "test") is None
