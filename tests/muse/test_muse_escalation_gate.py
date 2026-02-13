"""Muse escalation gate tests — TM-24 through TM-28.

Tests for INV-MUSE-7: enhanced confirmation when web-derived content
is present in the conversation context. Per Erratum 2, ALL tool calls
(read, write, destructive) require enhanced confirmation.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from episodic.config import config
from episodic.mcp.security.action_gate import ActionGate, ActionProposal, _WEB_DERIVED_WARNING
from episodic.mcp.security.types import SecurityContext


@pytest.fixture(autouse=True)
def isolated_config(reset_singletons):
    """Reset config for each test."""
    pass


def _make_proposal(tool_name="test_tool", summary="do something"):
    """Create a test ActionProposal."""
    return ActionProposal(
        tool_name=tool_name,
        args={"arg1": "value1"},
        summary=summary,
        context={},
    )


def _make_ctx_with_handler():
    """Create a SecurityContext with a mock confirmation handler."""
    ctx = MagicMock(spec=SecurityContext)
    ctx.confirmation_handler = MagicMock()
    ctx.confirmation_handler.confirm = AsyncMock(return_value=True)
    return ctx


class TestEnhancedConfirmation:
    """TM-24, TM-25: Enhanced confirmation for tool calls with web-derived context."""

    def test_tm24_write_tool_with_web_derived_gets_enhanced(self):
        """TM-24: Write tool call with web-derived context gets enhanced confirmation."""
        config.set("muse_escalation_gate", True)

        gate = ActionGate()
        proposal = _make_proposal("email.send", "send email")
        ctx = _make_ctx_with_handler()

        asyncio.run(
            gate.confirm(proposal, ctx, context_has_web_derived=True)
        )

        # Verify the handler was called with enhanced warning
        call_args = ctx.confirmation_handler.confirm.call_args
        context_dict = call_args[0][2]  # Third positional arg
        assert context_dict.get("web_derived_warning") == _WEB_DERIVED_WARNING
        assert context_dict.get("enhanced_confirmation") is True

    def test_tm25_write_tool_without_web_derived_normal(self):
        """TM-25: Write tool call without web-derived context gets normal confirmation."""
        config.set("muse_escalation_gate", True)

        gate = ActionGate()
        proposal = _make_proposal("email.send", "send email")
        ctx = _make_ctx_with_handler()

        asyncio.run(
            gate.confirm(proposal, ctx, context_has_web_derived=False)
        )

        call_args = ctx.confirmation_handler.confirm.call_args
        context_dict = call_args[0][2]
        assert "web_derived_warning" not in context_dict


class TestReadToolEscalation:
    """TM-26: Read tools with web-derived context also get enhanced confirmation (Erratum 2)."""

    def test_tm26_read_tool_with_web_derived_gets_enhanced(self):
        """TM-26: Read tool call with web-derived context gets enhanced confirmation."""
        config.set("muse_escalation_gate", True)

        gate = ActionGate()
        proposal = _make_proposal("email.search", "search emails")
        ctx = _make_ctx_with_handler()

        asyncio.run(
            gate.confirm(proposal, ctx, context_has_web_derived=True)
        )

        call_args = ctx.confirmation_handler.confirm.call_args
        context_dict = call_args[0][2]
        assert context_dict.get("web_derived_warning") == _WEB_DERIVED_WARNING
        assert context_dict.get("enhanced_confirmation") is True


class TestHardcodedWarning:
    """TM-27: Enhanced confirmation text is hardcoded, not LLM-generated."""

    def test_tm27_warning_text_is_hardcoded_constant(self):
        """TM-27: Warning text is a module-level constant, not generated."""
        # _WEB_DERIVED_WARNING is a module constant in action_gate.py
        assert isinstance(_WEB_DERIVED_WARNING, str)
        assert "web search" in _WEB_DERIVED_WARNING
        assert "\u26a0\ufe0f" in _WEB_DERIVED_WARNING  # Warning emoji


class TestEscalationGateConfig:
    """TM-28: Config disables escalation gate."""

    def test_tm28_escalation_gate_disabled_no_enhanced(self):
        """TM-28: muse_escalation_gate=false means no enhanced confirmation."""
        config.set("muse_escalation_gate", False)

        gate = ActionGate()
        proposal = _make_proposal("email.send", "send email")
        ctx = _make_ctx_with_handler()

        asyncio.run(
            gate.confirm(proposal, ctx, context_has_web_derived=True)
        )

        call_args = ctx.confirmation_handler.confirm.call_args
        context_dict = call_args[0][2]
        assert "web_derived_warning" not in context_dict
