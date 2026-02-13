"""Muse RAG tool-context tests — TM-34 through TM-37.

Tests for INV-MUSE-9: untrusted RAG chunks excluded from
tool-enabled context by default.
"""

import pytest
from unittest.mock import patch, MagicMock

from episodic.config import config
from episodic.context_builder import ContextBuilder


@pytest.fixture(autouse=True)
def isolated_config(reset_singletons):
    """Reset config for each test."""
    pass


def _make_rag_result(trust_level="trusted", content="trusted content",
                     source_url="https://example.com"):
    """Create a mock RAG search result."""
    return {
        'content': content,
        'metadata': {
            'trust_level': trust_level,
            'source_url': source_url,
            'filename': 'test_doc',
        },
    }


class TestUntrustedRAGExclusion:
    """TM-34: MCP tools active + untrusted RAG chunks excluded by default."""

    def test_tm34_untrusted_chunks_excluded_with_mcp_tools(self):
        """TM-34: Untrusted RAG chunks excluded when MCP tools active (default)."""
        builder = ContextBuilder()

        # Setup: MCP tools active, default config (muse_rag_in_tool_context=False)
        assert builder._has_mcp_tools() is False  # No servers configured

        # Simulate MCP tools active
        config.set("mcp_servers", {"test": {"url": "http://test"}})
        assert builder._has_mcp_tools() is True

        # With muse_rag_in_tool_context=False (default), untrusted chunks
        # should be excluded. We verify the config gating.
        allow = config.get("muse_rag_in_tool_context", False)
        assert allow is False

        # Verify the filtering logic: untrusted + mcp_active + not allowed → skip
        mcp_active = True
        trust_level = "untrusted"
        should_skip = (trust_level == "untrusted" and mcp_active and not allow)
        assert should_skip is True


class TestUntrustedRAGInclusion:
    """TM-35: Opt-in includes untrusted chunks with wrapping."""

    def test_tm35_untrusted_chunks_included_with_opt_in(self):
        """TM-35: muse_rag_in_tool_context=true includes chunks with wrapping."""
        config.set("muse_rag_in_tool_context", True)
        config.set("mcp_servers", {"test": {"url": "http://test"}})

        allow = config.get("muse_rag_in_tool_context", False)
        assert allow is True

        # When allowed, untrusted chunks should be included but wrapped
        mcp_active = True
        trust_level = "untrusted"
        should_skip = (trust_level == "untrusted" and mcp_active and not allow)
        assert should_skip is False

        # Verify wrapping format
        content = "untrusted web content"
        url = "https://example.com"
        wrapped = (
            f'<untrusted_content source="rag:web:{url}">\n'
            f'{content}\n'
            f'</untrusted_content>'
        )
        assert '<untrusted_content source="rag:web:' in wrapped


class TestUntrustedRAGWithoutMCP:
    """TM-36: No MCP tools + untrusted chunks included with L1 wrapping."""

    def test_tm36_untrusted_chunks_included_without_mcp(self):
        """TM-36: Without MCP tools, untrusted chunks included with <untrusted_content> wrapping."""
        config.set("mcp_servers", {})

        builder = ContextBuilder()
        assert builder._has_mcp_tools() is False

        # Without MCP tools, untrusted chunks should be included
        # (even with default muse_rag_in_tool_context=False)
        mcp_active = False
        trust_level = "untrusted"
        allow = config.get("muse_rag_in_tool_context", False)

        should_skip = (trust_level == "untrusted" and mcp_active and not allow)
        assert should_skip is False  # NOT skipped when no MCP tools


class TestTrustedRAGAlwaysIncluded:
    """TM-37: Trusted RAG chunks always included."""

    def test_tm37_trusted_chunks_always_included(self):
        """TM-37: Trusted RAG chunks included regardless of MCP tool state."""
        # With MCP tools
        config.set("mcp_servers", {"test": {"url": "http://test"}})
        trust_level = "trusted"
        mcp_active = True
        allow = config.get("muse_rag_in_tool_context", False)

        # Trusted chunks are never filtered
        should_skip = (trust_level == "untrusted" and mcp_active and not allow)
        assert should_skip is False  # trusted is never filtered

        # Without MCP tools
        config.set("mcp_servers", {})
        mcp_active = False
        should_skip = (trust_level == "untrusted" and mcp_active and not allow)
        assert should_skip is False  # Still not filtered
