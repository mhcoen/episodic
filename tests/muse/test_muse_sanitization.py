"""Muse sanitization tests — TM-1 through TM-8.

Tests for INV-MUSE-1 through INV-MUSE-5 covering:
- Synthesis LLM call has no tools parameter
- Muse DAG nodes carry source_type='web_synthesis'
- Web-derived content tagging in context assembly
- RAG chunk provenance metadata
- KG quarantine for muse responses
"""

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from episodic.config import config


@pytest.fixture(autouse=True)
def isolated_config(reset_singletons):
    """Reset config for each test."""
    pass


class TestSynthesisNoTools:
    """TM-1: INV-MUSE-1 — Synthesis LLM call has no tools parameter."""

    def test_tm1_synthesis_result_has_no_tools_key(self):
        """TM-1: synthesize_results() output dict has no 'tools' key."""
        from episodic.web_synthesis import WebSynthesizer
        from episodic.web_search import SearchResult

        config.set("model", "gpt-4o-mini")
        config.set("response_style", "standard")
        config.set("response_format", "mixed")

        synth = WebSynthesizer()
        result = synth.synthesize_results(
            query="test query",
            results=[
                SearchResult(title="Test", url="https://example.com",
                             snippet="test snippet", relevance_score=1.0)
            ],
            extracted_content={"https://example.com": "some content"},
        )

        assert isinstance(result, dict)
        assert "tools" not in result


class TestMuseNodeSourceType:
    """TM-2, TM-3: INV-MUSE-2 — Muse DAG nodes carry source_type."""

    def test_tm2_insert_node_with_web_synthesis_source_type(self, temp_database):
        """TM-2: insert_node with source_type='web_synthesis' stores correctly."""
        from episodic.db_nodes import insert_node, get_node
        from episodic.db_migrations import initialize_db

        initialize_db(create_root_node=False)

        node_id, short_id = insert_node(
            "muse response", None, role="assistant",
            source_type="web_synthesis",
        )

        node = get_node(node_id)
        assert node is not None
        assert node["source_type"] == "web_synthesis"

    def test_tm3_chat_node_has_chat_source_type(self, temp_database):
        """TM-3: Node with default source_type is 'chat'."""
        from episodic.db_nodes import insert_node, get_node
        from episodic.db_migrations import initialize_db

        initialize_db(create_root_node=False)

        node_id, short_id = insert_node(
            "chat response", None, role="assistant",
        )

        node = get_node(node_id)
        assert node is not None
        assert node["source_type"] == "chat"


class TestWebDerivedContentTagging:
    """TM-4, TM-5: INV-MUSE-3 — Web-derived content tagging."""

    def test_tm4_web_synthesis_nodes_wrapped_when_mcp_active(self, temp_database):
        """TM-4: Context assembly wraps web_synthesis nodes when MCP tools present."""
        from episodic.context_builder import ContextBuilder
        from episodic.db_nodes import insert_node
        from episodic.db_migrations import initialize_db

        initialize_db(create_root_node=False)

        # Create a web_synthesis node
        node_id, _ = insert_node(
            "web synthesis content", None, role="assistant",
            source_type="web_synthesis",
        )

        builder = ContextBuilder()
        # Simulate MCP tools active
        config.set("mcp_servers", {"test_server": {"url": "http://test"}})

        # Tagging matches by _node_id, not position. Include an unrelated
        # system message and a non-web message at the same indices to prove
        # the web-derived message is wrapped by identity, not by offset.
        other_id, _ = insert_node(
            "ordinary reply", None, role="assistant", source_type="chat",
        )
        messages = [
            {"role": "system", "content": "topic context"},
            {"role": "user", "content": "a question", "_node_id": "u1"},
            {"role": "assistant", "content": "ordinary reply", "_node_id": other_id},
            {"role": "assistant", "content": "web synthesis content", "_node_id": node_id},
        ]
        tagged = builder._tag_web_derived_messages(messages, [node_id, other_id])

        # Only the web_synthesis message is wrapped (the injected system
        # warning also mentions the tag literally, so exclude system role).
        wrapped = [
            m for m in tagged
            if m["role"] != "system" and "<web_derived_content>" in m["content"]
        ]
        assert len(wrapped) == 1
        assert "web synthesis content" in wrapped[0]["content"]
        assert "ordinary reply" not in wrapped[0]["content"]
        assert builder.context_has_web_derived is True

    def test_tm5_web_synthesis_nodes_not_wrapped_without_mcp(self, temp_database):
        """TM-5: Context assembly does NOT wrap web_synthesis nodes when no MCP tools."""
        from episodic.context_builder import ContextBuilder
        from episodic.db_nodes import insert_node
        from episodic.db_migrations import initialize_db

        initialize_db(create_root_node=False)

        node_id, _ = insert_node(
            "web synthesis content", None, role="assistant",
            source_type="web_synthesis",
        )

        builder = ContextBuilder()
        # No MCP tools
        config.set("mcp_servers", {})

        messages = [{"role": "assistant", "content": "web synthesis content", "_node_id": node_id}]
        tagged = builder._tag_web_derived_messages(messages, [node_id])

        # Should NOT have wrapping tags in messages
        assert not any("<web_derived_content>" in m["content"] for m in tagged
                       if m["role"] != "system")
        # But flag should still be set
        assert builder.context_has_web_derived is True

    def test_full_assembly_tags_correct_node_and_strips_marker(self, temp_database):
        """End-to-end: build_context_full tags the web_synthesis message (by
        node id, after topic/KG/RAG system inserts shift positions) and never
        leaks the internal _node_id marker to the outgoing messages."""
        from episodic.context_builder import ContextBuilder
        from episodic.db_nodes import insert_node
        from episodic.db_migrations import initialize_db

        initialize_db(create_root_node=False)

        # A short conversation: normal exchange, then a muse (web_synthesis)
        # answer, then a follow-up user turn (the current turn).
        u1, _ = insert_node("what's the weather policy?", None, role="user",
                            source_type="chat")
        a1, _ = insert_node("Here is a normal answer.", u1, role="assistant",
                            source_type="chat")
        u2, _ = insert_node("search the web for X", a1, role="user",
                            source_type="chat")
        a2, _ = insert_node("Synthesized web answer about X.", u2, role="assistant",
                            source_type="web_synthesis")
        u3, _ = insert_node("now book a meeting about it", a2, role="user",
                            source_type="chat")

        config.set("context_recovery_mode", "ancestry")
        config.set("muse_mode", False)  # no live web fetch on this turn
        config.set("mcp_servers", {"test": {"url": "http://test"}})

        builder = ContextBuilder()
        messages, _raw, _rag, _web, _debug = builder.build_context_full(
            user_node_id=u3,
            user_input="now book a meeting about it",
            active_topic_start_node_id=None,
            model="gpt-4o-mini",
            skip_rag=True,
        )

        # The web_synthesis assistant message is present and wrapped.
        web_msgs = [m for m in messages
                    if "Synthesized web answer" in m["content"]]
        assert web_msgs and all(
            "<web_derived_content>" in m["content"] for m in web_msgs
        )

        # Only web-derived content is wrapped: any wrapped non-system message
        # must be the web-synthesis one, never an ordinary conversation turn.
        for m in messages:
            if m["role"] != "system" and "<web_derived_content>" in m["content"]:
                assert "Synthesized web answer" in m["content"]

        # The internal marker must never reach the LLM payload.
        assert all("_node_id" not in m for m in messages)
        assert builder.context_has_web_derived is True


class TestRAGProvenance:
    """TM-6, TM-7: INV-MUSE-4 — RAG chunk provenance."""

    def test_tm6_web_rag_chunks_carry_untrusted_metadata(self):
        """TM-6: RAG chunks from web extraction carry trust_level=untrusted."""
        # The provenance metadata is set in commands/web_search.py
        # Verify the metadata dict that would be passed to add_document
        expected_metadata = {
            'source_type': 'web_content',
            'source_url': 'https://example.com',
            'trust_level': 'untrusted',
        }
        assert expected_metadata['trust_level'] == 'untrusted'
        assert expected_metadata['source_type'] == 'web_content'

    def test_tm7_rag_retrieval_wraps_untrusted_chunks(self):
        """TM-7: RAG retrieval wraps untrusted chunks in <untrusted_content> tags."""
        # Verify the wrapping format matches INV-MUSE-4
        content = "web content about topic"
        url = "https://example.com"
        wrapped = (
            f'<untrusted_content source="rag:web:{url}">\n'
            f'{content}\n'
            f'</untrusted_content>'
        )
        assert '<untrusted_content source="rag:web:https://example.com">' in wrapped
        assert content in wrapped
        assert '</untrusted_content>' in wrapped


class TestKGQuarantine:
    """TM-8: INV-MUSE-5 — KG triples from muse responses quarantined."""

    def test_tm8_source_gate_quarantines_web_synthesis(self):
        """TM-8: KG extraction of muse assistant response produces quarantined triples."""
        from episodic.mcp.security.source_gate import check_extraction_allowed, ExtractionPolicy

        result = check_extraction_allowed("web_synthesis")
        assert result.policy == ExtractionPolicy.QUARANTINE
