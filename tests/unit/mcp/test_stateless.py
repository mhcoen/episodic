"""Tests for episodic.mcp.stateless module — stateless LLM tools."""

from unittest.mock import MagicMock, patch

import pytest

from episodic.mcp.stateless import (
    _format_memory_context,
    _format_rag_context,
    ask_llm_stateless,
    index_document,
)


# ===================================================================
# index_document
# ===================================================================

class TestIndexDocument:
    @patch("episodic.rag.get_rag_system")
    def test_indexes_content(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ("doc-123", 3)
        mock_get_rag.return_value = mock_rag

        result = index_document("Hello world", "test.txt")

        assert result["document_id"] == "doc-123"
        assert result["chunks_indexed"] == 3
        assert result["source_name"] == "test.txt"

    @patch("episodic.rag.get_rag_system")
    def test_passes_metadata(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ("doc-1", 1)
        mock_get_rag.return_value = mock_rag

        index_document("content", "src", content_type="markdown", client_id="c1")

        call_kwargs = mock_rag.add_document.call_args[1]
        meta = call_kwargs["metadata"]
        assert meta["indexed_via"] == "mcp"
        assert meta["content_type"] == "markdown"
        assert meta["client_id"] == "c1"
        assert meta["source_name"] == "src"
        assert "indexed_at" in meta

    @patch("episodic.rag.get_rag_system")
    def test_rag_disabled(self, mock_get_rag):
        mock_get_rag.return_value = None

        result = index_document("content", "test.txt")

        assert result["error"] == "unavailable"

    @patch("episodic.rag.get_rag_system")
    def test_no_client_id_in_metadata(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ("doc-1", 1)
        mock_get_rag.return_value = mock_rag

        index_document("content", "src")

        meta = mock_rag.add_document.call_args[1]["metadata"]
        assert "client_id" not in meta

    @patch("episodic.rag.get_rag_system")
    def test_chunk_enabled(self, mock_get_rag):
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ("doc-1", 1)
        mock_get_rag.return_value = mock_rag

        index_document("content", "src")

        assert mock_rag.add_document.call_args[1]["chunk"] is True


# ===================================================================
# ask_llm_stateless
# ===================================================================

class TestAskLlmStateless:
    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_basic_query(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(side_effect=lambda key, default=None: {
            "model": "gpt-4o-mini",
        }.get(key, default))
        mock_llm.return_value = ("Paris", {"input_tokens": 20, "output_tokens": 5})

        result = ask_llm_stateless("Capital of France?")

        assert result["response"] == "Paris"
        assert result["tokens_in"] == 20
        assert result["tokens_out"] == 5
        assert result["model"] == "gpt-4o-mini"
        assert result["provider"] == "openai"

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_no_rag_or_memory_by_default(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        result = ask_llm_stateless("Question")

        assert "rag_sources" not in result
        assert "memory_sources" not in result

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_includes_rag_sources_when_enabled(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        with patch("episodic.mcp.stateless._search_rag", return_value=[
            {"content": "doc text", "metadata": {"source": "file"}}
        ]):
            result = ask_llm_stateless("Question", include_rag=True)

        assert "rag_sources" in result
        assert len(result["rag_sources"]) == 1

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_includes_memory_sources_when_enabled(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        with patch("episodic.mcp.stateless._search_memory", return_value=[
            {"content": "past exchange", "role": "user"}
        ]):
            result = ask_llm_stateless("Question", include_memory=True)

        assert "memory_sources" in result
        assert len(result["memory_sources"]) == 1

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_rag_context_in_system_message(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        with patch("episodic.mcp.stateless._search_rag", return_value=[
            {"content": "relevant doc", "metadata": {"source": "notes"}}
        ]):
            ask_llm_stateless("Question", include_rag=True)

        # Check system message includes context
        messages = mock_llm.call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "Relevant context" in system_msg
        assert "relevant doc" in system_msg

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_custom_rag_query(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        with patch("episodic.mcp.stateless._search_rag") as mock_search:
            mock_search.return_value = []
            ask_llm_stateless("Question", include_rag=True, rag_query="custom query")

        mock_search.assert_called_once_with("custom query", 5)

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_cost_info_missing(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {})

        result = ask_llm_stateless("Q")

        assert result["tokens_in"] == 0
        assert result["tokens_out"] == 0

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_messages_structure(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        ask_llm_stateless("What is Python?")

        messages = mock_llm.call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "What is Python?"}

    @patch("episodic.llm._execute_llm_query")
    @patch("episodic.llm_config.get_current_provider", return_value="openai")
    @patch("episodic.config.config")
    def test_non_streaming(self, mock_config, mock_provider, mock_llm):
        mock_config.get = MagicMock(return_value="gpt-4o-mini")
        mock_llm.return_value = ("Answer", {"input_tokens": 10, "output_tokens": 5})

        ask_llm_stateless("Q")

        assert mock_llm.call_args[1]["stream"] is False


# ===================================================================
# _search_rag / _search_memory helpers
# ===================================================================

class TestSearchHelpers:
    @patch("episodic.rag.get_rag_system")
    def test_search_rag_returns_results(self, mock_get_rag):
        from episodic.mcp.stateless import _search_rag

        mock_rag = MagicMock()
        mock_rag.search.return_value = {
            "results": [{"content": "doc"}],
            "total": 1,
        }
        mock_get_rag.return_value = mock_rag

        results = _search_rag("query", 3)
        assert len(results) == 1

    @patch("episodic.rag.get_rag_system")
    def test_search_rag_disabled(self, mock_get_rag):
        from episodic.mcp.stateless import _search_rag

        mock_get_rag.return_value = None

        results = _search_rag("query", 3)
        assert results == []

    @patch("episodic.rag.get_rag_system")
    def test_search_rag_exception(self, mock_get_rag):
        from episodic.mcp.stateless import _search_rag

        mock_get_rag.side_effect = Exception("boom")

        results = _search_rag("query", 3)
        assert results == []

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_search_memory_returns_results(self, mock_memory):
        from episodic.mcp.stateless import _search_memory

        mock_memory.search_memories.return_value = [{"content": "mem"}]

        results = _search_memory("query", 3)
        assert len(results) == 1

    @patch("episodic.rag_memory_sqlite.memory_rag")
    def test_search_memory_exception(self, mock_memory):
        from episodic.mcp.stateless import _search_memory

        mock_memory.search_memories.side_effect = Exception("boom")

        results = _search_memory("query", 3)
        assert results == []


# ===================================================================
# Formatting helpers
# ===================================================================

class TestFormatHelpers:
    def test_format_rag_context(self):
        sources = [
            {"content": "Python is a language", "metadata": {"source": "wiki"}},
            {"content": "Python 3.12 released", "metadata": {"source_name": "news"}},
        ]
        text = _format_rag_context(sources)
        assert "Documents:" in text
        assert "[1] (wiki):" in text
        assert "[2] (news):" in text
        assert "Python is a language" in text

    def test_format_rag_context_empty(self):
        text = _format_rag_context([])
        assert text == "Documents:"

    def test_format_memory_context(self):
        memories = [
            {"content": "We discussed Python"},
            {"text": "And also JavaScript"},
        ]
        text = _format_memory_context(memories)
        assert "Conversation memories:" in text
        assert "[1]: We discussed Python" in text
        assert "[2]: And also JavaScript" in text

    def test_format_memory_context_empty(self):
        text = _format_memory_context([])
        assert text == "Conversation memories:"

    def test_format_rag_unknown_source(self):
        sources = [{"content": "text", "metadata": {}}]
        text = _format_rag_context(sources)
        assert "(unknown)" in text
