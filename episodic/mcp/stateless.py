"""
Stateless MCP tools: index_document and ask_llm_stateless.

index_document: Indexes content into the RAG system with MCP provenance.
ask_llm_stateless: One-shot LLM query with optional RAG/memory context.
No DAG mutation — ephemeral query-response.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def index_document(
    content: str,
    source_name: str,
    content_type: str = "text",
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Index a document into the RAG system with MCP provenance metadata.

    Args:
        content: Document text to index.
        source_name: Human-readable source identifier.
        content_type: 'text', 'markdown', or 'code'.
        client_id: MCP client ID for provenance.

    Returns:
        Dict with document_id, chunks_indexed, source_name.
        Or error dict on failure.
    """
    from episodic.rag import get_rag_system

    rag = get_rag_system()
    if rag is None:
        return {
            "error": "unavailable",
            "message": "RAG system not initialized (rag_enabled may be false)",
        }

    metadata = {
        "indexed_via": "mcp",
        "content_type": content_type,
        "source_name": source_name,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    if client_id:
        metadata["client_id"] = client_id

    doc_id, chunk_count = rag.add_document(
        content=content,
        source=source_name,
        metadata=metadata,
        chunk=True,
    )

    return {
        "document_id": doc_id,
        "chunks_indexed": chunk_count,
        "source_name": source_name,
    }


def ask_llm_stateless(
    message: str,
    include_rag: bool = False,
    rag_query: Optional[str] = None,
    max_rag_results: int = 5,
    include_memory: bool = False,
    memory_query: Optional[str] = None,
    max_memory_results: int = 5,
) -> Dict[str, Any]:
    """One-shot LLM query with optional RAG/memory context. No DAG mutation.

    Args:
        message: User message text.
        include_rag: Whether to search RAG for context.
        rag_query: Custom RAG search query (defaults to message).
        max_rag_results: Maximum RAG results to include.
        include_memory: Whether to search conversation memory.
        memory_query: Custom memory search query (defaults to message).
        max_memory_results: Maximum memory results to include.

    Returns:
        Dict with response, tokens_in, tokens_out, model, provider,
        and optional rag_sources/memory_sources.
    """
    from episodic.config import config
    from episodic.llm import _execute_llm_query
    from episodic.llm_config import get_current_provider

    model = config.get("model", "gpt-4o-mini")
    provider = get_current_provider()

    # Gather optional context
    rag_sources: List[Dict[str, Any]] = []
    memory_sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []

    if include_rag:
        rag_sources = _search_rag(rag_query or message, max_rag_results)
        if rag_sources:
            context_parts.append(_format_rag_context(rag_sources))

    if include_memory:
        memory_sources = _search_memory(memory_query or message, max_memory_results)
        if memory_sources:
            context_parts.append(_format_memory_context(memory_sources))

    # Build messages
    system_content = "You are a helpful assistant."
    if context_parts:
        system_content += "\n\nRelevant context:\n" + "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": message},
    ]

    # Call LLM (non-streaming for MCP)
    response_text, cost_info = _execute_llm_query(
        messages=messages,
        model=model,
        stream=False,
    )

    tokens_in = cost_info.get("input_tokens", 0) if cost_info else 0
    tokens_out = cost_info.get("output_tokens", 0) if cost_info else 0

    result: Dict[str, Any] = {
        "response": response_text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
        "provider": provider,
    }

    if include_rag:
        result["rag_sources"] = rag_sources
    if include_memory:
        result["memory_sources"] = memory_sources

    return result


def _search_rag(query: str, n_results: int) -> List[Dict[str, Any]]:
    """Search RAG system, return list of source dicts."""
    try:
        from episodic.rag import get_rag_system

        rag = get_rag_system()
        if rag is None:
            return []

        raw = rag.search(query=query, n_results=n_results)
        return raw.get("results", [])
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return []


def _search_memory(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search conversation memory, return list of memory dicts."""
    try:
        from episodic.rag_memory_sqlite import memory_rag

        return memory_rag.search_memories(query=query, limit=limit)
    except Exception as e:
        logger.warning("Memory search failed: %s", e)
        return []


def _format_rag_context(sources: List[Dict[str, Any]]) -> str:
    """Format RAG results into context text for the system message."""
    lines = ["Documents:"]
    for i, src in enumerate(sources, 1):
        content = src.get("content", "")
        meta = src.get("metadata", {})
        source_name = meta.get("source_name", meta.get("source", "unknown"))
        lines.append(f"[{i}] ({source_name}): {content}")
    return "\n".join(lines)


def _format_memory_context(memories: List[Dict[str, Any]]) -> str:
    """Format memory results into context text for the system message."""
    lines = ["Conversation memories:"]
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", mem.get("text", ""))
        lines.append(f"[{i}]: {content}")
    return "\n".join(lines)
