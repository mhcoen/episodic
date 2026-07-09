"""
MCP LLM tools for Episodic (tools 7-9).

Split out of tools.py to keep it under the size limit:
  - ask_llm_stateful: stateful LLM conversation via thread handle
  - index_document: index content into RAG with provenance
  - ask_llm_stateless: one-shot LLM query with optional RAG/memory context

These delegate their real work to episodic.mcp.stateful / stateless / threads
(lazy-imported inside _impl). The trace wrapper (_trace_call) and DB connection
helper (_get_db_connection) are lazy-imported from episodic.mcp.tools at call
time so that test patches on those names still apply here.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Tool 7: ask_llm_stateful
# ===================================================================

def tool_ask_llm_stateful(
    thread_handle: str,
    message: str,
    purpose: str = "interactive",
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Stateful LLM conversation turn via thread handle.

    Validates the handle, assembles context from thread history,
    calls the LLM, and appends user+assistant nodes to the thread's DAG.

    Args:
        thread_handle: Plaintext thread handle for authentication.
        message: User message text.
        purpose: 'interactive' or 'background'.
        client_id: MCP client ID (for tracing).

    Returns dict with response, node_id, thread_id, tokens_in,
    tokens_out, model, provider. Or error dict on failure.
    """
    from episodic.mcp.tools import _get_db_connection, _trace_call

    params = {"message": message, "purpose": purpose}

    if not message or not message.strip():
        return {"error": "invalid_request", "message": "Empty message"}

    def _impl():
        try:
            from episodic.mcp.threads import validate_thread_handle
            from episodic.mcp.stateful import ask_llm_stateful as _ask
        except ImportError:
            return {"error": "unavailable", "message": "Stateful module not available"}

        try:
            conn = _get_db_connection()
            try:
                # Validate handle (requires write permission)
                handle_info = validate_thread_handle(
                    conn, thread_handle, required_permission="write"
                )
                if handle_info is None:
                    return {
                        "error": "forbidden",
                        "message": "Invalid, revoked, or insufficient permissions",
                    }

                result = _ask(
                    conn,
                    thread_id=handle_info["thread_id"],
                    client_id=handle_info["client_id"],
                    message=message,
                    purpose=purpose,
                )
                return result
            finally:
                conn.close()
        except Exception as e:
            logger.warning("ask_llm_stateful failed: %s", e)
            return {"error": "unavailable", "message": str(e)}

    return _trace_call("ask_llm_stateful", client_id, params, _impl)


# ===================================================================
# Tool 8: index_document
# ===================================================================

def tool_index_document(
    content: str,
    source_name: str,
    content_type: str = "text",
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Index content into the RAG system with MCP provenance.

    Args:
        content: Document text to index.
        source_name: Human-readable source identifier.
        content_type: 'text', 'markdown', or 'code'.
        client_id: MCP client ID (for tracing/provenance).

    Returns dict with document_id, chunks_indexed, source_name.
    Or error dict on failure.
    """
    from episodic.mcp.tools import _trace_call

    params = {"source_name": source_name, "content_type": content_type}

    if not content or not content.strip():
        return {"error": "invalid_request", "message": "Empty content"}

    if not source_name or not source_name.strip():
        return {"error": "invalid_request", "message": "Empty source_name"}

    def _impl():
        try:
            from episodic.mcp.stateless import index_document as _index
        except ImportError:
            return {"error": "unavailable", "message": "Stateless module not available"}

        try:
            return _index(
                content=content,
                source_name=source_name,
                content_type=content_type,
                client_id=client_id,
            )
        except Exception as e:
            logger.warning("index_document failed: %s", e)
            return {"error": "unavailable", "message": str(e)}

    return _trace_call("index_document", client_id, params, _impl)


# ===================================================================
# Tool 9: ask_llm_stateless
# ===================================================================

def tool_ask_llm_stateless(
    message: str,
    include_rag: bool = False,
    rag_query: Optional[str] = None,
    max_rag_results: int = 5,
    include_memory: bool = False,
    memory_query: Optional[str] = None,
    max_memory_results: int = 5,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot LLM query with optional RAG/memory context. No DAG mutation.

    Args:
        message: User message text.
        include_rag: Search RAG for context.
        rag_query: Custom RAG query (defaults to message).
        max_rag_results: Maximum RAG results.
        include_memory: Search conversation memory.
        memory_query: Custom memory query (defaults to message).
        max_memory_results: Maximum memory results.
        client_id: MCP client ID (for tracing).

    Returns dict with response, tokens, model, provider, and
    optional rag_sources/memory_sources. Or error dict on failure.
    """
    from episodic.mcp.tools import _trace_call

    params = {"message": message, "include_rag": include_rag, "include_memory": include_memory}

    if not message or not message.strip():
        return {"error": "invalid_request", "message": "Empty message"}

    def _impl():
        try:
            from episodic.mcp.stateless import ask_llm_stateless as _ask
        except ImportError:
            return {"error": "unavailable", "message": "Stateless module not available"}

        try:
            return _ask(
                message=message,
                include_rag=include_rag,
                rag_query=rag_query,
                max_rag_results=max_rag_results,
                include_memory=include_memory,
                memory_query=memory_query,
                max_memory_results=max_memory_results,
            )
        except Exception as e:
            logger.warning("ask_llm_stateless failed: %s", e)
            return {"error": "unavailable", "message": str(e)}

    return _trace_call("ask_llm_stateless", client_id, params, _impl)
