"""
MCP tools for Episodic.

Seven tools that expose conversation memory and configuration to MCP clients:
  - get_model_info: current model and provider
  - get_runtime_state: curated config subset (no secrets)
  - get_topics: topic list with metadata
  - search_knowledge: RAG document search
  - search_memory: conversation memory search
  - create_thread: create a new conversation thread with handle
  - ask_llm_stateful: stateful LLM conversation via thread handle
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data directory / DB helpers (mirror server.py pattern)
# ---------------------------------------------------------------------------

def _get_data_dir() -> Path:
    return Path(os.environ.get("EPISODIC_DATA_DIR", Path.home() / ".episodic"))


def _get_db_path() -> str:
    return os.environ.get(
        "EPISODIC_DB_PATH", str(_get_data_dir() / "episodic.db")
    )


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _trace_call(tool_name, client_id, parameters, fn):
    """Run fn inside a trace_tool_call context, or fall back to plain call."""
    try:
        from episodic.mcp.trace import trace_tool_call
        conn = _get_db_connection()
        try:
            with trace_tool_call(conn, tool_name, client_id, parameters) as ctx:
                result = fn()
                ctx["output"] = result
            return result
        finally:
            conn.close()
    except Exception:
        # If tracing fails (DB issue, etc.), still run the tool
        return fn()


# ===================================================================
# Tool 1: get_model_info
# ===================================================================

def get_model_info() -> Dict[str, Any]:
    """Return current model names and provider information.

    Returns dict with keys:
        current_model, current_provider, topic_detection_model,
        compression_model, intent_model, synthesis_model
    """
    def _impl():
        try:
            from episodic.config import config
            from episodic.llm_config import get_current_provider
        except ImportError:
            return {"error": "Episodic core not available"}

        return {
            "current_model": config.get("model", "unknown"),
            "current_provider": get_current_provider(),
            "topic_detection_model": config.get("topic_detection_model"),
            "compression_model": config.get("compression_model"),
            "intent_model": config.get("intent_model"),
            "synthesis_model": config.get("synthesis_model"),
        }

    return _trace_call("get_model_info", None, {}, _impl)


# ===================================================================
# Tool 2: get_runtime_state
# ===================================================================

# Curated config keys — safe to expose (no secrets, no API keys)
_RUNTIME_STATE_KEYS = [
    "debug",
    "show_cost",
    "muse_mode",
    "automatic_topic_detection",
    "stream_responses",
    "context_depth",
    "color_mode",
    "topic_strategy",
    "rag_enabled",
]


def get_runtime_state() -> Dict[str, Any]:
    """Return a curated subset of runtime configuration (no secrets).

    Returns dict with 9 safe config keys plus meta fields.
    """
    def _impl():
        try:
            from episodic.config import config
        except ImportError:
            return {"error": "Episodic core not available"}

        state: Dict[str, Any] = {}
        for key in _RUNTIME_STATE_KEYS:
            state[key] = config.get(key)

        state["data_dir"] = str(_get_data_dir())
        state["db_exists"] = Path(_get_db_path()).exists()
        return state

    return _trace_call("get_runtime_state", None, {}, _impl)


# ===================================================================
# Tool 3: get_topics
# ===================================================================

def get_topics(limit: Optional[int] = 50) -> Dict[str, Any]:
    """Return topic list with metadata.

    Args:
        limit: Max topics to return (None for all, default 50).

    Returns dict with keys: topics (list), total (int).
    """
    params = {"limit": limit}

    def _impl():
        try:
            conn = _get_db_connection()
        except Exception as e:
            return {"topics": [], "total": 0, "error": str(e)}

        try:
            cursor = conn.cursor()

            # Check if confidence column exists
            cursor.execute("PRAGMA table_info(topics)")
            columns = [col[1] for col in cursor.fetchall()]
            if not columns:
                return {"topics": [], "total": 0}

            has_confidence = "confidence" in columns

            if has_confidence:
                select = "SELECT name, start_node_id, end_node_id, confidence FROM topics ORDER BY ROWID DESC"
            else:
                select = "SELECT name, start_node_id, end_node_id, NULL as confidence FROM topics ORDER BY ROWID DESC"

            if limit is not None:
                select += f" LIMIT {int(limit)}"

            cursor.execute(select)
            rows = cursor.fetchall()

            topics = []
            for name, start_id, end_id, confidence in rows:
                topics.append({
                    "name": name,
                    "start_node_id": start_id,
                    "end_node_id": end_id,
                    "confidence": confidence,
                })

            return {"topics": topics, "total": len(topics)}
        except Exception as e:
            return {"topics": [], "total": 0, "error": str(e)}
        finally:
            conn.close()

    return _trace_call("get_topics", None, params, _impl)


# ===================================================================
# Tool 4: search_knowledge
# ===================================================================

def search_knowledge(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Search user-indexed documents via RAG.

    Args:
        query: Search query text.
        n_results: Maximum results to return (default 5).

    Returns dict with keys: query, results (list), total (int).
    """
    params = {"query": query, "n_results": n_results}

    if not query or not query.strip():
        return {"query": query, "results": [], "total": 0, "error": "Empty query"}

    def _impl():
        try:
            from episodic.rag import get_rag_system
        except ImportError:
            return {"query": query, "results": [], "total": 0, "error": "RAG not available"}

        rag = get_rag_system()
        if rag is None:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": "RAG system not initialized (rag_enabled may be false)",
            }

        try:
            raw = rag.search(query=query, n_results=n_results)
            return {
                "query": raw.get("query", query),
                "results": raw.get("results", []),
                "total": raw.get("total", 0),
            }
        except Exception as e:
            return {"query": query, "results": [], "total": 0, "error": str(e)}

    return _trace_call("search_knowledge", None, params, _impl)


# ===================================================================
# Tool 5: search_memory
# ===================================================================

def search_memory(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search conversation memory for relevant past exchanges.

    Args:
        query: Search query text.
        limit: Maximum results to return (default 5).

    Returns dict with keys: query, memories (list), total (int).
    """
    params = {"query": query, "limit": limit}

    if not query or not query.strip():
        return {"query": query, "memories": [], "total": 0, "error": "Empty query"}

    def _impl():
        try:
            from episodic.rag_memory_sqlite import memory_rag
        except ImportError:
            return {"query": query, "memories": [], "total": 0, "error": "Memory RAG not available"}

        try:
            memories = memory_rag.search_memories(query=query, limit=limit)
            return {
                "query": query,
                "memories": memories,
                "total": len(memories),
            }
        except Exception as e:
            return {"query": query, "memories": [], "total": 0, "error": str(e)}

    return _trace_call("search_memory", None, params, _impl)


# ===================================================================
# Tool 6: create_thread
# ===================================================================

def create_thread(
    background_influences_topics: bool = False,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new conversation thread and return a handle.

    Args:
        background_influences_topics: Whether this thread's traffic
            affects topic segmentation.
        client_id: ID of the creating client (from auth middleware).

    Returns dict with keys: thread_id, thread_handle, handle_id, permissions.
    """
    params = {"background_influences_topics": background_influences_topics}

    def _impl():
        try:
            from episodic.mcp.threads import create_thread as _create
        except ImportError:
            return {"error": "Thread module not available"}

        try:
            conn = _get_db_connection()
            try:
                result = _create(
                    conn,
                    client_id=client_id or "anonymous",
                    background_influences_topics=background_influences_topics,
                )
                return result
            finally:
                conn.close()
        except Exception as e:
            return {"error": str(e)}

    return _trace_call("create_thread", client_id, params, _impl)


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
# Registration helper — called by server.py
# ===================================================================

def register_tools(server) -> None:
    """Register all tools with a FastMCP server instance."""

    @server.tool()
    def mcp_get_model_info() -> Dict[str, Any]:
        """Get current model and provider information for the Episodic instance."""
        return get_model_info()

    @server.tool()
    def mcp_get_runtime_state() -> Dict[str, Any]:
        """Get curated runtime configuration (no secrets) for the Episodic instance."""
        return get_runtime_state()

    @server.tool()
    def mcp_get_topics(limit: int = 50) -> Dict[str, Any]:
        """Get conversation topics with metadata.

        Args:
            limit: Maximum topics to return (default 50).
        """
        return get_topics(limit=limit)

    @server.tool()
    def mcp_search_knowledge(query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search user-indexed documents via RAG.

        Args:
            query: Search query text.
            n_results: Maximum results to return (default 5).
        """
        return search_knowledge(query=query, n_results=n_results)

    @server.tool()
    def mcp_search_memory(query: str, limit: int = 5) -> Dict[str, Any]:
        """Search conversation memory for relevant past exchanges.

        Args:
            query: Search query text.
            limit: Maximum results to return (default 5).
        """
        return search_memory(query=query, limit=limit)

    @server.tool()
    def mcp_create_thread(
        background_influences_topics: bool = False,
    ) -> Dict[str, Any]:
        """Create a new conversation thread with a handle.

        Returns a thread handle that can be used for stateful conversation
        via ask_llm_stateful. The handle is shown once and cannot be
        retrieved again.

        Args:
            background_influences_topics: Whether this thread's traffic
                affects topic segmentation (default False).
        """
        return create_thread(
            background_influences_topics=background_influences_topics,
        )

    @server.tool()
    def mcp_ask_llm_stateful(
        thread_handle: str,
        message: str,
        purpose: str = "interactive",
    ) -> Dict[str, Any]:
        """Send a message in a stateful conversation thread.

        Validates the thread handle, assembles context from the thread's
        conversation history, calls the LLM, and appends user+assistant
        nodes to the thread's DAG.

        Args:
            thread_handle: Thread handle from create_thread.
            message: User message text.
            purpose: 'interactive' or 'background' (default 'interactive').
        """
        return tool_ask_llm_stateful(
            thread_handle=thread_handle,
            message=message,
            purpose=purpose,
        )
