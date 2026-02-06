"""
Stateful LLM conversation via thread handles.

Provides ask_llm_stateful: validates thread handle, assembles context
from thread's node ancestry, calls LLM, appends user+assistant nodes
to the thread's DAG, and updates the thread's head pointer.

Does NOT touch the global state.head_id — only the thread's
conversations.current_head_id is updated.
"""

import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _insert_thread_node(
    conn: sqlite3.Connection,
    thread_id: int,
    content: str,
    parent_id: Optional[str] = None,
    role: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Insert a node into the DAG and update the thread's head pointer.

    Unlike db_nodes.insert_node(), this does NOT update the global
    state.head_id — only conversations.current_head_id for the thread.

    Args:
        conn: Database connection (caller manages transaction).
        thread_id: The conversation/thread ID.
        content: Node content text.
        parent_id: Parent node ID (None for root).
        role: Node role ('user', 'assistant', 'system').
        provider: LLM provider name.
        model: LLM model name.

    Returns:
        (node_id, short_id)
    """
    from episodic.db_ids import generate_short_id

    node_id = str(uuid.uuid4())
    short_id = generate_short_id()

    conn.execute(
        "INSERT INTO nodes (id, short_id, parent_id, content, role, provider, model) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (node_id, short_id, parent_id, content, role, provider, model),
    )

    # Update thread head — NOT the global state table
    conn.execute(
        "UPDATE conversations SET current_head_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (node_id, thread_id),
    )

    return node_id, short_id


def _get_thread_head(conn: sqlite3.Connection, thread_id: int) -> Optional[str]:
    """Get the current head node ID for a thread."""
    row = conn.execute(
        "SELECT current_head_id FROM conversations WHERE id = ?",
        (thread_id,),
    ).fetchone()
    return row[0] if row else None


def _get_thread_ancestry(
    conn: sqlite3.Connection,
    node_id: str,
) -> List[Dict[str, Any]]:
    """Walk parent_id chain from node to root, return oldest-first."""
    ancestry = []
    current_id = node_id

    while current_id:
        row = conn.execute(
            "SELECT id, parent_id, content, role, provider, model, created_at "
            "FROM nodes WHERE id = ?",
            (current_id,),
        ).fetchone()
        if row is None:
            break
        ancestry.append({
            "id": row[0],
            "parent_id": row[1],
            "content": row[2],
            "role": row[3],
            "provider": row[4],
            "model": row[5],
            "created_at": row[6],
        })
        current_id = row[1]

    ancestry.reverse()  # oldest first
    return ancestry


def _build_context_messages(
    ancestry: List[Dict[str, Any]],
    user_message: str,
    system_message: str = "You are a helpful assistant.",
    context_depth: int = 5,
) -> List[Dict[str, str]]:
    """Build LLM message list from thread ancestry.

    Args:
        ancestry: Nodes oldest-first (from _get_thread_ancestry).
        user_message: The new user message to append.
        system_message: System prompt.
        context_depth: Number of exchanges (user+assistant pairs) to include.

    Returns:
        Messages list for LLM API call.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_message},
    ]

    # Take last context_depth exchanges from ancestry
    # An exchange = one user + one assistant message
    relevant = []
    exchange_count = 0
    last_role = None

    for node in reversed(ancestry):
        role = node.get("role")
        if role not in ("user", "assistant"):
            continue
        if last_role == "user" and role == "assistant":
            exchange_count += 1
        if exchange_count >= context_depth:
            break
        relevant.append(node)
        last_role = role

    relevant.reverse()  # back to chronological

    for node in relevant:
        role = node.get("role")
        content = node.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    return messages


def ask_llm_stateful(
    conn: sqlite3.Connection,
    thread_id: int,
    client_id: str,
    message: str,
    purpose: str = "interactive",
    system_message: str = "You are a helpful assistant.",
    context_depth: int = 5,
) -> Dict[str, Any]:
    """Execute a stateful LLM conversation turn on a thread.

    1. Get thread's current head node
    2. Build context from ancestry
    3. Call LLM (non-streaming)
    4. Insert user + assistant nodes into DAG
    5. Update thread head
    6. Return response with metadata

    Args:
        conn: Database connection.
        thread_id: Conversation/thread ID.
        client_id: MCP client ID (for tracing).
        message: User message text.
        purpose: 'interactive' or 'background'.
        system_message: System prompt for LLM.
        context_depth: Number of exchanges to include in context.

    Returns:
        Dict with response, node_id, thread_id, tokens_in, tokens_out,
        model, provider.
    """
    from episodic.config import config
    from episodic.llm import _execute_llm_query
    from episodic.llm_config import get_current_provider

    model = config.get("model", "gpt-4o-mini")
    provider = get_current_provider()

    # 1. Get thread's current head
    head_id = _get_thread_head(conn, thread_id)

    # 2. Build context from ancestry
    if head_id:
        ancestry = _get_thread_ancestry(conn, head_id)
    else:
        ancestry = []

    messages = _build_context_messages(
        ancestry, message, system_message, context_depth
    )

    # 3. Call LLM (non-streaming for MCP)
    response_text, cost_info = _execute_llm_query(
        messages=messages,
        model=model,
        stream=False,
    )

    # 4. Insert user node
    user_node_id, _ = _insert_thread_node(
        conn, thread_id, message,
        parent_id=head_id,
        role="user",
    )

    # 5. Insert assistant node
    assistant_node_id, _ = _insert_thread_node(
        conn, thread_id, response_text,
        parent_id=user_node_id,
        role="assistant",
        provider=provider,
        model=model,
    )

    conn.commit()

    # 6. Return structured response
    tokens_in = cost_info.get("input_tokens", 0) if cost_info else 0
    tokens_out = cost_info.get("output_tokens", 0) if cost_info else 0

    return {
        "response": response_text,
        "node_id": assistant_node_id,
        "thread_id": thread_id,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
        "provider": provider,
    }
