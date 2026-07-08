"""
Ancestry-based context recovery strategy.

Wraps the existing context building logic that traverses the DAG ancestry
to build conversation context.
"""

import sqlite3
from typing import Any, Optional
import logging

from .strategy import ContextAssemblyResult, ContextRecoveryMode

logger = logging.getLogger(__name__)


class AncestryStrategy:
    """
    Traditional ancestry-based context recovery.

    Uses DAG ancestry traversal to include recent conversation history,
    regardless of topic boundaries.
    """

    def assemble(
        self,
        user_turn_text: str,
        user_node_id: Optional[str],
        active_topic_start_node_id: Optional[str],
        user_embedding: Optional[Any],
        token_budget: int,
        conn: Optional[sqlite3.Connection] = None,
        chroma_collection: Optional[Any] = None,
    ) -> ContextAssemblyResult:
        """
        Assemble context using DAG ancestry.

        This wraps the existing behavior from ContextBuilder._build_basic_context().
        """
        from episodic.config import config
        from episodic.db import get_ancestry

        messages = []
        included_node_ids = []
        debug = {
            "mode": ContextRecoveryMode.ANCESTRY.value,
            "topic_start_node_id": active_topic_start_node_id,
            "included_node_ids": [],
            "token_counts": {},
            "truncation_info": None,
            "reactivation_fired": False,
        }

        if user_node_id is None:
            # No node yet, return empty context
            return ContextAssemblyResult(messages=messages, debug=debug)

        # Get context depth from config
        context_depth = config.get("context_depth", 10)

        # Get the conversation history from root to current node
        # Note: get_ancestry uses its own connection internally
        conversation_chain = get_ancestry(user_node_id)

        # Newest-first, non-empty nodes only.
        nodes_desc = [
            n for n in reversed(conversation_chain)
            if (n.get("content") or "").strip()
        ]

        # Assemble context = the current user message + up to context_depth
        # complete (assistant, user) exchanges before it. Counting exchanges
        # on completion (not on the boundary) is what makes depth=N yield N
        # prior exchanges rather than N-1.
        filtered_desc = []  # newest-first

        idx = 0
        # Current turn: the trailing user message(s) that have no answer yet.
        if idx < len(nodes_desc) and nodes_desc[idx].get("role") == "user":
            filtered_desc.append(nodes_desc[idx])
            idx += 1

        # Prior exchanges: assistant then user, going backward.
        exchanges = 0
        while idx + 1 < len(nodes_desc) and exchanges < context_depth:
            asst = nodes_desc[idx]
            usr = nodes_desc[idx + 1]
            if asst.get("role") == "assistant" and usr.get("role") == "user":
                filtered_desc.append(asst)
                filtered_desc.append(usr)
                idx += 2
                exchanges += 1
            else:
                break

        # Chronological order (oldest to newest).
        filtered_desc.reverse()

        # Safety net: never start on an assistant message (some providers
        # reject histories that don't begin with a user turn).
        while filtered_desc and filtered_desc[0].get("role") != "user":
            filtered_desc.pop(0)

        filtered_messages = [
            {"role": n.get("role"), "content": n.get("content"), "_node_id": n.get("id")}
            for n in filtered_desc
        ]
        included_node_ids = [n.get("id") for n in filtered_desc]

        # Keep the internal _node_id marker on each message so downstream
        # security tagging (INV-MUSE-3) can identify web-derived content by
        # node id regardless of later system-message insertions. The caller
        # (ContextBuilder.build_context_full) strips it before the LLM call.
        messages = [
            {"role": msg["role"], "content": msg["content"], "_node_id": msg["_node_id"]}
            for msg in filtered_messages
        ]

        # Estimate token counts (rough approximation: 4 chars = 1 token)
        total_chars = sum(len(msg["content"]) for msg in messages)
        debug["token_counts"] = {
            "conversation_history": total_chars // 4,
            "total_estimate": total_chars // 4,
        }
        debug["included_node_ids"] = included_node_ids

        return ContextAssemblyResult(messages=messages, debug=debug)
