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

        # conversation_chain is from oldest to newest, we want newest first for filtering
        reversed_chain = list(reversed(conversation_chain))

        # Count exchanges (user + assistant pairs)
        exchange_count = 0
        last_role = None
        filtered_messages = []

        for node in reversed_chain:
            # Skip empty messages
            content = node.get("content")
            if not content or not content.strip():
                continue

            node_id = node.get("id")
            current_role = node.get("role")

            # Track when we complete an exchange
            if last_role == "assistant" and current_role == "user":
                exchange_count += 1

            # Stop if we've collected enough exchanges
            if exchange_count >= context_depth:
                break

            # Add the message to our filtered list
            filtered_messages.append({
                "role": current_role,
                "content": content,
                "_node_id": node_id,
            })
            included_node_ids.append(node_id)

            last_role = current_role

        # Reverse back to chronological order (oldest to newest)
        filtered_messages.reverse()
        included_node_ids.reverse()

        # Always ensure we have an even number of messages (complete exchanges)
        if len(filtered_messages) % 2 != 0 and len(filtered_messages) > 1:
            # Remove the oldest message if we have an odd number
            filtered_messages.pop(0)
            included_node_ids.pop(0)

        # Ensure we start with a user message
        while filtered_messages and filtered_messages[0]["role"] != "user":
            filtered_messages.pop(0)
            included_node_ids.pop(0)

        # Build final messages list (strip internal _node_id)
        messages = [
            {"role": msg["role"], "content": msg["content"]}
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
