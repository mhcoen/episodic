#!/usr/bin/env python3
"""Test bold markdown formatting."""

from episodic.conversation import ConversationManager
from episodic.db import initialize_db, insert_node, set_head
from episodic.config import config

# Initialize
initialize_db()
root_id, _ = insert_node("Test", None, role="system")
set_head(root_id)

cm = ConversationManager()
cm.set_current_node_id(root_id)

# Configure for immediate streaming
config.set("stream_responses", True)
config.set("stream_constant_rate", False)
config.set("text_wrap", True)
config.set("debug", False)

# Test with a request that produces bold text
user_msg = "List three famous novels with bold titles using ** markdown."

print("=== Testing bold formatting ===\n")

assistant_id, response = cm.handle_chat_message(
    user_msg,
    model="gpt-4o-mini",
    system_message="You are a helpful assistant. Use markdown bold (**text**) for emphasis.",
    context_depth=5
)

print(f"\n\n=== Raw response ===")
print(repr(response))