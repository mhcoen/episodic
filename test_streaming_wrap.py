#!/usr/bin/env python3
"""Test streaming with word wrap."""

import os
from episodic.conversation import ConversationManager
from episodic.db import initialize_db, insert_node, set_head
from episodic.config import config

# Initialize
initialize_db()
root_id, _ = insert_node("Test", None, role="system")
set_head(root_id)

cm = ConversationManager()
cm.set_current_node_id(root_id)

# Configure for immediate streaming with wrapping
config.set("stream_responses", True)
config.set("stream_constant_rate", False)
config.set("text_wrap", True)

print(f"Wrap width: {cm.get_wrap_width()}")
print(f"Text wrap enabled: {config.get('text_wrap', True)}")
print(f"Stream constant rate: {config.get('stream_constant_rate', False)}")

# Test with a request that should produce wrapped output
user_msg = "Write a single long paragraph about space exploration without any line breaks."

print("\n=== Testing streaming with word wrap ===\n")

assistant_id, response = cm.handle_chat_message(
    user_msg,
    model="gpt-4o-mini",
    system_message="You are a helpful assistant. Write in a single paragraph without line breaks.",
    context_depth=5
)

print(f"\n\n=== Stored response length: {len(response)} ===")