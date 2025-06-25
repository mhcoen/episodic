#!/usr/bin/env python3
"""Debug streaming output formatting."""

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

# Configure for immediate streaming
config.set("stream_responses", True)
config.set("stream_constant_rate", False)
config.set("text_wrap", True)

# Test with a message that has explicit line breaks
user_msg = """List two items with descriptions. Format as:
1. Item name - Description
2. Item name - Description"""

print("=== Testing streaming output ===\n")

assistant_id, response = cm.handle_chat_message(
    user_msg,
    model="gpt-4o-mini",
    system_message="You are a helpful assistant. Be brief.",
    context_depth=5
)

print(f"\n\n=== Raw response ===")
print(repr(response))