#!/usr/bin/env python3
"""Debug word wrapping."""

from episodic.conversation import ConversationManager
from episodic.db import initialize_db, insert_node, set_head
from episodic.config import config

# Initialize
initialize_db()
root_id, _ = insert_node("Test", None, role="system")
set_head(root_id)

cm = ConversationManager()
cm.set_current_node_id(root_id)

# Enable debug mode
config.set("stream_responses", True)
config.set("stream_constant_rate", False)
config.set("text_wrap", True)
config.set("debug", True)

print(f"Terminal wrap width: {cm.get_wrap_width()}")

# Test with a message that should produce a long line
user_msg = "Write a single sentence about space exploration that is very long and detailed."

print("\n=== Testing word wrap ===\n")

assistant_id, response = cm.handle_chat_message(
    user_msg,
    model="gpt-4o-mini",
    system_message="Write very long sentences without line breaks.",
    context_depth=5
)