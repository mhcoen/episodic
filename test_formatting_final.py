#!/usr/bin/env python3
"""Test final formatting with longer response."""

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

# Test both streaming modes
for stream_mode in [False, True]:
    config.set("stream_responses", True)
    config.set("stream_constant_rate", stream_mode)
    config.set("text_wrap", True)
    
    mode_name = "constant-rate" if stream_mode else "immediate"
    print(f"\n=== Testing {mode_name} streaming ===\n")
    
    # Test with a request that produces paragraphs and lists
    user_msg = "List three famous novels with brief descriptions. Include why each is significant."
    
    assistant_id, response = cm.handle_chat_message(
        user_msg,
        model="gpt-4o-mini", 
        system_message="You are a helpful assistant.",
        context_depth=5
    )
    
    print(f"\n{'='*60}")