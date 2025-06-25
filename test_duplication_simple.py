#!/usr/bin/env python3
"""
Simple test to check for output duplication in streaming mode.
"""

import os
import sys

# Set debug mode
os.environ["EPISODIC_DEBUG"] = "true"

# Import after setting env var
from episodic.conversation import ConversationManager
from episodic.db import initialize_db, insert_node, set_head
from episodic.config import config

def test_streaming():
    """Test streaming output for duplication."""
    print("Testing streaming output...\n")
    
    # Initialize
    initialize_db()
    
    # Create a simple conversation
    root_id, _ = insert_node("Test", None, role="system")
    set_head(root_id)
    
    # Create conversation manager
    cm = ConversationManager()
    cm.set_current_node_id(root_id)
    
    # Enable streaming
    config.set("stream_responses", True)
    config.set("debug", False)  # Disable debug for clean output
    config.set("stream_constant_rate", True)  # Test constant-rate streaming
    
    print("=== Testing with streaming enabled ===")
    
    # Test message with longer response
    user_msg = "List the top 3 sci-fi movies with brief descriptions. Use **bold** for titles."
    
    # Handle the message
    assistant_id, response = cm.handle_chat_message(
        user_msg,
        model="gpt-4o-mini",
        system_message="You are a helpful assistant. Be very brief.",
        context_depth=5
    )
    
    print(f"\n=== Response stored in DB: {repr(response)} ===")

if __name__ == "__main__":
    test_streaming()