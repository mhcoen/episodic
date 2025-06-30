#!/usr/bin/env python3
"""Test hybrid topic detection integration with main app."""

import sys
sys.path.insert(0, '.')

from episodic.conversation import ConversationManager
from episodic.config import config
from episodic.db import initialize_db, get_recent_nodes
import typer

# Initialize database
initialize_db()

# Enable hybrid detection
config.set("use_hybrid_topic_detection", True)
config.set("debug", True)
config.set("hybrid_topic_weights", {
    "semantic_drift": 0.6,
    "keyword_explicit": 0.25,
    "keyword_domain": 0.1,
    "message_gap": 0.025,
    "conversation_flow": 0.025
})
config.set("hybrid_topic_threshold", 0.55)
config.set("hybrid_llm_threshold", 0.3)

# Create conversation manager
conv_mgr = ConversationManager()

# Simulate a conversation
test_messages = [
    "Hello, I'd like to learn about machine learning",
    "What are neural networks?",
    "Let's change topics. What's a good pasta recipe?",
    "How long should I cook spaghetti?",
]

print("Testing conversation flow with hybrid topic detection:\n")

for i, msg in enumerate(test_messages):
    print(f"\n{'='*60}")
    print(f"User message {i+1}: {msg}")
    
    # Get recent nodes before sending message
    recent = get_recent_nodes(limit=10)
    print(f"Recent nodes: {len(recent)}")
    
    # This would normally call handle_chat_message but we'll simulate the detection part
    try:
        # Check if we can import and use the hybrid detector
        from episodic.topics_hybrid import detect_topic_change_hybrid
        
        if len(recent) >= 1:  # Need at least 1 previous message
            changed, new_topic, metadata = detect_topic_change_hybrid(
                recent, 
                msg,
                current_topic=conv_mgr.get_current_topic()
            )
            
            print(f"\nTopic detection result:")
            print(f"  Changed: {changed}")
            print(f"  Score: {metadata.get('score', 'N/A')}")
            print(f"  Method: {metadata.get('method', 'N/A')}")
            print(f"  Explanation: {metadata.get('explanation', 'N/A')}")
            if metadata.get('signals'):
                print(f"  Signals: {metadata['signals']}")
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Integration test completed!")