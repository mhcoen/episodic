#!/usr/bin/env python3
"""
Automated test of dual-window detection.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from episodic.config import config
from episodic.conversation import ConversationManager
from episodic.db import get_recent_topics
import typer

def main():
    print("Automated Dual-Window Detection Test")
    print("=" * 60)
    
    # Delete existing database
    db_path = Path.home() / '.episodic' / 'episodic.db'
    if db_path.exists():
        print(f"Deleting existing database: {db_path}")
        db_path.unlink()
    
    # Set configuration
    print("\nSetting configuration...")
    config.set('debug', True)
    config.set('use_dual_window_detection', True)
    config.set('use_sliding_window_detection', False)
    config.set('dual_window_high_precision_threshold', 0.2)
    config.set('dual_window_safety_net_threshold', 0.25)
    config.set('skip_llm_response', True)
    config.set('automatic_topic_detection', True)
    
    # Create conversation manager
    print("\nCreating conversation manager...")
    conv_manager = ConversationManager()
    
    # Load test messages
    print("\nLoading test messages from populate_database.txt...")
    script_path = project_root / 'scripts' / 'populate_database.txt'
    
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    # Process messages
    messages = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            messages.append(line)
    
    print(f"\nProcessing {len(messages)} messages...")
    print("-" * 60)
    
    for i, message in enumerate(messages):
        print(f"\n[Message {i+1}/{len(messages)}] {message[:50]}...")
        
        # Send message
        try:
            # Use the actual conversation flow
            _, _ = conv_manager.handle_chat_message(
                user_input=message,
                model=None,  # Will use default
                system_message=None  # Will use default
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        # Check for topic changes
        topics = get_recent_topics(limit=10)
        if topics:
            latest_topic = topics[0]
            print(f"  Current topic: {latest_topic.get('name', 'Unknown')}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    topics = get_recent_topics(limit=20)
    print(f"\nTotal topics created: {len(topics)}")
    
    for i, topic in enumerate(topics):
        print(f"\n{i+1}. {topic.get('name', 'Unknown Topic')}")
        print(f"   Status: {topic.get('status', 'unknown')}")
        if topic.get('start_node_short_id'):
            print(f"   Start: {topic['start_node_short_id']}")
        if topic.get('end_node_short_id'):
            print(f"   End: {topic['end_node_short_id']}")

if __name__ == "__main__":
    main()