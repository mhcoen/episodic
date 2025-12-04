#!/usr/bin/env python3
"""Test topic tracking to verify the fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.conversation import conversation_manager
from episodic.db import get_recent_topics, get_connection, initialize_db
from episodic.config import config

# Enable debug mode
config.data["debug"] = True

print("=== Testing Topic Tracking ===\n")

# Initialize fresh database
print("1. Initializing database...")
initialize_db(erase=True)
conversation_manager.current_node_id = None
conversation_manager.current_topic = None
conversation_manager.reset_session_costs()

print(f"   Current topic after init: {conversation_manager.current_topic}")

# Simulate conversation initialization
print("\n2. Initializing conversation...")
conversation_manager.initialize_conversation()
print(f"   Current topic after initialize: {conversation_manager.current_topic}")

# Check database state
print("\n3. Checking database topics:")
topics = get_recent_topics(limit=10)
print(f"   Number of topics: {len(topics)}")
for t in topics:
    print(f"   - {t['name']}: {t.get('start_node_id', 'None')} -> {t.get('end_node_id', 'None')}")

print("\n=== Test Complete ===")