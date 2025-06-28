#!/usr/bin/env python3
"""Debug script to understand topic behavior."""

from episodic.db import get_connection, get_recent_topics
from episodic.conversation import conversation_manager

# Check what's in the database
print("=== Database Topics ===")
with get_connection() as conn:
    c = conn.cursor()
    c.execute("""
        SELECT name, start_node_id, end_node_id, created_at 
        FROM topics 
        ORDER BY ROWID
    """)
    for row in c.fetchall():
        name, start, end, created = row
        print(f"{name}: {start} -> {end or 'None'}")

print("\n=== Current State ===")
print(f"current_topic: {conversation_manager.current_topic}")
print(f"current_node_id: {conversation_manager.current_node_id}")

# Count messages
with get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM nodes WHERE role='user'")
    user_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM nodes")
    total_count = c.fetchone()[0]
    print(f"Total messages: {total_count} ({user_count} user messages)")