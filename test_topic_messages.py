#!/usr/bin/env python3
"""Test topic message counting fix."""

import sys
sys.path.insert(0, '.')

from episodic.db import get_recent_topics
from episodic.topics import count_nodes_in_topic

# Get recent topics
topics = get_recent_topics(limit=10)

print("Topic Message Count Test")
print("=" * 60)

for i, topic in enumerate(topics):
    name = topic['name']
    start_id = topic['start_node_id']
    end_id = topic['end_node_id']
    
    # Count messages
    count = count_nodes_in_topic(start_id, end_id)
    
    status = "✓" if end_id else "○"
    end_str = "ongoing" if not end_id else f"node {end_id[:8]}"
    
    print(f"\n[{i+1}] {status} {name}")
    print(f"    Start: {start_id[:8]}... → End: {end_str}")
    print(f"    Messages: {count}")

print("\n" + "=" * 60)
print("Test complete!")