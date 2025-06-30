#!/usr/bin/env python3
"""Check actual conversation nodes vs topic boundaries."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import get_recent_nodes, get_recent_topics, get_node

print("Recent Nodes:")
print("=" * 60)
nodes = get_recent_nodes(20)
for node in nodes[:10]:
    role = "User" if node.get('message') else "Asst"
    content = node.get('message') or node.get('response', '')
    content = content.replace('\n', ' ')[:50] + '...'
    print(f"[{node['short_id']}] {node['id'][:8]}... {role}: {content}")

print("\nTopics with Short IDs:")
print("=" * 60)
topics = get_recent_topics(10)
for topic in topics:
    start_node = get_node(topic['start_node_id'])
    end_node = get_node(topic['end_node_id']) if topic['end_node_id'] else None
    
    start_short = start_node['short_id'] if start_node else '??'
    end_short = end_node['short_id'] if end_node else 'ongoing'
    
    print(f"{topic['name']:20} {start_short} → {end_short}")