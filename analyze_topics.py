#!/usr/bin/env python3
"""Analyze topic boundaries and messages."""

import sys
sys.path.insert(0, '.')

from episodic.db import get_recent_topics, get_ancestry, get_node

# Get topics
topics = get_recent_topics(limit=10)

print("Topic Analysis")
print("=" * 80)

for i, topic in enumerate(topics):
    print(f"\n[{i+1}] {topic['name']}")
    print(f"    Start: {topic['start_node_id'][:8]}... → End: {'ongoing' if not topic['end_node_id'] else topic['end_node_id'][:8]+'...'}")
    
    # Get nodes in this topic
    if topic['end_node_id']:
        ancestry = get_ancestry(topic['end_node_id'])
    else:
        from episodic.db import get_head
        current_head = get_head()
        ancestry = get_ancestry(current_head) if current_head else []
    
    # Find nodes in topic range
    # Note: ancestry is from newest to oldest, so we need to find the range correctly
    nodes_in_topic = []
    start_idx = -1
    end_idx = -1
    
    for idx, node in enumerate(ancestry):
        if node['id'] == topic['start_node_id']:
            start_idx = idx
        if topic['end_node_id'] and node['id'] == topic['end_node_id']:
            end_idx = idx
            
    if start_idx >= 0:
        if end_idx >= 0:
            # Closed topic: get nodes from end to start (inclusive)
            nodes_in_topic = ancestry[end_idx:start_idx+1]
        else:
            # Ongoing topic: get nodes from current (0) to start
            nodes_in_topic = ancestry[:start_idx+1]
            
    # Reverse to show in chronological order
    nodes_in_topic = list(reversed(nodes_in_topic))
    
    # Show first few messages
    print(f"    Messages ({len(nodes_in_topic)} total):")
    for j, node in enumerate(nodes_in_topic[:4]):  # Show first 4
        if node.get('message'):
            msg = node['message'].replace('\n', ' ')[:60]
            print(f"      [{node['short_id']}] User: {msg}...")
        elif node.get('response'):
            resp = node['response'].replace('\n', ' ')[:60]
            print(f"      [{node['short_id']}] Asst: {resp}...")
    
    if len(nodes_in_topic) > 4:
        print(f"      ... and {len(nodes_in_topic) - 4} more messages")