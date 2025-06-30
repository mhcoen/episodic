#!/usr/bin/env python3
"""Show actual content of each topic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import get_recent_topics, get_node, get_ancestry

# Get topics
topics = get_recent_topics(limit=10)

print("Topic Contents")
print("=" * 80)

for i, topic in enumerate(topics):
    print(f"\n[{i+1}] {topic['name']} ({topic['start_node_id'][:8]}... → {topic['end_node_id'][:8] if topic['end_node_id'] else 'ongoing'}...)")
    
    # Get all nodes from start to end
    start_node = get_node(topic['start_node_id'])
    end_node = get_node(topic['end_node_id']) if topic['end_node_id'] else None
    
    if not start_node:
        print("    ERROR: Start node not found!")
        continue
        
    print(f"    Start: [{start_node['short_id']}]")
    if end_node:
        print(f"    End: [{end_node['short_id']}]")
    else:
        print(f"    End: ongoing")
    
    # Get ancestry to find all nodes
    if topic['end_node_id']:
        ancestry = get_ancestry(topic['end_node_id'])
    else:
        from episodic.db import get_head
        current_head = get_head()
        if current_head:
            ancestry = get_ancestry(current_head)
        else:
            ancestry = []
    
    # Collect nodes in this topic
    nodes_in_topic = []
    collecting = False
    
    for node in reversed(ancestry):  # Go chronologically
        if node['id'] == topic['start_node_id']:
            collecting = True
            
        if collecting:
            nodes_in_topic.append(node)
            
        if topic['end_node_id'] and node['id'] == topic['end_node_id']:
            break
    
    print(f"    Messages ({len(nodes_in_topic)}):")
    for node in nodes_in_topic:
        if node.get('message'):  # User message
            msg = node['message'].replace('\n', ' ')[:80]
            print(f"      [{node['short_id']}] USER: {msg}")
        elif node.get('response'):  # Assistant response
            resp = node['response'].replace('\n', ' ')[:50] + "..."
            print(f"      [{node['short_id']}] ASST: {resp}")
    
    print()