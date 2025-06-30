#!/usr/bin/env python3
"""Show all nodes in order with their topics."""

import sys
sys.path.insert(0, '.')

from episodic.db import get_recent_nodes, get_recent_topics

# Get all nodes
nodes = get_recent_nodes(limit=100)

# Get topics for reference
topics = get_recent_topics(limit=10)
topic_map = {}
for topic in topics:
    start_node = topic['start_node_id']
    end_node = topic['end_node_id']
    topic_map[topic['name']] = (start_node, end_node)

print("All Nodes in Conversation (newest first):")
print("=" * 80)

for i, node in enumerate(nodes):
    if i > 40:  # Limit output
        print(f"... and {len(nodes) - 40} more nodes")
        break
        
    short_id = node.get('short_id', '??')
    
    # Find which topic this node belongs to
    node_topic = "unknown"
    for topic_name, (start, end) in topic_map.items():
        if node['id'] == start:
            node_topic = f"START OF {topic_name}"
            break
        elif node['id'] == end:
            node_topic = f"END OF {topic_name}"
            break
    
    # Check both content and role fields
    if node.get('content'):
        role = node.get('role', 'unknown').upper()
        content = node['content'].replace('\n', ' ')[:60]
        print(f"[{short_id}] {role}: {content}... [{node_topic}]")
    elif node.get('message'):  # Old format - user message
        content = node['message'].replace('\n', ' ')[:60]
        print(f"[{short_id}] USER: {content}... [{node_topic}]")
    elif node.get('response'):  # Old format - assistant response
        content = node['response'].replace('\n', ' ')[:60]
        print(f"[{short_id}] ASST: {content}... [{node_topic}]")
    else:
        # Debug what fields exist
        fields = [k for k in node.keys() if k not in ['id', 'short_id', 'parent_id', 'created_at']]
        print(f"[{short_id}] EMPTY NODE [{node_topic}] Fields: {fields}")