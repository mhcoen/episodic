#!/usr/bin/env python3
"""Debug script to show topic boundary detection issues."""

import sys
sys.path.insert(0, '.')

from episodic.db import get_recent_nodes, get_recent_topics

# Get topics
topics = get_recent_topics(limit=10)
print("Current Topics:")
print("=" * 80)
for topic in topics:
    print(f"- {topic['name']}: {topic['start_node_id']} → {topic['end_node_id']}")

print("\nTopic Boundaries:")
print("=" * 80)

# Get all nodes
nodes = get_recent_nodes(limit=100)
nodes.reverse()  # Show in chronological order

# Track where topics should logically change based on content
mars_keywords = ['mars', 'rover', 'space', 'terraform', 'astronaut', 'nasa']
cooking_keywords = ['pasta', 'carbonara', 'italian', 'marinara', 'pantry', 'cook']
ml_keywords = ['neural', 'network', 'learning', 'backpropagation', 'gradient', 'transformer']

current_detected_topic = None
user_message_count = 0

for node in nodes:
    short_id = node.get('short_id', '??')
    role = node.get('role', 'unknown')
    
    if role == 'user':
        user_message_count += 1
        content = (node.get('content') or node.get('message', '')).lower()
        
        # Detect topic from content
        detected_topic = None
        if any(kw in content for kw in mars_keywords):
            detected_topic = 'mars'
        elif any(kw in content for kw in cooking_keywords):
            detected_topic = 'cooking'
        elif any(kw in content for kw in ml_keywords):
            detected_topic = 'ml'
        
        if detected_topic and detected_topic != current_detected_topic:
            print(f"\n🔄 TOPIC CHANGE at node {short_id} (user msg #{user_message_count}): {current_detected_topic} → {detected_topic}")
            print(f"   Content: {content[:80]}...")
            current_detected_topic = detected_topic
        
        # Check if this is a topic boundary in the database
        for topic in topics:
            if node['id'] == topic['start_node_id']:
                print(f"   ✅ DB: Start of topic '{topic['name']}'")
            elif node['id'] == topic['end_node_id']:
                print(f"   ✅ DB: End of topic '{topic['name']}'")

print(f"\n\nTotal user messages: {user_message_count}")
print("\nExpected topics based on content:")
print("1. Mars/space exploration (messages 1-5)")
print("2. Italian cooking (messages 6-10)")  
print("3. Machine learning (messages 11-19)")