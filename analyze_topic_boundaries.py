#!/usr/bin/env python3
"""Analyze why topic boundaries are misaligned."""

import sys
sys.path.insert(0, '.')

from episodic.db import get_recent_nodes

# Get nodes in chronological order
nodes = list(reversed(get_recent_nodes(limit=50)))

print("Topic Boundary Analysis")
print("=" * 80)
print("\nShowing conversation flow with actual vs expected topics:\n")

current_topic = "Unknown"
expected_topic = None

for i, node in enumerate(nodes):
    if not node.get('content'):
        continue
        
    short_id = node.get('short_id', '??')
    role = node.get('role', 'unknown').upper()
    content = node['content'].replace('\n', ' ')[:60] + '...'
    
    # Determine expected topic based on content
    content_lower = node['content'].lower()
    
    if any(word in content_lower for word in ['mars', 'rover', 'astronaut', 'terraform', 'space']):
        expected_topic = "Mars/Space"
    elif any(word in content_lower for word in ['pasta', 'italian', 'carbonara', 'marinara', 'cooking', 'recipe', 'ingredient', 'pantry']):
        expected_topic = "Italian Cooking"
    elif any(word in content_lower for word in ['neural', 'network', 'machine learning', 'backpropagation', 'activation', 'supervised', 'unsupervised', 'transformer', 'dropout', 'gradient', 'deep learning']):
        expected_topic = "ML/Neural Networks"
    
    # Mark topic boundaries
    boundary_marker = ""
    if short_id == '02':
        boundary_marker = " [START: mars-rover]"
        current_topic = "mars-rover"
    elif short_id == '09':
        boundary_marker = " [END: mars-rover]"
    elif short_id == '0a':
        boundary_marker = " [START: space-exploration] ❌"  # Wrong!
        current_topic = "space-exploration"
    elif short_id == '0f':
        boundary_marker = " [END: space-exploration]"
    elif short_id == '0g':
        boundary_marker = " [START: pasta-cooking] ❌"  # Wrong!
        current_topic = "pasta-cooking"
    elif short_id == '0j':
        boundary_marker = " [END: pasta-cooking]"
    elif short_id == '0k':
        boundary_marker = " [START: neural-networks] ❌"  # Wrong!
        current_topic = "neural-networks"
    
    # Show mismatch
    mismatch = ""
    if expected_topic and current_topic:
        if expected_topic == "Mars/Space" and current_topic not in ["mars-rover", "space-exploration"]:
            mismatch = f" ⚠️ Should be: {current_topic}"
        elif expected_topic == "Italian Cooking" and current_topic != "pasta-cooking":
            mismatch = f" ⚠️ Should be: pasta-cooking"
        elif expected_topic == "ML/Neural Networks" and current_topic not in ["neural-networks", "machine-learning", "deep-learning"]:
            mismatch = f" ⚠️ Should be: ML topic"
    
    print(f"[{short_id}] {role}: {content}{boundary_marker}{mismatch}")
    if expected_topic:
        print(f"      Expected: {expected_topic}, Current: {current_topic}")
    
    # Show clear topic transitions
    if i < len(nodes) - 1:
        next_node = nodes[i + 1]
        if next_node.get('content'):
            curr_lower = node['content'].lower()
            next_lower = next_node['content'].lower()
            
            # Check for actual topic changes
            if ('mars' in curr_lower or 'space' in curr_lower) and ('pasta' in next_lower or 'italian' in next_lower):
                print("      >>> ACTUAL TOPIC CHANGE: Space → Cooking <<<")
            elif ('pasta' in curr_lower or 'cooking' in curr_lower or 'italian' in curr_lower) and ('neural' in next_lower or 'machine' in next_lower):
                print("      >>> ACTUAL TOPIC CHANGE: Cooking → ML <<<")

print("\n\nSummary of Issues:")
print("1. Topic 'space-exploration' starts at 0a, but that's still about Mars (terraforming)")
print("2. Topic 'pasta-cooking' starts at 0g, but cooking actually starts at 0c")
print("3. Topic 'neural-networks' starts at 0k, but that's still about cooking (marinara sauce)")
print("4. The actual ML topic starts at 0m")
print("\nThis shows a consistent pattern of topic boundaries being set 2-4 messages late!")