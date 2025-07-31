#!/usr/bin/env python3
"""
Retroactively create topics for existing conversations based on drift scores.
This script analyzes the conversation history and creates topic boundaries
where significant drift is detected.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json

# Configuration
DRIFT_THRESHOLD = 0.6  # Topic boundary threshold
MIN_TOPIC_LENGTH = 2   # Minimum messages per topic

def get_db_connection():
    """Get database connection."""
    db_path = Path.home() / '.episodic' / 'episodic.db'
    return sqlite3.connect(str(db_path))

def analyze_conversation_drift():
    """Analyze the conversation and identify topic boundaries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all nodes in order
    cursor.execute("""
        SELECT n.id, n.short_id, n.role, n.content, n.created_at,
               tds.drift_score
        FROM nodes n
        LEFT JOIN topic_detection_scores tds ON n.short_id = tds.user_node_short_id
        ORDER BY n.created_at
    """)
    
    nodes = []
    for row in cursor.fetchall():
        nodes.append({
            'id': row[0],
            'short_id': row[1],
            'role': row[2],
            'content': row[3][:100] + '...' if len(row[3]) > 100 else row[3],
            'created_at': row[4],
            'drift_score': row[5] or 0.0
        })
    
    # Identify topic boundaries
    topic_boundaries = []
    current_topic_start = 0
    
    for i, node in enumerate(nodes):
        if node['drift_score'] > DRIFT_THRESHOLD and i > current_topic_start + MIN_TOPIC_LENGTH:
            # Found a topic boundary
            topic_boundaries.append({
                'start_idx': current_topic_start,
                'end_idx': i - 1,
                'start_node': nodes[current_topic_start],
                'end_node': nodes[i - 1],
                'trigger_node': node,
                'drift_score': node['drift_score']
            })
            current_topic_start = i
    
    # Don't forget the last topic
    if current_topic_start < len(nodes) - 1:
        topic_boundaries.append({
            'start_idx': current_topic_start,
            'end_idx': len(nodes) - 1,
            'start_node': nodes[current_topic_start],
            'end_node': nodes[-1],
            'trigger_node': None,
            'drift_score': 0.0
        })
    
    conn.close()
    return nodes, topic_boundaries

def suggest_topic_names(nodes, boundaries):
    """Suggest names for each topic based on content."""
    suggestions = []
    
    for i, boundary in enumerate(boundaries):
        # Get key content from the topic
        topic_nodes = nodes[boundary['start_idx']:boundary['end_idx'] + 1]
        user_messages = [n for n in topic_nodes if n['role'] == 'user']
        
        # Simple heuristic: use keywords from first user message
        if user_messages:
            first_msg = user_messages[0]['content'].lower()
            
            # Extract potential topic name
            if 'python' in first_msg and 'csv' in first_msg:
                name = "Python CSV Processing"
            elif 'pandas' in first_msg:
                name = "Pandas Data Analysis"
            elif 'carbonara' in first_msg or 'pasta' in first_msg:
                name = "Italian Cooking"
            elif 'machine learning' in first_msg or 'supervised' in first_msg:
                name = "Machine Learning Basics"
            elif 'japan' in first_msg or 'tokyo' in first_msg:
                name = "Japan Travel Planning"
            elif 'react' in first_msg or 'vue' in first_msg:
                name = "Web Development"
            elif 'workout' in first_msg or 'exercise' in first_msg:
                name = "Fitness and Exercise"
            elif 'garden' in first_msg or 'vegetable' in first_msg:
                name = "Home Gardening"
            elif 'budget' in first_msg or 'finance' in first_msg:
                name = "Personal Finance"
            else:
                # Generic name based on position
                name = f"Topic {i + 1}"
        else:
            name = f"Topic {i + 1}"
        
        suggestions.append({
            'boundary': boundary,
            'suggested_name': name,
            'message_count': boundary['end_idx'] - boundary['start_idx'] + 1
        })
    
    return suggestions

def generate_sql_commands(suggestions):
    """Generate SQL commands to create the topics."""
    sql_commands = []
    
    # First, check if topics table is empty
    sql_commands.append("-- Check current topics")
    sql_commands.append("SELECT COUNT(*) as topic_count FROM topics;")
    sql_commands.append("")
    
    sql_commands.append("-- Create topics based on drift analysis")
    
    for i, suggestion in enumerate(suggestions):
        boundary = suggestion['boundary']
        
        # Generate unique topic ID
        topic_id = f"retro_topic_{i + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        sql = f"""
INSERT INTO topics (id, name, start_node_id, end_node_id, status, created_at)
VALUES (
    '{topic_id}',
    '{suggestion['suggested_name']}',
    '{boundary['start_node']['id']}',
    '{boundary['end_node']['id']}',
    'completed',
    CURRENT_TIMESTAMP
);"""
        sql_commands.append(sql)
    
    return sql_commands

def main():
    """Main execution."""
    print("Analyzing conversation for topic boundaries...")
    print(f"Using drift threshold: {DRIFT_THRESHOLD}")
    print()
    
    # Analyze conversation
    nodes, boundaries = analyze_conversation_drift()
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Topic boundaries found: {len(boundaries)}")
    print()
    
    # Get topic suggestions
    suggestions = suggest_topic_names(nodes, boundaries)
    
    # Display analysis
    print("Suggested Topics:")
    print("=" * 80)
    
    for i, suggestion in enumerate(suggestions):
        boundary = suggestion['boundary']
        print(f"\nTopic {i + 1}: {suggestion['suggested_name']}")
        print(f"  Messages: {suggestion['message_count']}")
        print(f"  Range: {boundary['start_node']['short_id']} → {boundary['end_node']['short_id']}")
        
        if boundary['trigger_node']:
            print(f"  Triggered by: {boundary['trigger_node']['short_id']} (drift: {boundary['drift_score']:.3f})")
        
        print(f"  Start: {boundary['start_node']['content']}")
        print(f"  End: {boundary['end_node']['content']}")
    
    # Generate SQL
    print("\n" + "=" * 80)
    print("SQL Commands to Create Topics:")
    print("=" * 80)
    
    sql_commands = generate_sql_commands(suggestions)
    for cmd in sql_commands:
        print(cmd)
    
    # Save to file
    output_file = "retroactive_topics.sql"
    with open(output_file, 'w') as f:
        f.write("-- Retroactive topic creation based on drift analysis\n")
        f.write(f"-- Generated: {datetime.now()}\n")
        f.write(f"-- Drift threshold: {DRIFT_THRESHOLD}\n\n")
        f.write("\n".join(sql_commands))
    
    print(f"\nSQL commands saved to: {output_file}")
    print("\nTo apply these topics, run:")
    print(f"  sqlite3 ~/.episodic/episodic.db < {output_file}")

if __name__ == "__main__":
    main()