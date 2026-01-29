#!/usr/bin/env python3
"""
List and explore topics.

Usage:
    ./topics.py              # List all topics
    ./topics.py cooking      # Show exchanges in 'cooking' topic
    ./topics.py 5            # Show exchanges in topic ID 5
"""
import sys
import sqlite3
sys.path.insert(0, '/Users/mhcoen/proj/episodic')

from episodic.db_connection import get_db_path
from episodic.retrieval.segment import get_all_topics, get_cached_segment_nodes
from episodic.retrieval.display import get_exchange_for_display

def list_topics():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    topics = get_all_topics(conn)
    
    print("📚 Topics:\n")
    print(f"{'ID':<4} {'Name':<30} {'Nodes':<8} {'Status'}")
    print("-" * 55)
    
    for t in topics:
        nodes, _ = get_cached_segment_nodes(conn, t['id'])
        status = "ongoing" if t['end_node_id'] is None else "closed"
        print(f"{t['id']:<4} {t['name'][:28]:<30} {len(nodes):<8} {status}")
    
    conn.close()

def show_topic(identifier):
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    
    topics = get_all_topics(conn)
    target = None
    
    # Find by ID or name
    if identifier.isdigit():
        tid = int(identifier)
        target = next((t for t in topics if t['id'] == tid), None)
    else:
        target = next((t for t in topics if identifier.lower() in t['name'].lower()), None)
    
    if not target:
        print(f"⚠️  Topic '{identifier}' not found")
        return
    
    nodes, _ = get_cached_segment_nodes(conn, target['id'])
    
    print(f"📖 Topic: {target['name']}")
    print(f"   ID: {target['id']}, Nodes: {len(nodes)}")
    status = "ongoing" if target['end_node_id'] is None else "closed"
    print(f"   Status: {status}\n")
    print("=" * 70)
    
    # Get user nodes (exchanges) in this topic
    cursor = conn.cursor()
    user_nodes = []
    for nid in nodes:
        cursor.execute("SELECT id, role, created_at FROM nodes WHERE id = ?", (nid,))
        row = cursor.fetchone()
        if row and row['role'] == 'user':
            user_nodes.append(dict(row))
    
    user_nodes.sort(key=lambda x: x['created_at'])
    
    for i, un in enumerate(user_nodes, 1):
        ex = get_exchange_for_display(conn, un['id'], None)
        ts = un['created_at'][:19]
        print(f"\n[{i}] {ts}")
        print(f"    You: {ex['user_content'][:200]}")
        if ex.get('assistant_content'):
            print(f"    AI:  {ex['assistant_content'][:200]}")
        print("-" * 70)
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_topics()
    else:
        show_topic(sys.argv[1])
