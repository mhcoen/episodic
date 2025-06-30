#!/usr/bin/env python3
"""Check for duplicate topic names in the database."""

from episodic.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    # Check for duplicate topic names
    c.execute("""
        SELECT name, COUNT(*) as count, GROUP_CONCAT(start_node_id, ', ') as start_nodes
        FROM topics
        GROUP BY name
        HAVING COUNT(*) > 1
    """)
    
    duplicates = c.fetchall()
    
    if duplicates:
        print("Found duplicate topic names:")
        for name, count, start_nodes in duplicates:
            print(f"  '{name}' appears {count} times with start nodes: {start_nodes}")
    else:
        print("No duplicate topic names found.")
    
    # Show all topics
    print("\nAll topics:")
    c.execute("""
        SELECT name, start_node_id, end_node_id
        FROM topics
        ORDER BY ROWID
    """)
    
    for name, start, end in c.fetchall():
        print(f"  {name}: {start} -> {end}")