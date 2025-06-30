#!/usr/bin/env python3
"""Test topic flow to understand the bug."""

from episodic.db import get_recent_topics, get_connection

print("Current topics in database:\n")

with get_connection() as conn:
    c = conn.cursor()
    
    # Get all topics with their details
    c.execute("""
        SELECT 
            t.ROWID,
            t.name, 
            t.start_node_id,
            t.end_node_id,
            n1.short_id as start_short,
            n2.short_id as end_short,
            t.created_at,
            t.confidence
        FROM topics t
        JOIN nodes n1 ON t.start_node_id = n1.id
        LEFT JOIN nodes n2 ON t.end_node_id = n2.id
        ORDER BY t.ROWID
    """)
    
    for row in c.fetchall():
        rowid, name, start_id, end_id, start_short, end_short, created, conf = row
        print(f"{rowid}. {name:30s} {start_short:4s} → {end_short or 'None':4s} ({conf})")
        print(f"   Start ID: {start_id}")
        print(f"   End ID:   {end_id}")
        print(f"   Created:  {created}")
        print()
        
    # Check for duplicate end nodes
    print("\nChecking for topics with duplicate end nodes:")
    c.execute("""
        SELECT end_node_id, GROUP_CONCAT(name, ', '), COUNT(*)
        FROM topics
        WHERE end_node_id IS NOT NULL
        GROUP BY end_node_id
        HAVING COUNT(*) > 1
    """)
    
    duplicates = c.fetchall()
    if duplicates:
        print("FOUND DUPLICATES:")
        for end_id, names, count in duplicates:
            print(f"  Node {end_id} is end node for {count} topics: {names}")
    else:
        print("  No duplicates found")
        
    # Check topic update history
    print("\n\nChecking topic name history (if logged):")
    # This would require checking logs or adding logging