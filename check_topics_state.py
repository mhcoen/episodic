#!/usr/bin/env python3
"""Check the state of topics in the database."""

from episodic.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    print("=== Topics in Database ===")
    c.execute("""
        SELECT 
            t.name, 
            t.start_node_id,
            n1.short_id as start_short,
            t.end_node_id,
            n2.short_id as end_short,
            t.created_at
        FROM topics t
        JOIN nodes n1 ON t.start_node_id = n1.id
        LEFT JOIN nodes n2 ON t.end_node_id = n2.id
        ORDER BY t.ROWID
    """)
    
    for row in c.fetchall():
        name, start_id, start_short, end_id, end_short, created = row
        end_display = end_short or "None"
        print(f"{name:30s} {start_short} -> {end_display}")
        if end_id:
            # Count messages in topic
            c2 = conn.cursor()
            c2.execute("""
                WITH RECURSIVE topic_path AS (
                    SELECT id, parent_id, 1 as depth
                    FROM nodes
                    WHERE id = ?
                    
                    UNION ALL
                    
                    SELECT n.id, n.parent_id, tp.depth + 1
                    FROM nodes n
                    JOIN topic_path tp ON n.id = tp.parent_id
                    WHERE tp.depth < 100
                )
                SELECT COUNT(DISTINCT id) FROM topic_path
            """, (end_id,))
            count = c2.fetchone()[0]
            print(f"   Messages from start to end: ~{count}")