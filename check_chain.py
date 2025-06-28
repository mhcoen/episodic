#!/usr/bin/env python3
"""Check the parent-child chain of nodes."""

from episodic.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    # Get nodes from 09 to 0c
    c.execute("""
        SELECT n1.short_id, n1.role, n2.short_id as parent_short_id, 
               substr(n1.content, 1, 40) as content
        FROM nodes n1
        LEFT JOIN nodes n2 ON n1.parent_id = n2.id
        WHERE n1.short_id IN ('08', '09', '0a', '0b', '0c', '0d')
        ORDER BY n1.ROWID
    """)
    
    print("Node -> Parent (role): content")
    print("-" * 50)
    for short_id, role, parent_short_id, content in c.fetchall():
        print(f"{short_id} -> {parent_short_id or 'None'} ({role}): {content}...")
    
    # Check ancestry from 0c backwards
    print("\nAncestry from 0c:")
    c.execute("SELECT id FROM nodes WHERE short_id = '0c'")
    node_id = c.fetchone()[0]
    
    from episodic.db import get_ancestry
    ancestry = get_ancestry(node_id)
    
    print("Path from root to 0c:")
    for i, node in enumerate(ancestry):
        if node['short_id'] in ['08', '09', '0a', '0b', '0c']:
            print(f"  {node['short_id']} ({node.get('role', 'unknown')})")