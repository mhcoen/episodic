#!/usr/bin/env python3
"""Check what nodes are in the gaps between topics."""

from episodic.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    # Get all nodes in order
    c.execute("""
        SELECT short_id, role, content, id
        FROM nodes
        WHERE short_id IN ('09', '0a', '0b', '0c', '0f', '0g', '0h', '0i', '0j', '0k', '0l', '0m', '0p', '0q', '0r', '0s', '0t', '0u')
        ORDER BY ROWID
    """)
    
    for short_id, role, content, node_id in c.fetchall():
        content_preview = content[:60] + "..." if len(content) > 60 else content
        print(f"{short_id} ({role}): {content_preview}")