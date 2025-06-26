"""
Database operations for compression system.
Keeps compressions separate from the conversation tree.
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
import uuid
from episodic.db import get_connection


def create_compression_tables():
    """Create tables for compression system if they don't exist."""
    with get_connection() as conn:
        c = conn.cursor()
        
        # Add content column to compressions table if it doesn't exist
        c.execute("""
            SELECT COUNT(*) FROM pragma_table_info('compressions') 
            WHERE name='content'
        """)
        if c.fetchone()[0] == 0:
            c.execute("ALTER TABLE compressions ADD COLUMN content TEXT")
        
        # Create compression_nodes mapping table
        c.execute("""
            CREATE TABLE IF NOT EXISTS compression_nodes (
                compression_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                PRIMARY KEY (compression_id, node_id),
                FOREIGN KEY (compression_id) REFERENCES compressions(compressed_node_id),
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            )
        """)
        
        conn.commit()


def store_compression_v2(
    content: str,
    start_node_id: str,
    end_node_id: str,
    node_ids: List[str],
    original_node_count: int,
    original_words: int,
    compressed_words: int,
    compression_ratio: float,
    strategy: str,
    duration_seconds: Optional[float] = None
) -> str:
    """
    Store a compression without inserting it into the conversation tree.
    
    Returns:
        Compression ID
    """
    compression_id = str(uuid.uuid4())
    
    with get_connection() as conn:
        c = conn.cursor()
        
        # Store compression metadata and content
        c.execute("""
            INSERT INTO compressions (
                compressed_node_id, content, original_branch_head, 
                original_node_count, original_words, compressed_words,
                compression_ratio, strategy, duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            compression_id, content, start_node_id,
            original_node_count, original_words, compressed_words,
            compression_ratio, strategy, duration_seconds
        ))
        
        # Store node mappings
        for node_id in node_ids:
            c.execute("""
                INSERT INTO compression_nodes (compression_id, node_id)
                VALUES (?, ?)
            """, (compression_id, node_id))
        
        conn.commit()
    
    return compression_id


def get_compressions_for_node(node_id: str) -> List[Dict]:
    """Get all compressions that include a specific node."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT c.compressed_node_id, c.content, c.original_words, 
                   c.compressed_words, c.compression_ratio, c.strategy,
                   c.created_at
            FROM compressions c
            JOIN compression_nodes cn ON c.compressed_node_id = cn.compression_id
            WHERE cn.node_id = ?
            ORDER BY c.created_at DESC
        """, (node_id,))
        
        compressions = []
        for row in c.fetchall():
            compressions.append({
                'id': row[0],
                'content': row[1],
                'original_words': row[2],
                'compressed_words': row[3],
                'compression_ratio': row[4],
                'strategy': row[5],
                'created_at': row[6]
            })
        
        return compressions


def get_nodes_in_compression(compression_id: str) -> List[str]:
    """Get all node IDs that are part of a compression."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT node_id 
            FROM compression_nodes 
            WHERE compression_id = ?
            ORDER BY ROWID
        """, (compression_id,))
        
        return [row[0] for row in c.fetchall()]


def get_compression_content(compression_id: str) -> Optional[str]:
    """Get the content of a compression."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT content 
            FROM compressions 
            WHERE compressed_node_id = ?
        """, (compression_id,))
        
        row = c.fetchone()
        return row[0] if row else None