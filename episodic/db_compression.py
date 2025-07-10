"""
Compression operations for Episodic database.

This module handles database operations related to conversation compression.
"""

import json
import logging
from typing import List, Dict, Any

from .db_connection import get_connection

# Set up logging
logger = logging.getLogger(__name__)


def store_compression(compressed_node_id: str, original_branch_head: str, 
                      compressed_content: str, original_node_ids: List[str],
                      metadata: Dict[str, Any] = None):
    """
    Store a compression record in the database.
    
    Args:
        compressed_node_id: ID of the new compressed node
        original_branch_head: ID of the last node in the compressed branch
        compressed_content: The compressed/summarized content
        original_node_ids: List of node IDs that were compressed
        metadata: Optional metadata about the compression
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Store in compressions_v2 table
        c.execute("""
            INSERT INTO compressions_v2 
            (compressed_node_id, original_branch_head, compressed_content, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            compressed_node_id,
            original_branch_head,
            compressed_content,
            json.dumps(metadata) if metadata else None
        ))
        
        # Store individual node mappings
        for original_id in original_node_ids:
            c.execute("""
                INSERT INTO compression_nodes 
                (compression_id, original_node_id)
                VALUES (?, ?)
            """, (compressed_node_id, original_id))
            
        conn.commit()


def get_compression_stats():
    """
    Get statistics about compressions in the database.
    
    Returns:
        Dictionary with compression statistics
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        stats = {}
        
        # Check if compressions_v2 table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compressions_v2'")
        if c.fetchone():
            # Get total number of compressions
            c.execute("SELECT COUNT(*) FROM compressions_v2")
            stats['total_compressions'] = c.fetchone()[0]
            
            # Get total nodes compressed
            c.execute("""
                SELECT COUNT(DISTINCT original_node_id) 
                FROM compression_nodes
            """)
            stats['total_nodes_compressed'] = c.fetchone()[0]
            
            # Get average compression ratio (if metadata contains this info)
            c.execute("""
                SELECT AVG(
                    CAST(json_extract(metadata, '$.compression_ratio') AS REAL)
                )
                FROM compressions_v2
                WHERE metadata IS NOT NULL 
                AND json_extract(metadata, '$.compression_ratio') IS NOT NULL
            """)
            result = c.fetchone()
            stats['average_compression_ratio'] = result[0] if result[0] else 0
            
            # Get most recent compression
            c.execute("""
                SELECT compressed_node_id, created_at
                FROM compressions_v2
                ORDER BY created_at DESC
                LIMIT 1
            """)
            recent = c.fetchone()
            if recent:
                stats['most_recent_compression'] = {
                    'node_id': recent[0],
                    'created_at': recent[1]
                }
                
        else:
            # No compressions table
            stats['total_compressions'] = 0
            stats['total_nodes_compressed'] = 0
            stats['average_compression_ratio'] = 0
            
        return stats