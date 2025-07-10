"""
Node operations for Episodic database.

This module handles all node-related database operations including
creation, retrieval, and manipulation of conversation nodes.
"""

import uuid
import sqlite3
import logging
from typing import Optional, List, Dict, Any

from .configuration import MAX_DATABASE_RETRIES
from .db_connection import get_connection
from .db_ids import generate_short_id

# Set up logging
logger = logging.getLogger(__name__)


def insert_node(content, parent_id=None, role=None, provider=None, model=None, max_retries=MAX_DATABASE_RETRIES):
    """
    Insert a new node into the database.
    """
    node_id = str(uuid.uuid4())
    short_id = None
    retries = 0
    
    while retries < max_retries:
        try:
            short_id = generate_short_id()
            
            with get_connection() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO nodes (id, short_id, parent_id, content, role, provider, model) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (node_id, short_id, parent_id, content, role, provider, model))
                
                # Update head to point to this new node
                c.execute("UPDATE state SET head_id = ? WHERE name = 'head'", (node_id,))
                
                return node_id, short_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: nodes.short_id" in str(e):
                retries += 1
                logger.warning(f"Short ID collision for {short_id}, retrying... ({retries}/{max_retries})")
                if retries >= max_retries:
                    # Use fallback ID generation
                    short_id = generate_short_id(fallback=True)
                    logger.warning(f"Max retries reached, using fallback ID: {short_id}")
                continue
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error in insert_node: {e}")
            raise
            
    # This should never be reached due to fallback, but just in case
    raise RuntimeError(f"Failed to insert node after {max_retries} retries")


def get_node(node_id):
    """
    Get a node by its ID or short ID.
    """
    with get_connection() as conn:
        c = conn.cursor()
        # Try both id and short_id
        c.execute("SELECT * FROM nodes WHERE id = ? OR short_id = ?", (node_id, node_id))
        row = c.fetchone()
        if row:
            columns = [description[0] for description in c.description]
            return dict(zip(columns, row))
        return None


def get_ancestry(node_id):
    """
    Get the ancestry chain from root to the given node.
    Returns list of nodes from oldest to newest.
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Get the path from the node to the root
        ancestry = []
        current_id = node_id
        
        while current_id:
            c.execute("SELECT * FROM nodes WHERE id = ? OR short_id = ?", (current_id, current_id))
            row = c.fetchone()
            
            if row:
                columns = [description[0] for description in c.description]
                node = dict(zip(columns, row))
                ancestry.append(node)
                current_id = node.get('parent_id')
            else:
                break
                
        # Return in chronological order (oldest first)
        return list(reversed(ancestry))


def set_head(node_id):
    """Set the head pointer to a specific node."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE state SET head_id = ? WHERE name = 'head'", (node_id,))


def get_head():
    """Get the current head node ID."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT head_id FROM state WHERE name = 'head'")
        result = c.fetchone()
        return result[0] if result else None


def get_recent_nodes(limit=5):
    """
    Get the most recent nodes in the conversation.
    
    Args:
        limit: Maximum number of nodes to return
        
    Returns:
        List of node dictionaries ordered from newest to oldest
    """
    head_id = get_head()
    if not head_id:
        return []
    
    # Get ancestry returns oldest to newest, so we need to reverse and limit
    ancestry = get_ancestry(head_id)
    
    # Return the last 'limit' nodes in reverse order (newest first)
    if len(ancestry) > limit:
        return list(reversed(ancestry[-limit:]))
    else:
        return list(reversed(ancestry))


def get_all_nodes():
    """
    Get all nodes from the database.
    
    Returns:
        List of all node dictionaries ordered by creation time.
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Get all nodes ordered by their position in the DAG
        # We'll use ROWID as a proxy for creation order
        c.execute("""
            SELECT * FROM nodes 
            ORDER BY ROWID ASC
        """)
        
        columns = [description[0] for description in c.description]
        nodes = []
        
        for row in c.fetchall():
            node = dict(zip(columns, row))
            nodes.append(node)
            
        return nodes


def get_descendants(node_id):
    """
    Get all descendants of a node (all nodes that have this node as an ancestor).
    
    Args:
        node_id: The ID of the node to find descendants for
        
    Returns:
        List of descendant node dictionaries
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Find all nodes that have this node in their ancestry
        descendants = []
        
        # Get all nodes
        c.execute("SELECT * FROM nodes")
        columns = [description[0] for description in c.description]
        
        for row in c.fetchall():
            node = dict(zip(columns, row))
            # Check if node_id is in this node's ancestry
            ancestry = get_ancestry(node['id'])
            if any(n['id'] == node_id for n in ancestry[:-1]):  # Exclude the node itself
                descendants.append(node)
                
        return descendants


def get_children(node_id):
    """
    Get the direct children of a node.
    
    Args:
        node_id: The ID of the parent node
        
    Returns:
        List of child node dictionaries
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # Find nodes with this parent_id
        c.execute("SELECT * FROM nodes WHERE parent_id = ?", (node_id,))
        
        columns = [description[0] for description in c.description]
        children = []
        
        for row in c.fetchall():
            child = dict(zip(columns, row))
            children.append(child)
            
        return children


def delete_node(node_id):
    """
    Delete a node and optionally its descendants.
    
    Args:
        node_id: The ID of the node to delete
        
    Returns:
        Number of nodes deleted
    """
    with get_connection() as conn:
        c = conn.cursor()
        
        # First, check if this node has children
        children = get_children(node_id)
        
        if children:
            # If it has children, we need to handle them
            # For now, we'll prevent deletion of nodes with children
            raise ValueError(f"Cannot delete node {node_id} because it has {len(children)} children")
        
        # Delete the node
        c.execute("DELETE FROM nodes WHERE id = ? OR short_id = ?", (node_id, node_id))
        deleted_count = c.rowcount
        
        # If this was the head node, update head to its parent
        if get_head() == node_id:
            node = get_node(node_id)
            if node and node.get('parent_id'):
                set_head(node['parent_id'])
            else:
                # No parent, clear the head
                c.execute("UPDATE state SET head_id = NULL WHERE name = 'head'")
                
        return deleted_count


def resolve_node_ref(ref):
    """
    Resolve a node reference (ID, short ID, or special refs like HEAD/ROOT).
    
    Args:
        ref: Node reference (can be ID, short ID, 'HEAD', 'ROOT', or relative ref like 'HEAD~2')
        
    Returns:
        The resolved node ID, or None if not found
    """
    if not ref:
        return None
        
    ref = ref.upper()
    
    # Handle HEAD reference
    if ref == 'HEAD':
        return get_head()
    
    # Handle ROOT reference  
    if ref == 'ROOT':
        # Get the root node (node with no parent)
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT id FROM nodes WHERE parent_id IS NULL LIMIT 1")
            result = c.fetchone()
            return result[0] if result else None
    
    # Handle relative references like HEAD~2
    if ref.startswith('HEAD~'):
        try:
            steps = int(ref[5:])
            current = get_head()
            
            # Walk up the ancestry chain
            for _ in range(steps):
                if not current:
                    break
                node = get_node(current)
                if node:
                    current = node.get('parent_id')
                else:
                    current = None
                    
            return current
        except ValueError:
            pass
    
    # Try as a direct ID or short ID
    ref_lower = ref.lower()
    node = get_node(ref_lower)
    return node['id'] if node else None