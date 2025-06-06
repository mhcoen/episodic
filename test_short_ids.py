"""
Test script for the short ID functionality in Episodic.

This script tests:
1. Creating nodes with short IDs
2. Retrieving nodes by short ID
3. Displaying short IDs in various commands
4. Migrating existing nodes to use short IDs
"""

import os
import sys
import uuid
import sqlite3
from episodic.db import (
    initialize_db, insert_node, get_node, get_ancestry, 
    resolve_node_ref, get_head, set_head, migrate_to_short_ids,
    database_exists, DB_PATH
)

def setup():
    """Set up a clean test database."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    initialize_db()
    print("Database initialized.")

def test_insert_and_get():
    """Test inserting nodes and retrieving them by short ID."""
    # Insert a node
    node_id, short_id = insert_node("Test node 1", None)
    print(f"Added node {short_id} (UUID: {node_id})")
    
    # Get the node by UUID
    node = get_node(node_id)
    print(f"Retrieved by UUID: {node['short_id']} (UUID: {node['id']})")
    
    # Get the node by short ID
    node = get_node(short_id)
    print(f"Retrieved by short ID: {node['short_id']} (UUID: {node['id']})")
    
    # Insert a child node
    child_id, child_short_id = insert_node("Test node 2", node_id)
    print(f"Added child node {child_short_id} (UUID: {child_id})")
    
    # Get the child node
    child = get_node(child_id)
    print(f"Child node: {child['short_id']} (UUID: {child['id']})")
    print(f"Child's parent: {child['parent_id']}")
    
    # Get the ancestry
    ancestry = get_ancestry(child_id)
    print("Ancestry:")
    for ancestor in ancestry:
        print(f"  {ancestor['short_id']} (UUID: {ancestor['id']}): {ancestor['content']}")

def test_resolve_node_ref():
    """Test resolving node references with short IDs."""
    # Insert some nodes
    node1_id, node1_short_id = insert_node("Node 1", None)
    node2_id, node2_short_id = insert_node("Node 2", node1_id)
    node3_id, node3_short_id = insert_node("Node 3", node2_id)
    
    # Set the head to node3
    set_head(node3_id)
    
    # Resolve various references
    head_id = resolve_node_ref("HEAD")
    print(f"HEAD resolves to: {head_id}")
    assert head_id == node3_id
    
    head1_id = resolve_node_ref("HEAD~1")
    print(f"HEAD~1 resolves to: {head1_id}")
    assert head1_id == node2_id
    
    # Resolve by short ID
    resolved_id = resolve_node_ref(node1_short_id)
    print(f"Short ID {node1_short_id} resolves to: {resolved_id}")
    assert resolved_id == node1_id

def test_migration():
    """Test migrating existing nodes to use short IDs."""
    # Create a database with nodes that don't have short IDs
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    # Create the database manually without short IDs
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            parent_id TEXT,
            FOREIGN KEY(parent_id) REFERENCES nodes(id)
        )
    """)
    c.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    # Insert some nodes without short IDs
    for i in range(5):
        node_id = str(uuid.uuid4())
        c.execute("INSERT INTO nodes (id, content, parent_id) VALUES (?, ?, ?)",
                 (node_id, f"Test node {i+1}", None))
    
    conn.commit()
    conn.close()
    
    # Now migrate the nodes to use short IDs
    count = migrate_to_short_ids()
    print(f"Migrated {count} nodes to use short IDs")
    
    # Verify that all nodes now have short IDs
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, short_id, content FROM nodes")
    nodes = c.fetchall()
    conn.close()
    
    print("Nodes after migration:")
    for node in nodes:
        print(f"  {node[1]} (UUID: {node[0]}): {node[2]}")

def main():
    """Run all tests."""
    print("=== Testing Short IDs ===")
    
    print("\n--- Test 1: Insert and Get ---")
    setup()
    test_insert_and_get()
    
    print("\n--- Test 2: Resolve Node References ---")
    setup()
    test_resolve_node_ref()
    
    print("\n--- Test 3: Migration ---")
    test_migration()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()