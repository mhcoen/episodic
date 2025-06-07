"""
Test script for the visualization functionality in Episodic.

This script tests the visualization of the conversation DAG, including the highlighting
of the current node.
"""

import os
import sys
from episodic.db import initialize_db, insert_node, get_head, set_head, database_exists, DB_PATH
from episodic.visualization import visualize_dag

def setup():
    """Set up a clean test database with some test nodes."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    # Initialize the database
    initialize_db()
    print("Database initialized.")
    
    # Add some test nodes
    root_id, root_short_id = insert_node("Root node", None)
    print(f"Added root node {root_short_id} (UUID: {root_id})")
    
    child1_id, child1_short_id = insert_node("Child node 1", root_id)
    print(f"Added child node {child1_short_id} (UUID: {child1_id})")
    
    child2_id, child2_short_id = insert_node("Child node 2", root_id)
    print(f"Added child node {child2_short_id} (UUID: {child2_id})")
    
    grandchild_id, grandchild_short_id = insert_node("Grandchild node", child1_id)
    print(f"Added grandchild node {grandchild_short_id} (UUID: {grandchild_id})")
    
    # Set the head to a specific node
    set_head(child1_id)
    print(f"Set head to {child1_short_id} (UUID: {child1_id})")
    
    return root_id, child1_id, child2_id, grandchild_id

def test_visualization():
    """Test the visualization functionality."""
    # Set up the test database
    root_id, child1_id, child2_id, grandchild_id = setup()
    
    # Get the current head
    head_id = get_head()
    print(f"Current head: {head_id}")
    
    # Generate the visualization
    output_path = visualize_dag("test_visualization.html")
    if output_path:
        print(f"Visualization saved to: {output_path}")
        print(f"Open {output_path} in a browser to see the visualization.")
        print("The current node (head) should be highlighted in orange with a thicker border.")
    else:
        print("Failed to generate visualization.")

if __name__ == "__main__":
    test_visualization()