"""
Test script for native visualization functionality.

This script tests the native visualization functionality by creating a simple
conversation tree and displaying it in a native window using PyWebView.
"""

import time
from episodic.db import initialize_db, insert_node, get_head, set_head
from episodic.gui import visualize_native

def setup_test_data():
    """Set up test data for the native visualization test."""
    # Initialize the database with a clean slate
    initialize_db(erase=True)

    # Create a simple conversation tree for testing
    root_id, _ = insert_node("Root node")
    child1_id, _ = insert_node("Child 1", root_id)
    child2_id, _ = insert_node("Child 2", root_id)
    grandchild1_id, _ = insert_node("Grandchild 1", child1_id)

    print("Test data created:")
    print("- Root node")
    print("  - Child 1")
    print("    - Grandchild 1")
    print("  - Child 2")

    # Set the current node to Child 1
    set_head(child1_id)
    
    return {
        "root_id": root_id,
        "child1_id": child1_id,
        "child2_id": child2_id,
        "grandchild1_id": grandchild1_id
    }

def test_native_visualization():
    """Test the native visualization functionality."""
    # Set up test data
    node_ids = setup_test_data()

    print("\n=== NATIVE VISUALIZATION TEST INSTRUCTIONS ===")
    print("1. A native window will open with the visualization")
    print("2. Interact with the visualization (double-click, right-click, etc.)")
    print("3. Close the window when done")
    print("4. The server will be automatically stopped when the window is closed")
    
    # Open the native visualization window
    window = visualize_native(width=1200, height=900, server_port=5001)
    
    # Keep the script running until the user presses Ctrl+C
    try:
        print("\nPress Ctrl+C to stop the test...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest stopped by user.")

if __name__ == "__main__":
    test_native_visualization()