"""
Test script for WebSocket functionality in a real browser.

This script sets up a test environment, starts a server, and provides instructions
for testing the WebSocket functionality in a real browser.
"""

import os
import time
import webbrowser
from episodic.server import start_server, stop_server
from episodic.db import initialize_db, insert_node, get_head, set_head, delete_node

def setup_test_data():
    """Set up test data for the WebSocket test."""
    # Initialize the database with a clean slate
    initialize_db(erase=True)

    # Create a simple conversation tree for testing
    root_id, _ = insert_node("Root node")
    child1_id, _ = insert_node("Child 1", root_id)
    child2_id, _ = insert_node("Child 2", root_id)
    grandchild1_id, _ = insert_node("Grandchild 1", child1_id)
    grandchild2_id, _ = insert_node("Grandchild 2", child1_id)

    print("Test data created:")
    print("- Root node")
    print("  - Child 1")
    print("    - Grandchild 1")
    print("    - Grandchild 2")
    print("  - Child 2")

    # Set the current node to Child 1
    set_head(child1_id)
    
    return {
        "root_id": root_id,
        "child1_id": child1_id,
        "child2_id": child2_id,
        "grandchild1_id": grandchild1_id,
        "grandchild2_id": grandchild2_id
    }

def test_websocket_browser():
    """Test the WebSocket functionality in a real browser."""
    # Set up test data
    node_ids = setup_test_data()

    # Start the server on port 5001 to avoid conflicts with AirPlay on macOS
    server_url = start_server(server_port=5001)
    print(f"\nServer started at {server_url}")

    # Open the visualization in a browser
    webbrowser.open(server_url)
    
    # Print instructions
    print("\n=== WEBSOCKET BROWSER TEST INSTRUCTIONS ===")
    print("1. Open the visualization in two browser windows side by side")
    print("2. In the first window, double-click on a node to make it current")
    print("3. Verify that the second window updates automatically without reloading")
    print("4. In the first window, right-click on a node and delete it")
    print("5. Verify that the second window updates automatically without reloading")
    print("\nThis test will also automatically perform some actions to test WebSocket functionality.")
    print("You can observe these changes in both browser windows.")
    
    try:
        # Wait for the user to open the visualization in two browser windows
        input("\nPress Enter after opening the visualization in two browser windows...")
        
        # Perform some automated tests
        print("\nPerforming automated tests...")
        
        # Change the current node
        print(f"\nChanging current node to 'Child 2' (ID: {node_ids['child2_id']})...")
        set_head(node_ids['child2_id'])
        print("Current node changed. Both windows should update automatically.")
        
        # Wait for the user to verify that both windows updated
        input("\nPress Enter after verifying that both windows updated...")
        
        # Delete a node
        print(f"\nDeleting 'Grandchild 1' (ID: {node_ids['grandchild1_id']})...")
        delete_node(node_ids['grandchild1_id'])
        print("Node deleted. Both windows should update automatically.")
        
        # Wait for the user to verify that both windows updated
        input("\nPress Enter after verifying that both windows updated...")
        
        # Add a new node
        print("\nAdding a new node 'New Child' as a child of 'Root node'...")
        new_child_id, _ = insert_node("New Child", node_ids['root_id'])
        print("New node added. Both windows should update automatically.")
        
        # Wait for the user to verify that both windows updated
        input("\nPress Enter after verifying that both windows updated...")
        
        # Change the current node to the new node
        print(f"\nChanging current node to 'New Child' (ID: {new_child_id})...")
        set_head(new_child_id)
        print("Current node changed. Both windows should update automatically.")
        
        # Wait for the user to verify that both windows updated
        input("\nPress Enter after verifying that both windows updated...")
        
        print("\nWebSocket browser test completed successfully!")
        
        # Keep the server running until the user presses Ctrl+C
        print("\nPress Ctrl+C to stop the server and exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        stop_server()
        print("Server stopped.")

if __name__ == "__main__":
    test_websocket_browser()