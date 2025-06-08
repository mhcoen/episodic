"""
Test script for WebSocket functionality in the visualization.

This script tests the WebSocket functionality by simulating changes to the graph
and verifying that WebSocket events are emitted correctly.
"""

import os
import time
import threading
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

def simulate_changes(node_ids):
    """
    Simulate changes to the graph to test WebSocket updates.
    
    This function will:
    1. Wait for the user to open the visualization in two browser windows
    2. Change the current node
    3. Wait for the user to verify that both windows updated
    4. Delete a node
    5. Wait for the user to verify that both windows updated
    """
    # Wait for the user to open the visualization in two browser windows
    input("\nPress Enter after opening the visualization in two browser windows...")
    
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
    
    print("\nWebSocket test completed successfully!")

def test_websocket():
    """Test the WebSocket functionality."""
    # Set up test data
    node_ids = setup_test_data()

    # Start the server on port 5001 to avoid conflicts with AirPlay on macOS
    server_url = start_server(server_port=5001)
    print(f"\nServer started at {server_url}")

    # Open the visualization in a browser
    webbrowser.open(server_url)
    
    # Print instructions
    print("\n=== WEBSOCKET TEST INSTRUCTIONS ===")
    print("1. Open the visualization in two browser windows side by side")
    print("2. The script will change the current node")
    print("3. Verify that both windows update automatically without reloading")
    print("4. The script will delete a node")
    print("5. Verify that both windows update automatically without reloading")
    
    try:
        # Simulate changes to test WebSocket updates
        simulate_changes(node_ids)
        
        # Keep the server running until the user presses Ctrl+C
        print("\nTest completed. Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        stop_server()
        print("Server stopped.")

if __name__ == "__main__":
    test_websocket()