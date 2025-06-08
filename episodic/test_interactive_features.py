"""
Test script for interactive features in the visualization.

This script starts a Flask server and opens the visualization in a browser.
It provides instructions for testing the right-click context menu and node color change functionality.
"""

import os
import webbrowser
import time
from episodic.server import start_server, stop_server
from episodic.db import initialize_db, insert_node, get_head, set_head

def setup_test_data():
    """Set up test data for the visualization."""
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
    return child1_id

def test_interactive_features():
    """Test the interactive features of the visualization."""
    # Set up test data
    current_node_id = setup_test_data()

    # Start the server on port 5001 to avoid conflicts with AirPlay on macOS
    server_url = start_server(server_port=5001)
    print(f"\nServer started at {server_url}")

    # Open the visualization in a browser
    webbrowser.open(server_url)

    # Print instructions for testing
    print("\n=== TESTING INSTRUCTIONS ===")
    print("1. Right-click context menu test:")
    print("   - Right-click on any node (except the virtual root)")
    print("   - A context menu should appear with 'Delete node and descendants' option")
    print("   - Hover over the option (it should change color)")
    print("   - Click elsewhere to dismiss the menu")

    print("\n2. Node color change test:")
    print("   - Double-click on any node (except the virtual root)")
    print("   - You should see an alert confirming the node was made current")
    print("   - The node should change color to orange")
    print("   - The previously current node should return to blue")

    print("\n3. WebSocket real-time update test:")
    print("   - Open the visualization in two browser windows side by side")
    print("   - In the first window, double-click on a node to make it current")
    print("   - The second window should update automatically without reloading")
    print("   - In the first window, right-click on a node and delete it")
    print("   - The second window should update automatically without reloading")

    print("\nPress Ctrl+C when done testing to stop the server.")

    try:
        # Keep the server running until the user presses Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        stop_server()
        print("Server stopped.")

if __name__ == "__main__":
    test_interactive_features()
