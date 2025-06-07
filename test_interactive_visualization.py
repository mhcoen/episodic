#!/usr/bin/env python3
"""
Test script for Episodic's interactive visualization functionality.

This script demonstrates how to use the interactive visualization feature
that allows double-clicking on nodes to make them the current node.

Usage:
    python test_interactive_visualization.py [--port PORT] [--no-browser]

Options:
    --port PORT     Port to run the server on (default: 5000)
    --no-browser    Don't open browser automatically
"""

import argparse
import signal
import sys
import time
from episodic.server import visualize_interactive, stop_server, start_server
from episodic.db import database_exists
from episodic.visualization import get_all_nodes

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down the server."""
    print("\nShutting down server...")
    stop_server()
    sys.exit(0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Episodic's interactive visualization")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Print instructions
    print("=" * 80)
    print("EPISODIC INTERACTIVE VISUALIZATION TEST")
    print("=" * 80)
    print("\nThis script demonstrates the interactive visualization functionality.")
    print("\nFeatures:")
    print("  - Double-click on any node to make it the current node (highlighted in orange)")
    print("  - The server handles node operations and updates the visualization")
    print("\nInstructions:")
    print("  1. The visualization will open in your browser (unless --no-browser is specified)")
    print("  2. Double-click on any node to make it the current node")
    print("  3. The node will be highlighted in orange and set as the current node")
    print("  4. Press Ctrl+C in this terminal to stop the server when done")
    print("\nStarting server...")

    # Check if the database exists
    if not database_exists():
        print("Error: No database found. Please initialize the database first with:")
        print("  episodic init")
        sys.exit(1)

    # Check if there are any nodes in the database
    nodes = get_all_nodes()
    if not nodes:
        print("Error: No nodes found in the database. Please add some nodes first with:")
        print("  episodic add \"Your message here\"")
        print("  episodic query \"Your query here\"")
        print("  episodic chat \"Your chat message here\"")
        sys.exit(1)

    try:
        # Start the server with the specified port
        url = start_server(server_port=args.port)

        # Give the server a moment to start
        print(f"Server starting on port {args.port}...")
        time.sleep(1)

        # Open the browser if requested
        if not args.no_browser:
            import webbrowser
            webbrowser.open(url)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)

    print(f"\nInteractive visualization available at: {url}")

    if args.no_browser:
        print("\nSince you used --no-browser, you need to:")
        print("1. Open a web browser manually")
        print(f"2. Navigate to {url}")
        print("3. Interact with the visualization there")

    print("\nPress Enter to stop the server and exit, or type 'help' for more information.")
    print("You can also press Ctrl+C to exit immediately.")

    # Wait for user input instead of an infinite loop
    try:
        while True:
            user_input = input("> ").strip().lower()
            if user_input == "help":
                print("\nAvailable commands:")
                print("  help     - Show this help message")
                print("  open     - Open the visualization URL in a browser")
                print("  url      - Show the visualization URL again")
                print("  exit     - Stop the server and exit")
                print("  <Enter>  - Stop the server and exit")
            elif user_input == "open":
                import webbrowser
                webbrowser.open(url)
                print(f"Opening visualization in browser: {url}")
            elif user_input == "url":
                print(f"\nVisualization URL: {url}")
            elif user_input == "exit" or user_input == "":
                break
            else:
                print(f"Unknown command: {user_input}. Type 'help' for available commands.")

        print("\nShutting down server...")
        stop_server()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
