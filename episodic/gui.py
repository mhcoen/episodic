"""
Episodic GUI Module

This module provides a native GUI window for the Episodic project using PyWebView.
It enables displaying the interactive visualization in a native window instead of a web browser.

Usage:
    from episodic.gui import visualize_native

    # Display the visualization in a native window
    visualize_native()
"""

import os
import threading
import time
import webview
from episodic.server import start_server, stop_server
from episodic.visualization import visualize_dag

def visualize_native(width=1000, height=800, server_port=5001):
    """
    Create an interactive visualization and display it in a native window using PyWebView.

    Args:
        width (int): Width of the window (default: 1000)
        height (int): Height of the window (default: 800)
        server_port (int): The port to run the server on (default: 5001)

    Returns:
        The window object
    """
    # Start the server on the specified port
    server_url = start_server(server_port=server_port)
    print(f"Server started at {server_url}")

    # Create a window with the visualization
    window = webview.create_window(
        title="Episodic Conversation Visualization",
        url=server_url,
        width=width,
        height=height,
        resizable=True,
        min_size=(800, 600)
    )

    # Define a function to be called when the window is closed
    def on_closed():
        print("Window closed, stopping server...")
        stop_server()
        print("Server stopped.")

    # Set the on_closed event handler
    window.events.closed += on_closed

    # Start the GUI loop in a separate thread
    webview_thread = threading.Thread(target=webview.start)
    webview_thread.daemon = True
    webview_thread.start()

    print(f"Native visualization window opened.")
    print("The window will remain open until you close it.")
    print("The server will be automatically stopped when the window is closed.")

    return window

def visualize_native_blocking(width=1000, height=800, server_port=5001):
    """
    Create an interactive visualization and display it in a native window using PyWebView.
    This function blocks until the window is closed.

    Args:
        width (int): Width of the window (default: 1000)
        height (int): Height of the window (default: 800)
        server_port (int): The port to run the server on (default: 5001)
    """
    # Start the server on the specified port
    server_url = start_server(server_port=server_port)
    print(f"Server started at {server_url}")

    # Create a window with the visualization
    window = webview.create_window(
        title="Episodic Conversation Visualization",
        url=server_url,
        width=width,
        height=height,
        resizable=True,
        min_size=(800, 600)
    )

    # Start the GUI loop (this will block until the window is closed)
    try:
        webview.start()
    finally:
        # Stop the server when the window is closed
        print("Window closed, stopping server...")
        stop_server()
        print("Server stopped.")

if __name__ == "__main__":
    # If run directly, start the native visualization
    visualize_native_blocking()