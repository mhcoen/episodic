"""
Episodic Server Module

This module provides a web server for the Episodic project using Flask.
It enables interactive features in the visualization by providing endpoints
for node operations like setting the current node.

Usage:
    from episodic.server import start_server, stop_server

    # Start the server in the background
    start_server()

    # Stop the server when done
    stop_server()
"""

import os
import threading
import webbrowser
from flask import Flask, request, jsonify, send_file, send_from_directory

from episodic.db import get_node, set_head, get_head, delete_node, get_all_nodes
from episodic.visualization import visualize_dag
from episodic.configuration import (
    SERVER_SHUTDOWN_DELAY, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
)

# Create Flask application
app = Flask(__name__)


# Global variables
server_thread = None
server_running = False
port = DEFAULT_SERVER_PORT
host = DEFAULT_SERVER_HOST

# Function to get graph data for WebSocket updates
def get_graph_data():
    """Get the current graph data for WebSocket updates."""
    nodes = get_all_nodes()
    current_node_id = get_head()

    # Format the data for the client
    node_data = []
    edge_data = []

    for node in nodes:
        # Add node data
        node_data.append({
            'id': node['id'],
            'short_id': node['short_id'],
            'content': node['content'],
            'is_current': (node['id'] == current_node_id)
        })

        # Add edge data if the node has a parent
        if node['parent_id']:
            edge_data.append({
                'from': node['parent_id'],
                'to': node['id']
            })

    return {
        'nodes': node_data,
        'edges': edge_data,
        'current_node_id': current_node_id
    }

def broadcast_graph_update():
    """Placeholder for broadcasting graph updates to connected clients.
    
    TODO: Implement WebSocket or Server-Sent Events for real-time updates.
    Currently this is a no-op to prevent runtime errors.
    """

@app.route('/')
def index():
    """Serve the visualization."""
    try:
        # Generate a new visualization with interactive features
        # Use the actual server URL with the current port
        server_url = f"http://{host}:{port}"
        print(f"=== SERVER ROOT ROUTE ACCESSED ===")
        print(f"Generating visualization with server URL: {server_url}")
        output_path = visualize_dag(interactive=True, server_url=server_url)
        print(f"Visualization generated: {output_path}")
        print(f"Sending file to browser...")
        return send_file(output_path)
    except Exception as e:
        print(f"ERROR in root route: {e}")
        import traceback
        traceback.print_exc()
        return f"Server Error: {e}", 500

@app.route('/set_current_node', methods=['POST'])
def set_current_node():
    """Set the current node."""
    try:
        node_id = request.args.get('id')
        if not node_id:
            return jsonify({"error": "No node ID provided"}), 400

        # Verify that the node exists
        node = get_node(node_id)
        if not node:
            return jsonify({"error": f"Node not found: {node_id}"}), 404

        # Set the node as the current head
        set_head(node_id)

        # Broadcast the update to all connected clients
        broadcast_graph_update()

        return jsonify({
            "success": True,
            "message": f"Current node set to: {node['short_id']} (UUID: {node['id']})",
            "node": {
                "id": node['id'],
                "short_id": node['short_id'],
                "content": node['content']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_current_node')
def get_current_node():
    """Get the current node."""
    try:
        head_id = get_head()
        if not head_id:
            return jsonify({"error": "No current node set"}), 404

        node = get_node(head_id)
        if not node:
            return jsonify({"error": f"Current node not found: {head_id}"}), 404

        return jsonify({
            "node": {
                "id": node['id'],
                "short_id": node['short_id'],
                "content": node['content']
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_node', methods=['POST'])
def delete_node_endpoint():
    """Delete a node and all its descendants."""
    try:
        node_id = request.args.get('id')
        if not node_id:
            return jsonify({"error": "No node ID provided"}), 400

        # Verify that the node exists
        node = get_node(node_id)
        if not node:
            return jsonify({"error": f"Node not found: {node_id}"}), 404

        # Delete the node and its descendants
        deleted_nodes = delete_node(node_id)

        # Broadcast the update to all connected clients
        broadcast_graph_update()

        return jsonify({
            "success": True,
            "message": f"Deleted node {node['short_id']} and {len(deleted_nodes) - 1} descendants",
            "deleted_nodes": deleted_nodes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/lib/<path:filename>')
def serve_static(filename):
    """Serve static files from the lib directory."""
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(os.path.join(root_dir, 'lib'), filename)

@app.route('/get_graph_data')
def get_graph_data_endpoint():
    """Get the current graph data for HTTP polling fallback."""
    try:
        # Get the graph data using the same function used for WebSocket updates
        graph_data = get_graph_data()

        # Log the request
        print(f"HTTP polling request for graph data received. Returning data with {len(graph_data['nodes'])} nodes.")

        # Return the data as JSON
        return jsonify(graph_data)
    except Exception as e:
        print(f"Error handling get_graph_data request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Handle favicon.ico requests."""
    # Return a 204 No Content response
    return '', 204

def run_server():
    """Run the Flask server."""
    global server_running
    server_running = True
    app.run(host=host, port=port, debug=False, use_reloader=False)
    server_running = False

def start_server(server_port=5000):
    """
    Start the server in a background thread.

    Args:
        server_port (int): The port to run the server on

    Returns:
        The URL of the server
    """
    global server_thread, port
    port = server_port

    if server_thread and server_thread.is_alive():
        return f"http://{host}:{port}"

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    return f"http://{host}:{port}"

def stop_server():
    """Stop the server."""
    global server_running, server_thread
    if server_running:
        import requests
        try:
            requests.get(f"http://{host}:{port}/shutdown")
            print("Server shutdown request sent successfully.")
        except Exception as e:
            print(f"Error sending shutdown request: {e}")
            print("Server may still be running in the background.")

    # Reset the server thread
    server_thread = None

def visualize_interactive(open_browser=True):
    """
    Create an interactive visualization and serve it via the web server.

    Args:
        open_browser (bool): Whether to open the visualization in a browser

    Returns:
        The URL of the visualization
    """
    url = start_server()

    if open_browser:
        webbrowser.open(url)
        print(f"Opening interactive visualization in your browser at: {url}")
    else:
        print(f"Interactive visualization available at: {url}")
        print("You need to manually open this URL in your web browser.")

    print("The server is running in the background.")
    print("When you're done, call stop_server() to shut down the server.")

    return url

# Add a shutdown route for clean shutdown
@app.route('/shutdown')
def shutdown():
    """Shutdown the server."""
    # In newer versions of Werkzeug (2.0+), the 'werkzeug.server.shutdown' function
    # has been removed. Instead, we'll use a more modern approach.
    import os
    import threading
    import time

    def shutdown_server():
        # Give a small delay to allow the response to be sent
        time.sleep(SERVER_SHUTDOWN_DELAY)
        os._exit(0)

    # Start a thread to shut down the server
    threading.Thread(target=shutdown_server).start()

    return 'Server shutting down...'

if __name__ == '__main__':
    # If run directly, start the server
    visualize_interactive()
