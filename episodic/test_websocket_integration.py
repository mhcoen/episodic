"""
Integration tests for WebSocket functionality in the Episodic project.

This module contains tests that verify the WebSocket communication between
the server and clients, ensuring that real-time updates are properly broadcast
and received when changes are made to the conversation DAG.
"""

import unittest
import os
import tempfile
import threading
import time
import json
import socket
import requests
import socketio
from unittest.mock import patch, MagicMock
from episodic.db import initialize_db, insert_node, get_node, set_head, get_head, delete_node, close_connection
from episodic.server import start_server, stop_server

def find_free_port():
    """Find a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class TestWebSocketIntegration(unittest.TestCase):
    """Test the WebSocket communication between server and clients."""

    @classmethod
    def setUpClass(cls):
        """Set up a server and database for all tests in this class."""
        # Create a temporary directory for the database
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Set the database path to a file in the temporary directory
        cls.db_path = os.path.join(cls.temp_dir.name, "test.db")

        # Set the EPISODIC_DB_PATH environment variable to the test database path
        cls.original_db_path = os.environ.get("EPISODIC_DB_PATH")
        os.environ["EPISODIC_DB_PATH"] = cls.db_path

        # Initialize the database with test data
        initialize_db(erase=True, create_root_node=False)

        # Create some test nodes
        cls.root_id, _ = insert_node("Root node")
        cls.child1_id, _ = insert_node("Child 1", cls.root_id)
        cls.child2_id, _ = insert_node("Child 2", cls.root_id)
        cls.grandchild_id, _ = insert_node("Grandchild", cls.child1_id)

        # Set the head to child1
        set_head(cls.child1_id)

        # Find a free port
        cls.port = find_free_port()

        # Start the server on the free port
        cls.server_url = start_server(server_port=cls.port)
        print(f"Server started at {cls.server_url}")

        # Wait for the server to start
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in this class."""
        # Stop the server
        stop_server()

        # Close any open database connections
        close_connection()

        # Restore the original EPISODIC_DB_PATH environment variable
        if cls.original_db_path is not None:
            os.environ["EPISODIC_DB_PATH"] = cls.original_db_path
        else:
            os.environ.pop("EPISODIC_DB_PATH", None)

        # Remove the temporary directory and its contents
        cls.temp_dir.cleanup()

    def tearDown(self):
        """Clean up after each test."""
        # Close any open database connections
        close_connection()

    def test_websocket_connection(self):
        """Test that a client can connect to the WebSocket server."""
        # Create a Socket.IO client
        sio = socketio.Client()

        # Set up event handlers
        connected = threading.Event()

        @sio.event
        def connect():
            print("Connected to server")
            connected.set()

        @sio.event
        def disconnect():
            print("Disconnected from server")

        # Connect to the server
        try:
            sio.connect(self.server_url)

            # Wait for the connection to be established
            self.assertTrue(connected.wait(timeout=5), "Failed to connect to WebSocket server")

            # Disconnect from the server
            sio.disconnect()
        except Exception as e:
            self.fail(f"Failed to connect to WebSocket server: {e}")

    def test_set_current_node_broadcast(self):
        """Test that setting the current node broadcasts a graph update."""
        # Create a Socket.IO client
        sio = socketio.Client()

        # Set up event handlers
        connected = threading.Event()
        graph_update_received = threading.Event()
        graph_data = None

        @sio.event
        def connect():
            print("Connected to server")
            connected.set()

        @sio.on('graph_update')
        def on_graph_update(data):
            nonlocal graph_data
            print(f"Received graph update: {data}")
            graph_data = data
            graph_update_received.set()

        # Connect to the server
        try:
            sio.connect(self.server_url)

            # Wait for the connection to be established
            self.assertTrue(connected.wait(timeout=5), "Failed to connect to WebSocket server")

            # Clear any initial graph update event
            graph_update_received.clear()

            # Make a request to set the current node
            response = requests.post(f"{self.server_url}/set_current_node?id={self.child2_id}")
            self.assertEqual(response.status_code, 200)

            # Wait for the graph update event
            self.assertTrue(graph_update_received.wait(timeout=5), "No graph update received")

            # Verify that the graph data contains the correct current node
            self.assertIsNotNone(graph_data)
            self.assertEqual(graph_data['current_node_id'], self.child2_id)

            # Verify that the node with ID child2_id has is_current=True
            current_nodes = [node for node in graph_data['nodes'] if node['id'] == self.child2_id]
            self.assertEqual(len(current_nodes), 1)
            self.assertTrue(current_nodes[0]['is_current'])

            # Disconnect from the server
            sio.disconnect()
        except Exception as e:
            self.fail(f"Test failed: {e}")

    def test_delete_node_broadcast(self):
        """Test that deleting a node broadcasts a graph update."""
        # Create a Socket.IO client
        sio = socketio.Client()

        # Set up event handlers
        connected = threading.Event()
        graph_update_received = threading.Event()
        graph_data = None

        @sio.event
        def connect():
            print("Connected to server")
            connected.set()

        @sio.on('graph_update')
        def on_graph_update(data):
            nonlocal graph_data
            print(f"Received graph update: {data}")
            graph_data = data
            graph_update_received.set()

        # Connect to the server
        try:
            sio.connect(self.server_url)

            # Wait for the connection to be established
            self.assertTrue(connected.wait(timeout=5), "Failed to connect to WebSocket server")

            # Clear any initial graph update event
            graph_update_received.clear()

            # Make a request to delete a node
            response = requests.post(f"{self.server_url}/delete_node?id={self.grandchild_id}")
            self.assertEqual(response.status_code, 200)

            # Wait for the graph update event
            self.assertTrue(graph_update_received.wait(timeout=5), "No graph update received")

            # Verify that the graph data doesn't contain the deleted node
            self.assertIsNotNone(graph_data)
            deleted_nodes = [node for node in graph_data['nodes'] if node['id'] == self.grandchild_id]
            self.assertEqual(len(deleted_nodes), 0)

            # Verify that the node was deleted from the database
            self.assertIsNone(get_node(self.grandchild_id))

            # Disconnect from the server
            sio.disconnect()
        except Exception as e:
            self.fail(f"Test failed: {e}")

if __name__ == "__main__":
    unittest.main()
