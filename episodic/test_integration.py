"""
Integration tests for the Episodic project.

This module contains tests that verify the integration between different components
of the system, including visualization generation and WebSocket communication.
"""

import unittest
import os
import tempfile
import threading
import time
import json
import socket
import requests
from unittest.mock import patch
from episodic.db import initialize_db, insert_node, get_node, set_head, get_head, close_connection
from episodic.server import start_server, stop_server
from episodic.visualization import visualize_dag

class TestVisualization(unittest.TestCase):
    """Test the visualization generation."""

    def setUp(self):
        """Set up a temporary database and output file for each test."""
        # Create a temporary directory for the database
        self.temp_dir = tempfile.TemporaryDirectory()

        # Set the database path to a file in the temporary directory
        self.db_path = os.path.join(self.temp_dir.name, "test.db")

        # Set the EPISODIC_DB_PATH environment variable to the test database path
        self.original_db_path = os.environ.get("EPISODIC_DB_PATH")
        os.environ["EPISODIC_DB_PATH"] = self.db_path

        # Initialize the database with test data
        initialize_db(erase=True, create_root_node=False)

        # Create some test nodes
        self.root_id, _ = insert_node("Root node")
        self.child1_id, _ = insert_node("Child 1", self.root_id)
        self.child2_id, _ = insert_node("Child 2", self.root_id)
        self.grandchild_id, _ = insert_node("Grandchild", self.child1_id)

        # Set the head to child1
        set_head(self.child1_id)

        # Create a temporary file for the visualization output
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        self.output_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        """Clean up after each test."""
        # Close any open database connections
        close_connection()

        # Restore the original EPISODIC_DB_PATH environment variable
        if self.original_db_path is not None:
            os.environ["EPISODIC_DB_PATH"] = self.original_db_path
        else:
            os.environ.pop("EPISODIC_DB_PATH", None)

        # Remove the temporary directory and its contents
        self.temp_dir.cleanup()

        # Remove the temporary output file
        if os.path.exists(self.output_path):
            os.unlink(self.output_path)

    def test_visualize_dag_static(self):
        """Test generating a static visualization."""
        # Generate a visualization
        output_path = visualize_dag(self.output_path, interactive=False)

        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))

        # Check that the output file is an HTML file
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn('<!DOCTYPE html>', content)
            self.assertIn('<html>', content)

            # Check that the visualization contains the nodes
            self.assertIn('Root node', content)
            self.assertIn('Child 1', content)
            self.assertIn('Child 2', content)
            self.assertIn('Grandchild', content)

    def test_visualize_dag_interactive(self):
        """Test generating an interactive visualization."""
        # Generate an interactive visualization
        output_path = visualize_dag(self.output_path, interactive=True, server_url="http://127.0.0.1:5000")

        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))

        # Check that the output file is an HTML file
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn('<!DOCTYPE html>', content)
            self.assertIn('<html>', content)

            # Check that the visualization contains the nodes
            self.assertIn('Root node', content)
            self.assertIn('Child 1', content)
            self.assertIn('Child 2', content)
            self.assertIn('Grandchild', content)

            # Check that the interactive features are included
            self.assertIn('Double-click', content)
            self.assertIn('Right-click', content)
            self.assertIn('fetch', content)  # AJAX calls
            self.assertIn('socket', content)  # WebSocket

def find_free_port():
    """Find a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class TestWebSocket(unittest.TestCase):
    """Test the WebSocket communication."""

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

    def test_set_current_node_broadcast(self):
        """Test that setting the current node broadcasts a graph update."""
        # This test is a placeholder for a real WebSocket test
        # In a real test, we would:
        # 1. Connect to the WebSocket server
        # 2. Set up a listener for graph_update events
        # 3. Make a request to set the current node
        # 4. Verify that a graph_update event is received with the correct data

        # For now, we'll just verify that the set_current_node endpoint works
        response = requests.post(f"{self.server_url}/set_current_node?id={self.child2_id}")
        self.assertEqual(response.status_code, 200)

        # And that the head was updated in the database
        self.assertEqual(get_head(), self.child2_id)

    def test_delete_node_broadcast(self):
        """Test that deleting a node broadcasts a graph update."""
        # This test is a placeholder for a real WebSocket test
        # In a real test, we would:
        # 1. Connect to the WebSocket server
        # 2. Set up a listener for graph_update events
        # 3. Make a request to delete a node
        # 4. Verify that a graph_update event is received with the correct data

        # For now, we'll just verify that the delete_node endpoint works
        response = requests.post(f"{self.server_url}/delete_node?id={self.child2_id}")
        self.assertEqual(response.status_code, 200)

        # And that the node was deleted from the database
        self.assertIsNone(get_node(self.child2_id))

if __name__ == "__main__":
    unittest.main()
