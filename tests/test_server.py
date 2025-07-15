"""
Unit tests for the server module.

This module contains tests for the Flask routes and WebSocket functionality in server.py.
"""

import unittest
import os
import tempfile
import json
import threading
import time
from unittest.mock import patch
from flask import Flask
from episodic.server import (
    app, start_server, stop_server, index, set_current_node,
    get_current_node, delete_node_endpoint, get_graph_data_endpoint
)
from episodic.db import initialize_db, insert_node, get_node, set_head, get_head

class TestServer(unittest.TestCase):
    """Test the server endpoints."""

    def setUp(self):
        """Set up a test client and database for each test."""
        # Create a temporary directory for the database
        self.temp_dir = tempfile.TemporaryDirectory()

        # Set the database path to a file in the temporary directory
        self.db_path = os.path.join(self.temp_dir.name, "test.db")

        # Set the EPISODIC_DB_PATH environment variable to the test database path
        self.original_db_path = os.environ.get("EPISODIC_DB_PATH")
        os.environ["EPISODIC_DB_PATH"] = self.db_path

        # Initialize the database with test data
        initialize_db(erase=True, create_root_node=False)

        # Create a test client
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Create some test nodes
        self.root_id, self.root_short_id = insert_node("Root node")
        self.child1_id, self.child1_short_id = insert_node("Child 1", self.root_id)
        self.child2_id, self.child2_short_id = insert_node("Child 2", self.root_id)
        self.grandchild_id, self.grandchild_short_id = insert_node("Grandchild", self.child1_id)

        # Set the head to child1
        set_head(self.child1_id)

    def tearDown(self):
        """Clean up after each test."""
        # Close the connection pool to ensure clean state
        from episodic.db_connection import close_pool
        close_pool()

        # Restore the original EPISODIC_DB_PATH environment variable
        if self.original_db_path is not None:
            os.environ["EPISODIC_DB_PATH"] = self.original_db_path
        else:
            os.environ.pop("EPISODIC_DB_PATH", None)

        # Remove the temporary directory and its contents
        self.temp_dir.cleanup()

    def test_index(self):
        """Test the index route."""
        # Make a request to the index route
        response = self.client.get('/')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is an HTML file
        self.assertIn('text/html', response.content_type)

    def test_set_current_node(self):
        """Test the set_current_node route."""
        # Make a request to set the current node
        response = self.client.post(f'/set_current_node?id={self.child2_id}')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains the expected data
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('node', data)
        self.assertEqual(data['node']['id'], self.child2_id)

        # Check that the head was actually updated in the database
        self.assertEqual(get_head(), self.child2_id)

    def test_set_current_node_nonexistent(self):
        """Test setting a nonexistent node as current."""
        # Make a request to set a nonexistent node as current
        response = self.client.post('/set_current_node?id=nonexistent-id')

        # Check that the response is an error
        self.assertEqual(response.status_code, 404)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains an error message
        self.assertIn('error', data)

    def test_set_current_node_no_id(self):
        """Test setting current node without providing an ID."""
        # Make a request without an ID
        response = self.client.post('/set_current_node')

        # Check that the response is an error
        self.assertEqual(response.status_code, 400)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains an error message
        self.assertIn('error', data)

    def test_get_current_node(self):
        """Test the get_current_node route."""
        # Make a request to get the current node
        response = self.client.get('/get_current_node')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains the expected data
        self.assertIn('node', data)
        self.assertEqual(data['node']['id'], self.child1_id)

    def test_delete_node(self):
        """Test the delete_node route."""
        # Make a request to delete a node
        response = self.client.post(f'/delete_node?id={self.child2_id}')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains the expected data
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('deleted_nodes', data)
        self.assertEqual(len(data['deleted_nodes']), 1)
        self.assertIn(self.child2_id, data['deleted_nodes'])

        # Check that the node was actually deleted from the database
        self.assertIsNone(get_node(self.child2_id))

    def test_delete_node_with_descendants(self):
        """Test deleting a node with descendants."""
        # Make a request to delete a node with descendants
        response = self.client.post(f'/delete_node?id={self.child1_id}')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains the expected data
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('deleted_nodes', data)
        self.assertEqual(len(data['deleted_nodes']), 2)
        self.assertIn(self.child1_id, data['deleted_nodes'])
        self.assertIn(self.grandchild_id, data['deleted_nodes'])

        # Check that the nodes were actually deleted from the database
        self.assertIsNone(get_node(self.child1_id))
        self.assertIsNone(get_node(self.grandchild_id))

    def test_delete_nonexistent_node(self):
        """Test deleting a nonexistent node."""
        # Make a request to delete a nonexistent node
        response = self.client.post('/delete_node?id=nonexistent-id')

        # Check that the response is an error
        self.assertEqual(response.status_code, 404)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains an error message
        self.assertIn('error', data)

    def test_delete_node_no_id(self):
        """Test deleting a node without providing an ID."""
        # Make a request without an ID
        response = self.client.post('/delete_node')

        # Check that the response is an error
        self.assertEqual(response.status_code, 400)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains an error message
        self.assertIn('error', data)

    def test_get_graph_data_endpoint(self):
        """Test the get_graph_data_endpoint route."""
        # Make a request to get the graph data
        response = self.client.get('/get_graph_data')

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response is JSON
        self.assertIn('application/json', response.content_type)

        # Parse the response data
        data = json.loads(response.data)

        # Check that the response contains the expected data
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        self.assertIn('current_node_id', data)

        # Check that the correct number of nodes and edges are returned
        self.assertEqual(len(data['nodes']), 4)
        self.assertEqual(len(data['edges']), 3)

        # Check that the current node ID is correct
        self.assertEqual(data['current_node_id'], self.child1_id)

class TestServerLifecycle(unittest.TestCase):
    """Test the server lifecycle functions."""

    @patch('threading.Thread')
    def test_start_server(self, mock_thread):
        """Test starting the server."""
        # Start the server
        url = start_server()

        # Check that the URL is correct
        self.assertEqual(url, "http://127.0.0.1:5000")

        # Check that a thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch('requests.get')
    def test_stop_server(self, mock_get):
        """Test stopping the server."""
        # Set up the server_running flag in the server module
        import episodic.server
        episodic.server.server_running = True

        # Stop the server
        stop_server()

        # Check that a request was made to the shutdown endpoint
        mock_get.assert_called_once_with("http://127.0.0.1:5000/shutdown")

if __name__ == "__main__":
    unittest.main()
