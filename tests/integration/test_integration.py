#!/usr/bin/env python3
"""
Integration tests for Episodic components.

Tests the interaction between different parts of the system.
"""

import unittest
import tempfile
import shutil
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.config import config
from episodic.db import initialize_db, insert_node, get_node, database_exists
from episodic.core import Node, ConversationDAG


class TestDatabaseIntegration(unittest.TestCase):
    """Test integration between database and core components."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = config.get("database_path")
        config.set("database_path", os.path.join(self.test_dir, "test_integration.db"))
    
    def tearDown(self):
        """Clean up test environment."""
        if self.original_db_path:
            config.set("database_path", self.original_db_path)
        else:
            config.delete("database_path")
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_database_node_roundtrip(self):
        """Test that nodes can be stored and retrieved from database."""
        # Initialize database (it may return None if nodes already exist)
        initialize_db()
        
        # Create and insert a node
        test_content = "This is a test node"
        node_id, short_id = insert_node(test_content, None, role="user")
        
        # Retrieve the node
        retrieved_node = get_node(node_id)
        
        # Verify the node data
        self.assertIsNotNone(retrieved_node)
        self.assertEqual(retrieved_node["content"], test_content)
        self.assertEqual(retrieved_node["role"], "user")
        self.assertEqual(retrieved_node["id"], node_id)
        self.assertEqual(retrieved_node["short_id"], short_id)
    
    def test_conversation_dag_creation(self):
        """Test creating a conversation DAG with multiple nodes."""
        # Initialize database
        initialize_db()
        
        # Create a conversation thread
        root_id, root_short = insert_node("Hello", None, role="user")
        response_id, response_short = insert_node("Hi there!", root_id, role="assistant")
        followup_id, followup_short = insert_node("How are you?", response_id, role="user")
        
        # Verify the conversation structure
        root_node = get_node(root_id)
        response_node = get_node(response_id)
        followup_node = get_node(followup_id)
        
        self.assertEqual(root_node["parent_id"], None)
        self.assertEqual(response_node["parent_id"], root_id)
        self.assertEqual(followup_node["parent_id"], response_id)
        
        # Test roles
        self.assertEqual(root_node["role"], "user")
        self.assertEqual(response_node["role"], "assistant")
        self.assertEqual(followup_node["role"], "user")


class TestSystemIntegration(unittest.TestCase):
    """Test integration of system components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = config.get("database_path")
        config.set("database_path", os.path.join(self.test_dir, "test_system.db"))
    
    def tearDown(self):
        """Clean up test environment."""
        if self.original_db_path:
            config.set("database_path", self.original_db_path)
        else:
            config.delete("database_path")
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_database_persistence(self):
        """Test that database changes persist across connections."""
        # Initialize and add data
        initialize_db()
        node_id, short_id = insert_node("Persistent test", None)
        
        # Verify data exists
        node = get_node(node_id)
        self.assertIsNotNone(node)
        self.assertEqual(node["content"], "Persistent test")
        
        # Simulate reconnection by checking database exists
        self.assertTrue(database_exists())
        
        # Verify data still exists
        node_again = get_node(node_id)
        self.assertIsNotNone(node_again)
        self.assertEqual(node_again["content"], "Persistent test")


if __name__ == '__main__':
    unittest.main()