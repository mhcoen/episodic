"""
Unit tests for the database module.

This module contains tests for the database operations in db.py.
"""

import unittest
import os
import tempfile
import sqlite3
from episodic.db import (
    get_connection, database_exists, initialize_db, insert_node, get_node,
    get_ancestry, set_head, get_head, get_descendants, delete_node,
    resolve_node_ref, get_recent_nodes, get_all_nodes
)

class TestDatabase(unittest.TestCase):
    """Test the database operations."""

    def setUp(self):
        """Set up a temporary database for each test."""
        # Create a temporary directory for the database
        self.temp_dir = tempfile.TemporaryDirectory()

        # Set the database path to a file in the temporary directory
        self.db_path = os.path.join(self.temp_dir.name, "test.db")

        # Set the EPISODIC_DB_PATH environment variable to the test database path
        self.original_db_path = os.environ.get("EPISODIC_DB_PATH")
        os.environ["EPISODIC_DB_PATH"] = self.db_path

        # Initialize the database
        initialize_db(erase=True, create_root_node=False)

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

    def test_database_exists(self):
        """Test that the database_exists function works correctly."""
        # The database should exist after initialization
        self.assertTrue(database_exists())

        # Remove the database file if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # The database should no longer exist
        self.assertFalse(database_exists())

    def test_initialize_db(self):
        """Test that the database can be initialized."""
        # The database should exist after initialization
        self.assertTrue(database_exists())

        # Check that the nodes table exists
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
            self.assertIsNotNone(c.fetchone())

    def test_insert_node(self):
        """Test that a node can be inserted into the database."""
        # Insert a node
        content = "Test content"
        node_id, short_id = insert_node(content)

        # Check that the node was inserted
        self.assertIsNotNone(node_id)
        self.assertIsNotNone(short_id)

        # Check that the node can be retrieved
        node = get_node(node_id)
        self.assertIsNotNone(node)
        self.assertEqual(node["content"], content)
        self.assertEqual(node["id"], node_id)
        self.assertEqual(node["short_id"], short_id)
        self.assertIsNone(node["parent_id"])

    def test_insert_node_with_parent(self):
        """Test that a node can be inserted with a parent."""
        # Insert a parent node
        parent_content = "Parent content"
        parent_id, parent_short_id = insert_node(parent_content)

        # Insert a child node
        child_content = "Child content"
        child_id, child_short_id = insert_node(child_content, parent_id)

        # Check that the child node has the correct parent
        child = get_node(child_id)
        self.assertEqual(child["parent_id"], parent_id)

    def test_get_node(self):
        """Test that a node can be retrieved from the database."""
        # Insert a node
        content = "Test content"
        node_id, short_id = insert_node(content)

        # Get the node
        node = get_node(node_id)

        # Check that the node has the expected properties
        self.assertEqual(node["id"], node_id)
        self.assertEqual(node["short_id"], short_id)
        self.assertEqual(node["content"], content)
        self.assertIsNone(node["parent_id"])

    def test_get_nonexistent_node(self):
        """Test that getting a nonexistent node returns None."""
        # Try to get a nonexistent node
        node = get_node("nonexistent-id")

        # Check that None was returned
        self.assertIsNone(node)

    def test_get_ancestry(self):
        """Test that the ancestry of a node can be retrieved."""
        # Create a chain of nodes
        root_id, _ = insert_node("Root")
        child_id, _ = insert_node("Child", root_id)
        grandchild_id, _ = insert_node("Grandchild", child_id)

        # Get the ancestry of the grandchild
        ancestry = get_ancestry(grandchild_id)

        # Check that the ancestry is correct
        self.assertEqual(len(ancestry), 3)
        self.assertEqual(ancestry[0]["id"], root_id)
        self.assertEqual(ancestry[1]["id"], child_id)
        self.assertEqual(ancestry[2]["id"], grandchild_id)

    def test_set_and_get_head(self):
        """Test that the head node can be set and retrieved."""
        # Insert a node
        node_id, _ = insert_node("Head node")

        # Set the node as the head
        set_head(node_id)

        # Get the head
        head_id = get_head()

        # Check that the head is the node we set
        self.assertEqual(head_id, node_id)

    def test_get_head_when_not_set(self):
        """Test that get_head returns None when no head is set."""
        # Get the head without setting it
        head_id = get_head()

        # Check that None was returned
        self.assertIsNone(head_id)

    def test_get_descendants(self):
        """Test that the descendants of a node can be retrieved."""
        # Create a tree of nodes
        root_id, _ = insert_node("Root")
        child1_id, _ = insert_node("Child 1", root_id)
        child2_id, _ = insert_node("Child 2", root_id)
        grandchild1_id, _ = insert_node("Grandchild 1", child1_id)
        grandchild2_id, _ = insert_node("Grandchild 2", child1_id)

        # Get the descendants of root
        descendants = get_descendants(root_id)

        # Check that the correct descendants were returned
        self.assertEqual(len(descendants), 4)
        descendant_ids = [d['id'] for d in descendants]
        self.assertIn(child1_id, descendant_ids)
        self.assertIn(child2_id, descendant_ids)
        self.assertIn(grandchild1_id, descendant_ids)
        self.assertIn(grandchild2_id, descendant_ids)

        # Get the descendants of child1
        descendants = get_descendants(child1_id)

        # Check that the correct descendants were returned
        self.assertEqual(len(descendants), 2)
        descendant_ids = [d['id'] for d in descendants]
        self.assertIn(grandchild1_id, descendant_ids)
        self.assertIn(grandchild2_id, descendant_ids)

    def test_delete_node(self):
        """Test that a node can be deleted (but not if it has children)."""
        # Create a tree of nodes
        root_id, _ = insert_node("Root")
        child1_id, _ = insert_node("Child 1", root_id)
        child2_id, _ = insert_node("Child 2", root_id)
        grandchild_id, _ = insert_node("Grandchild", child1_id)

        # Try to delete child1 which has children - should fail
        with self.assertRaises(ValueError) as cm:
            delete_node(child1_id)
        self.assertIn("has 1 children", str(cm.exception))

        # Delete grandchild (no children) - should succeed
        deleted_count = delete_node(grandchild_id)
        self.assertEqual(deleted_count, 1)

        # Now child1 has no children, so we can delete it
        deleted_count = delete_node(child1_id)
        self.assertEqual(deleted_count, 1)

        # Check that the deleted nodes are no longer in the database
        self.assertIsNone(get_node(child1_id))
        self.assertIsNone(get_node(grandchild_id))

        # Check that the other nodes are still in the database
        self.assertIsNotNone(get_node(root_id))
        self.assertIsNotNone(get_node(child2_id))

    def test_delete_head_node(self):
        """Test that deleting the head node updates the head reference."""
        # Create a tree of nodes
        root_id, _ = insert_node("Root")
        child_id, _ = insert_node("Child", root_id)

        # Set the child as the head
        set_head(child_id)

        # Delete the child
        delete_node(child_id)

        # Check that the head is now the root
        self.assertEqual(get_head(), root_id)

    def test_delete_nonexistent_node(self):
        """Test that deleting a nonexistent node returns 0."""
        # Try to delete a nonexistent node
        deleted_count = delete_node("nonexistent-id")

        # Check that 0 was returned
        self.assertEqual(deleted_count, 0)

    def test_resolve_node_ref(self):
        """Test that a node reference can be resolved to its UUID."""
        # Insert a node
        node_id, short_id = insert_node("Test node")

        # Resolve the node ID
        resolved_id = resolve_node_ref(node_id)

        # Check that the resolved ID is the same as the original
        self.assertEqual(resolved_id, node_id)

        # Resolve the short ID
        resolved_id = resolve_node_ref(short_id)

        # Check that the resolved ID is the node ID
        self.assertEqual(resolved_id, node_id)

    def test_resolve_nonexistent_node_ref(self):
        """Test that resolving a nonexistent node reference returns None."""
        # Try to resolve a nonexistent reference
        ref = "nonexistent-ref"
        resolved_id = resolve_node_ref(ref)

        # Check that None was returned for non-existent node
        self.assertIsNone(resolved_id)

    def test_get_recent_nodes(self):
        """Test that recent nodes can be retrieved."""
        # Create a chain of nodes
        node_ids = []
        parent_id = None
        for i in range(10):
            node_id, _ = insert_node(f"Node {i}", parent_id)
            node_ids.append(node_id)
            parent_id = node_id  # Next node will be child of this one
        
        # Set the last node as head
        set_head(node_ids[-1])

        # Get the 5 most recent nodes
        recent_nodes = get_recent_nodes(5)

        # Check that the correct number of nodes was returned
        self.assertEqual(len(recent_nodes), 5)
        
        # Check that they are the most recent ones (in reverse order)
        recent_node_ids = [n['id'] for n in recent_nodes]
        expected_ids = list(reversed(node_ids[-5:]))
        self.assertEqual(recent_node_ids, expected_ids)

        # Check that the nodes are in reverse order of insertion
        for i, node in enumerate(recent_nodes):
            self.assertEqual(node["id"], node_ids[9 - i])

    def test_get_all_nodes(self):
        """Test that all nodes can be retrieved."""
        # Insert some nodes
        node_ids = []
        for i in range(5):
            node_id, _ = insert_node(f"Node {i}")
            node_ids.append(node_id)

        # Get all nodes
        all_nodes = get_all_nodes()

        # Check that the correct number of nodes was returned
        self.assertEqual(len(all_nodes), 5)

        # Check that all inserted nodes are in the result
        for node_id in node_ids:
            self.assertTrue(any(node["id"] == node_id for node in all_nodes))

if __name__ == "__main__":
    unittest.main()
