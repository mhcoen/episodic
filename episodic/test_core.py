"""
Unit tests for the core module.

This module contains tests for the ConversationDAG class and its methods.
"""

import unittest
from episodic.core import ConversationDAG, Node

class TestNode(unittest.TestCase):
    """Test the Node class."""

    def test_node_creation(self):
        """Test that a node can be created with the correct properties."""
        # Create a node
        message = "Test message"
        node = Node(message)
        
        # Check that the node has the expected properties
        self.assertEqual(node.message, message)
        self.assertIsNone(node.parent_id)
        self.assertIsNotNone(node.id)
        self.assertIsNotNone(node.timestamp)
    
    def test_node_with_parent(self):
        """Test that a node can be created with a parent ID."""
        # Create a node with a parent ID
        message = "Child message"
        parent_id = "parent-id"
        node = Node(message, parent_id)
        
        # Check that the node has the expected properties
        self.assertEqual(node.message, message)
        self.assertEqual(node.parent_id, parent_id)
    
    def test_to_dict(self):
        """Test that a node can be converted to a dictionary."""
        # Create a node
        message = "Test message"
        node = Node(message)
        
        # Convert to dictionary
        node_dict = node.to_dict()
        
        # Check that the dictionary has the expected keys and values
        self.assertEqual(node_dict["id"], node.id)
        self.assertEqual(node_dict["message"], message)
        self.assertEqual(node_dict["timestamp"], node.timestamp)
        self.assertIsNone(node_dict["parent_id"])

class TestConversationDAG(unittest.TestCase):
    """Test the ConversationDAG class."""

    def setUp(self):
        """Set up a new ConversationDAG for each test."""
        self.dag = ConversationDAG()
    
    def test_add_node(self):
        """Test that a node can be added to the DAG."""
        # Add a node
        message = "Test message"
        node = self.dag.add_node(message)
        
        # Check that the node was added to the DAG
        self.assertIn(node.id, self.dag.nodes)
        self.assertEqual(self.dag.nodes[node.id], node)
    
    def test_add_node_with_parent(self):
        """Test that a node can be added with a parent."""
        # Add a parent node
        parent_message = "Parent message"
        parent = self.dag.add_node(parent_message)
        
        # Add a child node
        child_message = "Child message"
        child = self.dag.add_node(child_message, parent.id)
        
        # Check that the child node has the correct parent
        self.assertEqual(child.parent_id, parent.id)
    
    def test_get_node(self):
        """Test that a node can be retrieved from the DAG."""
        # Add a node
        message = "Test message"
        node = self.dag.add_node(message)
        
        # Get the node
        retrieved_node = self.dag.get_node(node.id)
        
        # Check that the retrieved node is the same as the original
        self.assertEqual(retrieved_node, node)
    
    def test_get_nonexistent_node(self):
        """Test that getting a nonexistent node returns None."""
        # Try to get a nonexistent node
        node = self.dag.get_node("nonexistent-id")
        
        # Check that None was returned
        self.assertIsNone(node)
    
    def test_get_ancestry(self):
        """Test that the ancestry of a node can be retrieved."""
        # Create a chain of nodes
        root = self.dag.add_node("Root")
        child = self.dag.add_node("Child", root.id)
        grandchild = self.dag.add_node("Grandchild", child.id)
        
        # Get the ancestry of the grandchild
        ancestry = self.dag.get_ancestry(grandchild.id)
        
        # Check that the ancestry is correct
        self.assertEqual(len(ancestry), 3)
        self.assertEqual(ancestry[0], root)
        self.assertEqual(ancestry[1], child)
        self.assertEqual(ancestry[2], grandchild)
    
    def test_delete_node(self):
        """Test that a node and its descendants can be deleted."""
        # Create a tree of nodes
        root = self.dag.add_node("Root")
        child1 = self.dag.add_node("Child 1", root.id)
        child2 = self.dag.add_node("Child 2", root.id)
        grandchild = self.dag.add_node("Grandchild", child1.id)
        
        # Delete child1 and its descendants
        deleted_nodes = self.dag.delete_node(child1.id)
        
        # Check that the correct nodes were deleted
        self.assertEqual(len(deleted_nodes), 2)
        self.assertIn(child1.id, deleted_nodes)
        self.assertIn(grandchild.id, deleted_nodes)
        
        # Check that the deleted nodes are no longer in the DAG
        self.assertNotIn(child1.id, self.dag.nodes)
        self.assertNotIn(grandchild.id, self.dag.nodes)
        
        # Check that the other nodes are still in the DAG
        self.assertIn(root.id, self.dag.nodes)
        self.assertIn(child2.id, self.dag.nodes)
    
    def test_delete_nonexistent_node(self):
        """Test that deleting a nonexistent node returns an empty list."""
        # Try to delete a nonexistent node
        deleted_nodes = self.dag.delete_node("nonexistent-id")
        
        # Check that an empty list was returned
        self.assertEqual(deleted_nodes, [])
    
    def test_get_descendants(self):
        """Test that the descendants of a node can be retrieved."""
        # Create a tree of nodes
        root = self.dag.add_node("Root")
        child1 = self.dag.add_node("Child 1", root.id)
        child2 = self.dag.add_node("Child 2", root.id)
        grandchild1 = self.dag.add_node("Grandchild 1", child1.id)
        grandchild2 = self.dag.add_node("Grandchild 2", child1.id)
        
        # Get the descendants of root
        descendants = self.dag._get_descendants(root.id)
        
        # Check that the correct descendants were returned
        self.assertEqual(len(descendants), 4)
        self.assertIn(child1.id, descendants)
        self.assertIn(child2.id, descendants)
        self.assertIn(grandchild1.id, descendants)
        self.assertIn(grandchild2.id, descendants)
        
        # Get the descendants of child1
        descendants = self.dag._get_descendants(child1.id)
        
        # Check that the correct descendants were returned
        self.assertEqual(len(descendants), 2)
        self.assertIn(grandchild1.id, descendants)
        self.assertIn(grandchild2.id, descendants)

if __name__ == "__main__":
    unittest.main()