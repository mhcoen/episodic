"""
Unit tests for the core module.

This module contains tests for the ConversationDAG class and its methods.
"""

import pytest
from episodic.core import ConversationDAG, Node


class TestNode:
    """Test the Node class."""

    def test_node_creation(self):
        """Test that a node can be created with the correct properties."""
        message = "Test message"
        node = Node(message)

        assert node.message == message
        assert node.parent_id is None
        assert node.id is not None
        assert node.timestamp is not None

    def test_node_with_parent(self):
        """Test that a node can be created with a parent ID."""
        message = "Child message"
        parent_id = "parent-id"
        node = Node(message, parent_id)

        assert node.message == message
        assert node.parent_id == parent_id

    def test_to_dict(self):
        """Test that a node can be converted to a dictionary."""
        message = "Test message"
        node = Node(message)

        node_dict = node.to_dict()

        assert node_dict["id"] == node.id
        assert node_dict["message"] == message
        assert node_dict["timestamp"] == node.timestamp
        assert node_dict["parent_id"] is None


class TestConversationDAG:
    """Test the ConversationDAG class."""

    @pytest.fixture(autouse=True)
    def setup_dag(self):
        """Set up a new ConversationDAG for each test."""
        self.dag = ConversationDAG()

    def test_add_node(self):
        """Test that a node can be added to the DAG."""
        message = "Test message"
        node = self.dag.add_node(message)

        assert node.id in self.dag.nodes
        assert self.dag.nodes[node.id] == node

    def test_add_node_with_parent(self):
        """Test that a node can be added with a parent."""
        parent_message = "Parent message"
        parent = self.dag.add_node(parent_message)

        child_message = "Child message"
        child = self.dag.add_node(child_message, parent.id)

        assert child.parent_id == parent.id

    def test_get_node(self):
        """Test that a node can be retrieved from the DAG."""
        message = "Test message"
        node = self.dag.add_node(message)

        retrieved_node = self.dag.get_node(node.id)

        assert retrieved_node == node

    def test_get_nonexistent_node(self):
        """Test that getting a nonexistent node returns None."""
        node = self.dag.get_node("nonexistent-id")
        assert node is None

    def test_get_ancestry(self):
        """Test that the ancestry of a node can be retrieved."""
        root = self.dag.add_node("Root")
        child = self.dag.add_node("Child", root.id)
        grandchild = self.dag.add_node("Grandchild", child.id)

        ancestry = self.dag.get_ancestry(grandchild.id)

        assert len(ancestry) == 3
        assert ancestry[0] == root
        assert ancestry[1] == child
        assert ancestry[2] == grandchild

    def test_delete_node(self):
        """Test that a node and its descendants can be deleted."""
        root = self.dag.add_node("Root")
        child1 = self.dag.add_node("Child 1", root.id)
        child2 = self.dag.add_node("Child 2", root.id)
        grandchild = self.dag.add_node("Grandchild", child1.id)

        deleted_nodes = self.dag.delete_node(child1.id)

        assert len(deleted_nodes) == 2
        assert child1.id in deleted_nodes
        assert grandchild.id in deleted_nodes
        assert child1.id not in self.dag.nodes
        assert grandchild.id not in self.dag.nodes
        assert root.id in self.dag.nodes
        assert child2.id in self.dag.nodes

    def test_delete_nonexistent_node(self):
        """Test that deleting a nonexistent node returns an empty list."""
        deleted_nodes = self.dag.delete_node("nonexistent-id")
        assert deleted_nodes == []

    def test_get_descendants(self):
        """Test that the descendants of a node can be retrieved."""
        root = self.dag.add_node("Root")
        child1 = self.dag.add_node("Child 1", root.id)
        child2 = self.dag.add_node("Child 2", root.id)
        grandchild1 = self.dag.add_node("Grandchild 1", child1.id)
        grandchild2 = self.dag.add_node("Grandchild 2", child1.id)

        # Get descendants of root
        descendants = self.dag._get_descendants(root.id)
        assert len(descendants) == 4
        assert child1.id in descendants
        assert child2.id in descendants
        assert grandchild1.id in descendants
        assert grandchild2.id in descendants

        # Get descendants of child1
        descendants = self.dag._get_descendants(child1.id)
        assert len(descendants) == 2
        assert grandchild1.id in descendants
        assert grandchild2.id in descendants
