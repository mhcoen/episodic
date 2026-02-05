"""
Unit tests for the db_nodes module.

Tests node CRUD operations, ancestry traversal, and reference resolution.
"""

import pytest
from episodic import db_nodes


class TestInsertNode:
    """Test node insertion."""

    def test_insert_node_returns_ids(self, temp_database):
        """insert_node returns (node_id, short_id) tuple."""
        node_id, short_id = db_nodes.insert_node("Hello world")

        assert node_id is not None
        assert short_id is not None
        assert len(node_id) == 36  # UUID format
        assert len(short_id) == 2  # Short ID format

    def test_insert_node_with_parent(self, temp_database):
        """Nodes can be inserted with a parent reference."""
        parent_id, _ = db_nodes.insert_node("Parent message")
        child_id, _ = db_nodes.insert_node("Child message", parent_id=parent_id)

        child = db_nodes.get_node(child_id)
        assert child["parent_id"] == parent_id

    def test_insert_node_with_role(self, temp_database):
        """Nodes store role information."""
        node_id, _ = db_nodes.insert_node("User message", role="user")

        node = db_nodes.get_node(node_id)
        assert node["role"] == "user"

    def test_insert_node_with_provider_model(self, temp_database):
        """Nodes store provider and model information."""
        node_id, _ = db_nodes.insert_node(
            "Response", role="assistant", provider="openai", model="gpt-4"
        )

        node = db_nodes.get_node(node_id)
        assert node["provider"] == "openai"
        assert node["model"] == "gpt-4"

    def test_insert_node_updates_head(self, temp_database):
        """Inserting a node updates the head pointer."""
        node_id, _ = db_nodes.insert_node("First message")
        assert db_nodes.get_head() == node_id

        second_id, _ = db_nodes.insert_node("Second message", parent_id=node_id)
        assert db_nodes.get_head() == second_id

    def test_insert_meta_query_node(self, temp_database):
        """Meta-query nodes are marked appropriately."""
        node_id, _ = db_nodes.insert_node("Meta query", is_meta_query=True)

        node = db_nodes.get_node(node_id)
        assert node["is_meta_query"] == 1


class TestGetNode:
    """Test node retrieval."""

    def test_get_node_by_id(self, temp_database):
        """Nodes can be retrieved by full ID."""
        node_id, _ = db_nodes.insert_node("Test content")

        node = db_nodes.get_node(node_id)
        assert node is not None
        assert node["content"] == "Test content"

    def test_get_node_by_short_id(self, temp_database):
        """Nodes can be retrieved by short ID."""
        node_id, short_id = db_nodes.insert_node("Test content")

        node = db_nodes.get_node(short_id)
        assert node is not None
        assert node["id"] == node_id

    def test_get_nonexistent_node(self, temp_database):
        """Getting a nonexistent node returns None."""
        node = db_nodes.get_node("nonexistent-id")
        assert node is None


class TestAncestry:
    """Test ancestry chain traversal."""

    def test_get_ancestry_single_node(self, temp_database):
        """Ancestry of a single node is just itself."""
        node_id, _ = db_nodes.insert_node("Root")

        ancestry = db_nodes.get_ancestry(node_id)
        assert len(ancestry) == 1
        assert ancestry[0]["id"] == node_id

    def test_get_ancestry_chain(self, temp_database):
        """Ancestry returns nodes from root to current."""
        root_id, _ = db_nodes.insert_node("Root")
        child_id, _ = db_nodes.insert_node("Child", parent_id=root_id)
        grandchild_id, _ = db_nodes.insert_node("Grandchild", parent_id=child_id)

        ancestry = db_nodes.get_ancestry(grandchild_id)

        assert len(ancestry) == 3
        assert ancestry[0]["id"] == root_id
        assert ancestry[1]["id"] == child_id
        assert ancestry[2]["id"] == grandchild_id

    def test_get_ancestry_by_short_id(self, temp_database):
        """Ancestry can be retrieved using short ID."""
        root_id, _ = db_nodes.insert_node("Root")
        child_id, child_short = db_nodes.insert_node("Child", parent_id=root_id)

        ancestry = db_nodes.get_ancestry(child_short)
        assert len(ancestry) == 2


class TestHeadManagement:
    """Test head pointer management."""

    def test_get_head_empty_db(self, temp_database):
        """Head is None when no nodes exist."""
        head = db_nodes.get_head()
        assert head is None

    def test_set_head(self, temp_database):
        """Head can be set explicitly."""
        first_id, _ = db_nodes.insert_node("First")
        second_id, _ = db_nodes.insert_node("Second", parent_id=first_id)

        db_nodes.set_head(first_id)
        assert db_nodes.get_head() == first_id


class TestRecentNodes:
    """Test recent nodes retrieval."""

    def test_get_recent_nodes_empty(self, temp_database):
        """Returns empty list when no nodes exist."""
        recent = db_nodes.get_recent_nodes()
        assert recent == []

    def test_get_recent_nodes_respects_limit(self, temp_database):
        """Returns at most limit nodes."""
        parent_id = None
        for i in range(10):
            node_id, _ = db_nodes.insert_node(f"Message {i}", parent_id=parent_id)
            parent_id = node_id

        recent = db_nodes.get_recent_nodes(limit=3)
        assert len(recent) == 3

    def test_get_recent_nodes_newest_first(self, temp_database):
        """Returns nodes in newest-first order."""
        first_id, _ = db_nodes.insert_node("First")
        second_id, _ = db_nodes.insert_node("Second", parent_id=first_id)
        third_id, _ = db_nodes.insert_node("Third", parent_id=second_id)

        recent = db_nodes.get_recent_nodes(limit=3)
        assert recent[0]["id"] == third_id
        assert recent[2]["id"] == first_id


class TestGetAllNodes:
    """Test retrieving all nodes."""

    def test_get_all_nodes_empty(self, temp_database):
        """Returns empty list when no nodes exist."""
        nodes = db_nodes.get_all_nodes()
        assert nodes == []

    def test_get_all_nodes_returns_all(self, temp_database):
        """Returns all nodes in creation order."""
        db_nodes.insert_node("First")
        db_nodes.insert_node("Second")
        db_nodes.insert_node("Third")

        nodes = db_nodes.get_all_nodes()
        assert len(nodes) == 3
        assert nodes[0]["content"] == "First"
        assert nodes[2]["content"] == "Third"


class TestChildrenAndDescendants:
    """Test child and descendant retrieval."""

    def test_get_children(self, temp_database):
        """Get direct children of a node."""
        parent_id, _ = db_nodes.insert_node("Parent")
        child1_id, _ = db_nodes.insert_node("Child 1", parent_id=parent_id)
        child2_id, _ = db_nodes.insert_node("Child 2", parent_id=parent_id)

        children = db_nodes.get_children(parent_id)
        child_ids = {c["id"] for c in children}

        assert len(children) == 2
        assert child1_id in child_ids
        assert child2_id in child_ids

    def test_get_children_empty(self, temp_database):
        """Node with no children returns empty list."""
        node_id, _ = db_nodes.insert_node("Leaf")

        children = db_nodes.get_children(node_id)
        assert children == []

    def test_get_descendants(self, temp_database):
        """Get all descendants of a node."""
        root_id, _ = db_nodes.insert_node("Root")
        child_id, _ = db_nodes.insert_node("Child", parent_id=root_id)
        grandchild_id, _ = db_nodes.insert_node("Grandchild", parent_id=child_id)

        descendants = db_nodes.get_descendants(root_id)
        descendant_ids = {d["id"] for d in descendants}

        assert child_id in descendant_ids
        assert grandchild_id in descendant_ids
        assert root_id not in descendant_ids


class TestDeleteNode:
    """Test node deletion."""

    def test_delete_leaf_node(self, temp_database):
        """Leaf nodes can be deleted."""
        parent_id, _ = db_nodes.insert_node("Parent")
        child_id, _ = db_nodes.insert_node("Child", parent_id=parent_id)

        deleted = db_nodes.delete_node(child_id)
        assert deleted == 1
        assert db_nodes.get_node(child_id) is None

    def test_delete_node_with_children_fails(self, temp_database):
        """Cannot delete nodes that have children."""
        parent_id, _ = db_nodes.insert_node("Parent")
        db_nodes.insert_node("Child", parent_id=parent_id)

        with pytest.raises(ValueError, match="has.*children"):
            db_nodes.delete_node(parent_id)

    def test_delete_head_updates_to_parent(self, temp_database):
        """Deleting head node updates head to parent."""
        parent_id, _ = db_nodes.insert_node("Parent")
        child_id, _ = db_nodes.insert_node("Child", parent_id=parent_id)

        assert db_nodes.get_head() == child_id
        db_nodes.delete_node(child_id)
        assert db_nodes.get_head() == parent_id

    def test_delete_nonexistent_node(self, temp_database):
        """Deleting nonexistent node returns 0."""
        deleted = db_nodes.delete_node("nonexistent-id")
        assert deleted == 0


class TestResolveNodeRef:
    """Test node reference resolution."""

    def test_resolve_head(self, temp_database):
        """HEAD resolves to current head node."""
        node_id, _ = db_nodes.insert_node("Current head")

        resolved = db_nodes.resolve_node_ref("HEAD")
        assert resolved == node_id

    def test_resolve_head_case_insensitive(self, temp_database):
        """HEAD resolution is case-insensitive."""
        node_id, _ = db_nodes.insert_node("Current head")

        assert db_nodes.resolve_node_ref("head") == node_id
        assert db_nodes.resolve_node_ref("Head") == node_id

    def test_resolve_root(self, temp_database):
        """ROOT resolves to root node."""
        root_id, _ = db_nodes.insert_node("Root")
        db_nodes.insert_node("Child", parent_id=root_id)

        resolved = db_nodes.resolve_node_ref("ROOT")
        assert resolved == root_id

    def test_resolve_relative_ref(self, temp_database):
        """HEAD~N resolves to N steps back."""
        root_id, _ = db_nodes.insert_node("Root")
        child_id, _ = db_nodes.insert_node("Child", parent_id=root_id)
        grandchild_id, _ = db_nodes.insert_node("Grandchild", parent_id=child_id)

        assert db_nodes.resolve_node_ref("HEAD~1") == child_id
        assert db_nodes.resolve_node_ref("HEAD~2") == root_id

    def test_resolve_short_id(self, temp_database):
        """Short IDs resolve to full node ID."""
        node_id, short_id = db_nodes.insert_node("Test")

        resolved = db_nodes.resolve_node_ref(short_id)
        assert resolved == node_id

    def test_resolve_none_returns_none(self, temp_database):
        """None/empty ref returns None."""
        assert db_nodes.resolve_node_ref(None) is None
        assert db_nodes.resolve_node_ref("") is None

    def test_resolve_nonexistent_returns_none(self, temp_database):
        """Nonexistent ref returns None."""
        assert db_nodes.resolve_node_ref("xyz123") is None
