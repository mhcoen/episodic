"""
Unit tests for database node operations.

Tests the db_nodes module including node creation, retrieval,
ancestry, and deletion operations.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestNodeOperations:
    """Test basic node operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        # Import after database is initialized
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_insert_node_basic(self):
        """Test inserting a basic node."""
        node_id, short_id = self.db_nodes.insert_node(
            content="Test message",
            role="user"
        )

        assert node_id is not None
        assert short_id is not None
        assert len(short_id) >= 2  # Short ID has variable length

    def test_insert_node_with_parent(self):
        """Test inserting a node with a parent."""
        # Create parent
        parent_id, _ = self.db_nodes.insert_node(
            content="Parent message",
            role="user"
        )

        # Create child
        child_id, _ = self.db_nodes.insert_node(
            content="Child message",
            parent_id=parent_id,
            role="assistant"
        )

        # Verify relationship
        child = self.db_nodes.get_node(child_id)
        assert child['parent_id'] == parent_id

    def test_insert_node_with_provider_model(self):
        """Test inserting node with provider and model info."""
        node_id, _ = self.db_nodes.insert_node(
            content="Test message",
            role="assistant",
            provider="openai",
            model="gpt-4"
        )

        node = self.db_nodes.get_node(node_id)
        assert node['provider'] == "openai"
        assert node['model'] == "gpt-4"

    def test_get_node_by_id(self):
        """Test retrieving a node by its full ID."""
        node_id, _ = self.db_nodes.insert_node(
            content="Test content",
            role="user"
        )

        node = self.db_nodes.get_node(node_id)
        assert node is not None
        assert node['content'] == "Test content"
        assert node['role'] == "user"

    def test_get_node_by_short_id(self):
        """Test retrieving a node by its short ID."""
        node_id, short_id = self.db_nodes.insert_node(
            content="Test content",
            role="user"
        )

        node = self.db_nodes.get_node(short_id)
        assert node is not None
        assert node['id'] == node_id

    def test_get_nonexistent_node(self):
        """Test retrieving a node that doesn't exist."""
        node = self.db_nodes.get_node("nonexistent-id")
        assert node is None


class TestAncestry:
    """Test ancestry-related operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_get_ancestry_simple(self):
        """Test getting ancestry for a simple chain."""
        # Create a chain: root -> child -> grandchild
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )
        grandchild_id, _ = self.db_nodes.insert_node(
            content="Grandchild", parent_id=child_id, role="user"
        )

        ancestry = self.db_nodes.get_ancestry(grandchild_id)

        assert len(ancestry) == 3
        assert ancestry[0]['id'] == root_id  # Oldest first
        assert ancestry[1]['id'] == child_id
        assert ancestry[2]['id'] == grandchild_id  # Newest last

    def test_get_ancestry_single_node(self):
        """Test ancestry for a node with no parent."""
        node_id, _ = self.db_nodes.insert_node(content="Single", role="user")

        ancestry = self.db_nodes.get_ancestry(node_id)

        assert len(ancestry) == 1
        assert ancestry[0]['id'] == node_id

    def test_get_descendants(self):
        """Test getting all descendants of a node."""
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )
        grandchild_id, _ = self.db_nodes.insert_node(
            content="Grandchild", parent_id=child_id, role="user"
        )

        descendants = self.db_nodes.get_descendants(root_id)

        assert len(descendants) == 2
        descendant_ids = [d['id'] for d in descendants]
        assert child_id in descendant_ids
        assert grandchild_id in descendant_ids

    def test_get_children(self):
        """Test getting direct children of a node."""
        parent_id, _ = self.db_nodes.insert_node(content="Parent", role="user")
        child1_id, _ = self.db_nodes.insert_node(
            content="Child 1", parent_id=parent_id, role="assistant"
        )
        child2_id, _ = self.db_nodes.insert_node(
            content="Child 2", parent_id=parent_id, role="assistant"
        )

        children = self.db_nodes.get_children(parent_id)

        assert len(children) == 2
        child_ids = [c['id'] for c in children]
        assert child1_id in child_ids
        assert child2_id in child_ids


class TestHeadOperations:
    """Test head pointer operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_insert_updates_head(self):
        """Test that inserting a node updates the head pointer."""
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")

        head = self.db_nodes.get_head()
        assert head == node_id

    def test_set_head(self):
        """Test manually setting the head pointer."""
        node1_id, _ = self.db_nodes.insert_node(content="First", role="user")
        node2_id, _ = self.db_nodes.insert_node(
            content="Second", parent_id=node1_id, role="assistant"
        )

        # Head should be at node2
        assert self.db_nodes.get_head() == node2_id

        # Set head back to node1
        self.db_nodes.set_head(node1_id)
        assert self.db_nodes.get_head() == node1_id

    def test_get_recent_nodes(self):
        """Test getting recent nodes from head."""
        ids = []
        prev_id = None
        for i in range(5):
            node_id, _ = self.db_nodes.insert_node(
                content=f"Message {i}",
                parent_id=prev_id,
                role="user" if i % 2 == 0 else "assistant"
            )
            ids.append(node_id)
            prev_id = node_id

        recent = self.db_nodes.get_recent_nodes(limit=3)

        assert len(recent) == 3
        # Should be newest first
        assert recent[0]['id'] == ids[4]
        assert recent[1]['id'] == ids[3]
        assert recent[2]['id'] == ids[2]


class TestNodeDeletion:
    """Test node deletion operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_delete_leaf_node(self):
        """Test deleting a leaf node (no children)."""
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )

        # Delete the child (leaf)
        deleted = self.db_nodes.delete_node(child_id)

        assert deleted == 1
        assert self.db_nodes.get_node(child_id) is None
        assert self.db_nodes.get_node(root_id) is not None

    def test_delete_node_with_children_raises(self):
        """Test that deleting a node with children raises an error."""
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )

        with pytest.raises(ValueError, match="has.*children"):
            self.db_nodes.delete_node(root_id)

    def test_delete_head_updates_pointer(self):
        """Test that deleting the head node updates the head pointer."""
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )

        # Head should be at child
        assert self.db_nodes.get_head() == child_id

        # Delete child (head)
        self.db_nodes.delete_node(child_id)

        # Head should now be at root
        assert self.db_nodes.get_head() == root_id


class TestNodeReferenceResolution:
    """Test node reference resolution."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_resolve_head_reference(self):
        """Test resolving HEAD reference."""
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")

        resolved = self.db_nodes.resolve_node_ref("HEAD")
        assert resolved == node_id

    def test_resolve_root_reference(self):
        """Test resolving ROOT reference."""
        root_id, _ = self.db_nodes.insert_node(content="Root", role="user")
        child_id, _ = self.db_nodes.insert_node(
            content="Child", parent_id=root_id, role="assistant"
        )

        resolved = self.db_nodes.resolve_node_ref("ROOT")
        assert resolved == root_id

    def test_resolve_relative_reference(self):
        """Test resolving relative reference like HEAD~2."""
        node1_id, _ = self.db_nodes.insert_node(content="First", role="user")
        node2_id, _ = self.db_nodes.insert_node(
            content="Second", parent_id=node1_id, role="assistant"
        )
        node3_id, _ = self.db_nodes.insert_node(
            content="Third", parent_id=node2_id, role="user"
        )

        # HEAD~2 should resolve to first node
        resolved = self.db_nodes.resolve_node_ref("HEAD~2")
        assert resolved == node1_id

        # HEAD~1 should resolve to second node
        resolved = self.db_nodes.resolve_node_ref("HEAD~1")
        assert resolved == node2_id

    def test_resolve_direct_id(self):
        """Test resolving a direct node ID."""
        node_id, short_id = self.db_nodes.insert_node(content="Test", role="user")

        resolved = self.db_nodes.resolve_node_ref(node_id)
        assert resolved == node_id

        resolved = self.db_nodes.resolve_node_ref(short_id)
        assert resolved == node_id

    def test_resolve_nonexistent_reference(self):
        """Test resolving a reference that doesn't exist."""
        resolved = self.db_nodes.resolve_node_ref("nonexistent")
        assert resolved is None

    def test_resolve_none_reference(self):
        """Test resolving a None reference."""
        resolved = self.db_nodes.resolve_node_ref(None)
        assert resolved is None


class TestGetAllNodes:
    """Test getting all nodes."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_nodes
        self.db_nodes = db_nodes

    def test_get_all_nodes_empty(self):
        """Test getting all nodes from empty database."""
        nodes = self.db_nodes.get_all_nodes()
        assert nodes == []

    def test_get_all_nodes_with_data(self):
        """Test getting all nodes with data."""
        ids = []
        prev_id = None
        for i in range(3):
            node_id, _ = self.db_nodes.insert_node(
                content=f"Message {i}",
                parent_id=prev_id,
                role="user"
            )
            ids.append(node_id)
            prev_id = node_id

        nodes = self.db_nodes.get_all_nodes()

        assert len(nodes) == 3
        # Should be in creation order (oldest first by ROWID)
        assert nodes[0]['id'] == ids[0]
        assert nodes[1]['id'] == ids[1]
        assert nodes[2]['id'] == ids[2]
