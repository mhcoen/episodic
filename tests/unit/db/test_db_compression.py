"""
Unit tests for database compression operations.

Tests the db_compression module including compression storage
and statistics retrieval.
"""

import pytest


class TestCompressionStorage:
    """Test compression storage operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_compression, db_nodes
        self.db_compression = db_compression
        self.db_nodes = db_nodes

    def test_store_compression_basic(self):
        """Test storing a basic compression."""
        # Create some nodes first
        node1, _ = self.db_nodes.insert_node(content="Message 1", role="user")
        node2, _ = self.db_nodes.insert_node(
            content="Message 2", parent_id=node1, role="assistant"
        )
        compressed_id, _ = self.db_nodes.insert_node(
            content="Compressed summary", parent_id=node1, role="system"
        )

        self.db_compression.store_compression(
            compressed_node_id=compressed_id,
            original_branch_head=node2,
            compressed_content="Summary of messages 1 and 2",
            original_node_ids=[node1, node2]
        )

        stats = self.db_compression.get_compression_stats()
        assert stats['total_compressions'] == 1
        assert stats['total_nodes_compressed'] == 2

    def test_store_compression_with_metadata(self):
        """Test storing compression with metadata."""
        node1, _ = self.db_nodes.insert_node(content="Message 1", role="user")
        node2, _ = self.db_nodes.insert_node(
            content="Message 2", parent_id=node1, role="assistant"
        )
        compressed_id, _ = self.db_nodes.insert_node(
            content="Summary", parent_id=node1, role="system"
        )

        metadata = {
            'compression_ratio': 0.5,
            'original_tokens': 200,
            'compressed_tokens': 100,
            'model': 'test-model'
        }

        self.db_compression.store_compression(
            compressed_node_id=compressed_id,
            original_branch_head=node2,
            compressed_content="Summary",
            original_node_ids=[node1, node2],
            metadata=metadata
        )

        stats = self.db_compression.get_compression_stats()
        assert stats['total_compressions'] == 1
        assert stats['average_compression_ratio'] == 0.5

    def test_store_multiple_compressions(self):
        """Test storing multiple compressions."""
        for i in range(3):
            node1, _ = self.db_nodes.insert_node(
                content=f"Message {i}.1", role="user"
            )
            node2, _ = self.db_nodes.insert_node(
                content=f"Message {i}.2", parent_id=node1, role="assistant"
            )
            compressed_id, _ = self.db_nodes.insert_node(
                content=f"Summary {i}", parent_id=node1, role="system"
            )

            self.db_compression.store_compression(
                compressed_node_id=compressed_id,
                original_branch_head=node2,
                compressed_content=f"Summary {i}",
                original_node_ids=[node1, node2]
            )

        stats = self.db_compression.get_compression_stats()
        assert stats['total_compressions'] == 3
        # 3 compressions * 2 nodes each = 6 total nodes compressed
        assert stats['total_nodes_compressed'] == 6


class TestCompressionStats:
    """Test compression statistics retrieval."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_compression, db_nodes
        self.db_compression = db_compression
        self.db_nodes = db_nodes

    def test_stats_empty_database(self):
        """Test getting stats with no compressions."""
        stats = self.db_compression.get_compression_stats()

        assert stats['total_compressions'] == 0
        assert stats['total_nodes_compressed'] == 0
        assert stats['average_compression_ratio'] == 0

    def test_stats_with_data(self):
        """Test getting stats with compression data."""
        # Create a compression with metadata
        node1, _ = self.db_nodes.insert_node(content="Test", role="user")
        compressed_id, _ = self.db_nodes.insert_node(
            content="Summary", role="system"
        )

        self.db_compression.store_compression(
            compressed_node_id=compressed_id,
            original_branch_head=node1,
            compressed_content="Summary",
            original_node_ids=[node1],
            metadata={'compression_ratio': 0.7}
        )

        stats = self.db_compression.get_compression_stats()

        assert stats['total_compressions'] == 1
        assert stats['total_nodes_compressed'] == 1
        assert stats['average_compression_ratio'] == 0.7
        assert 'most_recent_compression' in stats
        assert stats['most_recent_compression']['node_id'] == compressed_id

    def test_stats_average_compression_ratio(self):
        """Test average compression ratio calculation."""
        ratios = [0.5, 0.6, 0.7]

        for i, ratio in enumerate(ratios):
            node_id, _ = self.db_nodes.insert_node(
                content=f"Message {i}", role="user"
            )
            compressed_id, _ = self.db_nodes.insert_node(
                content=f"Summary {i}", role="system"
            )

            self.db_compression.store_compression(
                compressed_node_id=compressed_id,
                original_branch_head=node_id,
                compressed_content=f"Summary {i}",
                original_node_ids=[node_id],
                metadata={'compression_ratio': ratio}
            )

        stats = self.db_compression.get_compression_stats()

        # Average of 0.5, 0.6, 0.7 = 0.6
        assert abs(stats['average_compression_ratio'] - 0.6) < 0.01

    def test_stats_ignores_null_ratios(self):
        """Test that stats ignore compressions without ratio metadata."""
        # Create compression with ratio
        node1, _ = self.db_nodes.insert_node(content="Test 1", role="user")
        compressed1, _ = self.db_nodes.insert_node(content="Summary 1", role="system")
        self.db_compression.store_compression(
            compressed_node_id=compressed1,
            original_branch_head=node1,
            compressed_content="Summary 1",
            original_node_ids=[node1],
            metadata={'compression_ratio': 0.5}
        )

        # Create compression without ratio
        node2, _ = self.db_nodes.insert_node(content="Test 2", role="user")
        compressed2, _ = self.db_nodes.insert_node(content="Summary 2", role="system")
        self.db_compression.store_compression(
            compressed_node_id=compressed2,
            original_branch_head=node2,
            compressed_content="Summary 2",
            original_node_ids=[node2],
            metadata=None  # No metadata
        )

        stats = self.db_compression.get_compression_stats()

        # Average should only consider the one with ratio
        assert stats['total_compressions'] == 2
        assert stats['average_compression_ratio'] == 0.5


class TestCompressionNodeMapping:
    """Test compression node mapping."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_compression, db_nodes
        from episodic.db_connection import get_connection
        self.db_compression = db_compression
        self.db_nodes = db_nodes
        self.get_connection = get_connection

    def test_node_mapping_stored(self):
        """Test that individual node mappings are stored."""
        node1, _ = self.db_nodes.insert_node(content="Message 1", role="user")
        node2, _ = self.db_nodes.insert_node(
            content="Message 2", parent_id=node1, role="assistant"
        )
        node3, _ = self.db_nodes.insert_node(
            content="Message 3", parent_id=node2, role="user"
        )
        compressed_id, _ = self.db_nodes.insert_node(
            content="Summary", role="system"
        )

        self.db_compression.store_compression(
            compressed_node_id=compressed_id,
            original_branch_head=node3,
            compressed_content="Summary of 3 messages",
            original_node_ids=[node1, node2, node3]
        )

        # Query the mapping table directly
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT original_node_id FROM compression_nodes WHERE compression_id = ?",
                (compressed_id,)
            )
            mapped_nodes = [row[0] for row in c.fetchall()]

        assert len(mapped_nodes) == 3
        assert node1 in mapped_nodes
        assert node2 in mapped_nodes
        assert node3 in mapped_nodes
