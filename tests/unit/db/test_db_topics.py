"""
Unit tests for database topic operations.

Tests the db_topics module including topic storage, retrieval,
and update operations.
"""

import pytest


class TestTopicStorage:
    """Test topic storage operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_topics, db_nodes
        self.db_topics = db_topics
        self.db_nodes = db_nodes

    def test_store_topic_basic(self):
        """Test storing a basic topic."""
        # Create a node first
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")

        self.db_topics.store_topic(
            name="Test Topic",
            start_node_id=node_id,
            confidence="detected"
        )

        topics = self.db_topics.get_all_topics()
        assert len(topics) == 1
        assert topics[0]['name'] == "Test Topic"
        assert topics[0]['start_node_id'] == node_id
        assert topics[0]['end_node_id'] is None

    def test_store_topic_with_end_node(self):
        """Test storing a topic with an end node."""
        start_id, _ = self.db_nodes.insert_node(content="Start", role="user")
        end_id, _ = self.db_nodes.insert_node(
            content="End", parent_id=start_id, role="assistant"
        )

        self.db_topics.store_topic(
            name="Complete Topic",
            start_node_id=start_id,
            end_node_id=end_id,
            confidence="manual"
        )

        topics = self.db_topics.get_all_topics()
        assert len(topics) == 1
        assert topics[0]['end_node_id'] == end_id
        assert topics[0]['confidence'] == "manual"

    def test_store_multiple_topics(self):
        """Test storing multiple topics."""
        node1, _ = self.db_nodes.insert_node(content="Topic 1", role="user")
        node2, _ = self.db_nodes.insert_node(content="Topic 2", role="user")
        node3, _ = self.db_nodes.insert_node(content="Topic 3", role="user")

        self.db_topics.store_topic("Topic A", node1)
        self.db_topics.store_topic("Topic B", node2)
        self.db_topics.store_topic("Topic C", node3)

        topics = self.db_topics.get_all_topics()
        assert len(topics) == 3


class TestTopicRetrieval:
    """Test topic retrieval operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_topics, db_nodes
        self.db_topics = db_topics
        self.db_nodes = db_nodes

    def test_get_recent_topics_default_limit(self):
        """Test getting recent topics with default limit."""
        # Create 15 topics
        for i in range(15):
            node_id, _ = self.db_nodes.insert_node(content=f"Msg {i}", role="user")
            self.db_topics.store_topic(f"Topic {i}", node_id)

        topics = self.db_topics.get_recent_topics()

        # Default limit is 10
        assert len(topics) == 10
        # Should be in chronological order (oldest first)
        assert topics[0]['name'] == "Topic 5"  # The 6th topic (skipping first 5)
        assert topics[-1]['name'] == "Topic 14"

    def test_get_recent_topics_custom_limit(self):
        """Test getting recent topics with custom limit."""
        for i in range(10):
            node_id, _ = self.db_nodes.insert_node(content=f"Msg {i}", role="user")
            self.db_topics.store_topic(f"Topic {i}", node_id)

        topics = self.db_topics.get_recent_topics(limit=5)

        assert len(topics) == 5
        # Should have the 5 most recent topics
        assert topics[0]['name'] == "Topic 5"
        assert topics[-1]['name'] == "Topic 9"

    def test_get_recent_topics_no_limit(self):
        """Test getting all topics with no limit."""
        for i in range(5):
            node_id, _ = self.db_nodes.insert_node(content=f"Msg {i}", role="user")
            self.db_topics.store_topic(f"Topic {i}", node_id)

        topics = self.db_topics.get_recent_topics(limit=None)

        assert len(topics) == 5

    def test_get_recent_topics_empty(self):
        """Test getting recent topics when none exist."""
        topics = self.db_topics.get_recent_topics()
        assert topics == []

    def test_get_all_topics(self):
        """Test getting all topics."""
        for i in range(5):
            node_id, _ = self.db_nodes.insert_node(content=f"Msg {i}", role="user")
            self.db_topics.store_topic(f"Topic {i}", node_id)

        topics = self.db_topics.get_all_topics()

        assert len(topics) == 5
        # Should be in chronological order
        assert topics[0]['name'] == "Topic 0"
        assert topics[4]['name'] == "Topic 4"


class TestTopicUpdate:
    """Test topic update operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_topics, db_nodes
        self.db_topics = db_topics
        self.db_nodes = db_nodes

    def test_update_topic_end_node(self):
        """Test updating a topic's end node."""
        start_id, _ = self.db_nodes.insert_node(content="Start", role="user")
        end_id, _ = self.db_nodes.insert_node(
            content="End", parent_id=start_id, role="assistant"
        )

        # Store topic without end node
        self.db_topics.store_topic("Test Topic", start_id)

        # Update with end node
        rows = self.db_topics.update_topic_end_node("Test Topic", start_id, end_id)

        assert rows == 1
        topics = self.db_topics.get_all_topics()
        assert topics[0]['end_node_id'] == end_id

    def test_update_topic_end_node_not_found(self):
        """Test updating end node for non-existent topic."""
        rows = self.db_topics.update_topic_end_node(
            "Non-existent", "fake-id", "fake-end"
        )
        assert rows == 0

    def test_update_topic_name(self):
        """Test updating a topic's name."""
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")
        self.db_topics.store_topic("Old Name", node_id)

        rows = self.db_topics.update_topic_name("Old Name", node_id, "New Name")

        assert rows == 1
        topics = self.db_topics.get_all_topics()
        assert topics[0]['name'] == "New Name"

    def test_update_topic_name_not_found(self):
        """Test updating name for non-existent topic."""
        rows = self.db_topics.update_topic_name(
            "Non-existent", "fake-id", "New Name"
        )
        assert rows == 0

    def test_update_specific_topic_when_multiple_exist(self):
        """Test updating a specific topic when there are multiple with same name."""
        node1, _ = self.db_nodes.insert_node(content="Msg 1", role="user")
        node2, _ = self.db_nodes.insert_node(content="Msg 2", role="user")

        # Create two topics with the same name (like recurring discussions)
        self.db_topics.store_topic("Repeated Topic", node1)
        self.db_topics.store_topic("Repeated Topic", node2)

        # Update only the second one
        end_id, _ = self.db_nodes.insert_node(content="End", role="assistant")
        rows = self.db_topics.update_topic_end_node("Repeated Topic", node2, end_id)

        assert rows == 1
        topics = self.db_topics.get_all_topics()
        # First topic should be unchanged
        assert topics[0]['end_node_id'] is None
        # Second topic should be updated
        assert topics[1]['end_node_id'] == end_id


class TestTopicConfidence:
    """Test topic confidence field handling."""

    @pytest.fixture(autouse=True)
    def setup_db(self, temp_database):
        """Set up database for each test."""
        self.db_path = temp_database
        from episodic import db_topics, db_nodes
        self.db_topics = db_topics
        self.db_nodes = db_nodes

    def test_store_topic_with_confidence(self):
        """Test storing topic with confidence level."""
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")

        self.db_topics.store_topic("Confident Topic", node_id, confidence="detected")

        topics = self.db_topics.get_all_topics()
        assert topics[0]['confidence'] == "detected"

    def test_store_topic_without_confidence(self):
        """Test storing topic without confidence level."""
        node_id, _ = self.db_nodes.insert_node(content="Test", role="user")

        self.db_topics.store_topic("No Confidence", node_id)

        topics = self.db_topics.get_all_topics()
        assert topics[0]['confidence'] is None

    def test_different_confidence_levels(self):
        """Test storing topics with different confidence levels."""
        confidence_levels = ["detected", "initial", "manual", None]

        for i, conf in enumerate(confidence_levels):
            node_id, _ = self.db_nodes.insert_node(content=f"Msg {i}", role="user")
            self.db_topics.store_topic(f"Topic {i}", node_id, confidence=conf)

        topics = self.db_topics.get_all_topics()
        assert len(topics) == 4
        assert topics[0]['confidence'] == "detected"
        assert topics[1]['confidence'] == "initial"
        assert topics[2]['confidence'] == "manual"
        assert topics[3]['confidence'] is None
