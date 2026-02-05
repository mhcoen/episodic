"""
Unit tests for the db_topics module.

Tests topic CRUD operations and querying.
"""

import pytest
from episodic import db_nodes, db_topics


class TestStoreTopic:
    """Test topic creation."""

    def test_store_topic_basic(self, temp_database):
        """Topics can be created with name and start node."""
        node_id, _ = db_nodes.insert_node("First message")

        db_topics.store_topic("test-topic", node_id)

        topics = db_topics.get_all_topics()
        assert len(topics) == 1
        assert topics[0]["name"] == "test-topic"
        assert topics[0]["start_node_id"] == node_id

    def test_store_topic_with_end_node(self, temp_database):
        """Topics can be created with end node."""
        start_id, _ = db_nodes.insert_node("Start")
        end_id, _ = db_nodes.insert_node("End", parent_id=start_id)

        db_topics.store_topic("bounded-topic", start_id, end_node_id=end_id)

        topics = db_topics.get_all_topics()
        assert topics[0]["end_node_id"] == end_id

    def test_store_topic_with_confidence(self, temp_database):
        """Topics can be created with confidence level."""
        node_id, _ = db_nodes.insert_node("Message")

        db_topics.store_topic("confident-topic", node_id, confidence="detected")

        topics = db_topics.get_all_topics()
        assert topics[0]["confidence"] == "detected"


class TestGetRecentTopics:
    """Test recent topics retrieval."""

    def test_get_recent_topics_empty(self, temp_database):
        """Returns empty list when no topics exist."""
        topics = db_topics.get_recent_topics()
        assert topics == []

    def test_get_recent_topics_respects_limit(self, temp_database):
        """Returns at most limit topics."""
        for i in range(5):
            node_id, _ = db_nodes.insert_node(f"Message {i}")
            db_topics.store_topic(f"topic-{i}", node_id)

        topics = db_topics.get_recent_topics(limit=3)
        assert len(topics) == 3

    def test_get_recent_topics_chronological_order(self, temp_database):
        """Returns topics in chronological order (oldest first)."""
        node1, _ = db_nodes.insert_node("First")
        db_topics.store_topic("topic-1", node1)

        node2, _ = db_nodes.insert_node("Second")
        db_topics.store_topic("topic-2", node2)

        node3, _ = db_nodes.insert_node("Third")
        db_topics.store_topic("topic-3", node3)

        topics = db_topics.get_recent_topics(limit=3)
        assert topics[0]["name"] == "topic-1"
        assert topics[2]["name"] == "topic-3"

    def test_get_recent_topics_no_limit(self, temp_database):
        """Passing None returns all topics."""
        for i in range(5):
            node_id, _ = db_nodes.insert_node(f"Message {i}")
            db_topics.store_topic(f"topic-{i}", node_id)

        topics = db_topics.get_recent_topics(limit=None)
        assert len(topics) == 5


class TestGetAllTopics:
    """Test retrieving all topics."""

    def test_get_all_topics_empty(self, temp_database):
        """Returns empty list when no topics exist."""
        topics = db_topics.get_all_topics()
        assert topics == []

    def test_get_all_topics_returns_all(self, temp_database):
        """Returns all topics in creation order."""
        node1, _ = db_nodes.insert_node("First")
        db_topics.store_topic("alpha", node1)

        node2, _ = db_nodes.insert_node("Second")
        db_topics.store_topic("beta", node2)

        topics = db_topics.get_all_topics()
        assert len(topics) == 2
        assert topics[0]["name"] == "alpha"
        assert topics[1]["name"] == "beta"


class TestUpdateTopicEndNode:
    """Test updating topic end node."""

    def test_update_end_node(self, temp_database):
        """End node can be updated."""
        start_id, _ = db_nodes.insert_node("Start")
        db_topics.store_topic("my-topic", start_id)

        new_end_id, _ = db_nodes.insert_node("New end", parent_id=start_id)
        rows = db_topics.update_topic_end_node("my-topic", start_id, new_end_id)

        assert rows == 1
        topics = db_topics.get_all_topics()
        assert topics[0]["end_node_id"] == new_end_id

    def test_update_end_node_wrong_name(self, temp_database):
        """Update returns 0 for non-matching name."""
        start_id, _ = db_nodes.insert_node("Start")
        db_topics.store_topic("my-topic", start_id)

        rows = db_topics.update_topic_end_node("wrong-name", start_id, "new-end")
        assert rows == 0

    def test_update_end_node_wrong_start(self, temp_database):
        """Update returns 0 for non-matching start node."""
        start_id, _ = db_nodes.insert_node("Start")
        db_topics.store_topic("my-topic", start_id)

        rows = db_topics.update_topic_end_node("my-topic", "wrong-start", "new-end")
        assert rows == 0


class TestUpdateTopicName:
    """Test renaming topics."""

    def test_rename_topic(self, temp_database):
        """Topics can be renamed."""
        start_id, _ = db_nodes.insert_node("Start")
        db_topics.store_topic("old-name", start_id)

        rows = db_topics.update_topic_name("old-name", start_id, "new-name")

        assert rows == 1
        topics = db_topics.get_all_topics()
        assert topics[0]["name"] == "new-name"

    def test_rename_nonexistent_topic(self, temp_database):
        """Renaming nonexistent topic returns 0."""
        rows = db_topics.update_topic_name("nonexistent", "fake-id", "new-name")
        assert rows == 0

    def test_rename_requires_correct_start_node(self, temp_database):
        """Rename requires matching start node for disambiguation."""
        start1, _ = db_nodes.insert_node("Start 1")
        start2, _ = db_nodes.insert_node("Start 2")

        db_topics.store_topic("same-name", start1)
        db_topics.store_topic("same-name", start2)

        # Only rename the one with start1
        rows = db_topics.update_topic_name("same-name", start1, "renamed")
        assert rows == 1

        topics = db_topics.get_all_topics()
        names = {t["name"] for t in topics}
        assert "renamed" in names
        assert "same-name" in names  # The other one still exists


class TestMultipleTopics:
    """Test scenarios with multiple topics."""

    def test_multiple_topics_same_start_node(self, temp_database):
        """Multiple topics can share the same start node."""
        node_id, _ = db_nodes.insert_node("Shared start")

        db_topics.store_topic("topic-a", node_id)
        db_topics.store_topic("topic-b", node_id)

        topics = db_topics.get_all_topics()
        assert len(topics) == 2

    def test_ongoing_vs_closed_topics(self, temp_database):
        """Topics can be ongoing (no end) or closed (with end)."""
        start1, _ = db_nodes.insert_node("Start 1")
        db_topics.store_topic("ongoing", start1)

        start2, _ = db_nodes.insert_node("Start 2")
        end2, _ = db_nodes.insert_node("End 2", parent_id=start2)
        db_topics.store_topic("closed", start2, end_node_id=end2)

        topics = db_topics.get_all_topics()

        ongoing = next(t for t in topics if t["name"] == "ongoing")
        closed = next(t for t in topics if t["name"] == "closed")

        assert ongoing["end_node_id"] is None
        assert closed["end_node_id"] == end2
