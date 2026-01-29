"""
Tests for centroid computation in reactivation system.

Covers edge cases like empty embeddings arrays that can crash numpy.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestComputeMedoid:
    """Tests for compute_medoid function."""

    def test_empty_exchanges_returns_none(self):
        """Test that empty exchanges list returns None."""
        from episodic.recall.centroid import compute_medoid

        conn = MagicMock()
        result = compute_medoid(conn, [])
        assert result is None

    def test_single_exchange_returns_that_exchange(self):
        """Test that single exchange returns its user node ID."""
        from episodic.recall.centroid import compute_medoid

        conn = MagicMock()
        result = compute_medoid(conn, [("user_1", "asst_1")])
        assert result == "user_1"

    def test_empty_embeddings_array_fallback(self):
        """Test that empty embeddings array doesn't crash with numpy ambiguity error.

        This was the bug: when Chroma returns empty embeddings, np.array([])
        can't be used in boolean context like `if not embeddings`.
        """
        from episodic.recall.centroid import compute_medoid

        conn = MagicMock()

        # Mock Chroma to return empty embeddings
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': [],
            'embeddings': []
        }

        mock_rag = MagicMock()
        mock_rag.get_collection.return_value = mock_collection

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            # Should not raise "The truth value of an empty array is ambiguous"
            result = compute_medoid(conn, [("user_1", "asst_1"), ("user_2", "asst_2")])

            # Should fallback to most recent exchange
            assert result == "user_2"

    def test_partial_embeddings_fallback(self):
        """Test that partial embeddings (some None) still works."""
        from episodic.recall.centroid import compute_medoid

        conn = MagicMock()

        # Mock Chroma to return some None embeddings
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['user_1', 'user_2', 'user_3'],
            'embeddings': [
                np.array([0.1, 0.2, 0.3]),
                None,  # Missing embedding
                None,  # Missing embedding
            ]
        }

        mock_rag = MagicMock()
        mock_rag.get_collection.return_value = mock_collection

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            # With only 1 valid embedding, should fallback
            result = compute_medoid(conn, [("user_1", "asst_1"), ("user_2", "asst_2"), ("user_3", "asst_3")])

            # Should fallback to most recent exchange
            assert result == "user_3"

    def test_numpy_array_embeddings_work(self):
        """Test that numpy array embeddings are handled correctly."""
        from episodic.recall.centroid import compute_medoid

        conn = MagicMock()

        # Mock Chroma to return valid embeddings
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # user_1
            np.array([0.9, 0.1, 0.0]),  # user_2 - similar to user_1
            np.array([0.0, 0.0, 1.0]),  # user_3 - different
        ]

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['user_1', 'user_2', 'user_3'],
            'embeddings': embeddings
        }

        mock_rag = MagicMock()
        mock_rag.get_collection.return_value = mock_collection

        with patch('episodic.rag_collections.get_multi_collection_rag', return_value=mock_rag):
            result = compute_medoid(conn, [("user_1", "asst_1"), ("user_2", "asst_2"), ("user_3", "asst_3")])

            # user_1 or user_2 should be medoid (they're similar to each other)
            assert result in ['user_1', 'user_2']


class TestIsCheckpoint:
    """Tests for is_checkpoint function."""

    def test_checkpoint_intervals(self):
        """Test that checkpoint intervals are recognized."""
        from episodic.recall.centroid import is_checkpoint

        # All checkpoints should return True
        checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for n in checkpoints:
            assert is_checkpoint(n) is True, f"Expected {n} to be a checkpoint"

        # Non-checkpoints should return False
        non_checkpoints = [0, 3, 5, 6, 7, 9, 10, 15, 17, 33, 65]
        for n in non_checkpoints:
            assert is_checkpoint(n) is False, f"Expected {n} to NOT be a checkpoint"
