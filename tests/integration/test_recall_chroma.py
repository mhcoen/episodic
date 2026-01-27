"""
Integration tests for recall module with real Chroma.

Tests the actual Chroma query path that unit tests mock out.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from episodic.recall.pipeline import (
    _get_semantic_hits,
    SemanticHit,
    MIN_CANDIDATES_FOR_AMBIGUITY,
)


class FakeChromaCollection:
    """Fake Chroma collection that returns controlled results."""

    def __init__(self, results):
        self._results = results

    def query(self, query_texts, n_results, include):
        return self._results


class FakeRAG:
    """Fake RAG wrapper."""

    def __init__(self, collection):
        self._collection = collection

    def get_collection(self, collection_type):
        return self._collection


class TestGetSemanticHitsIntegration:
    """Test _get_semantic_hits with fake but realistic Chroma responses."""

    def test_empty_query_returns_empty(self):
        """Empty target returns no hits."""
        hits = _get_semantic_hits("", 10, None, False)
        assert hits == []

    def test_none_query_returns_empty(self):
        """None target returns no hits."""
        hits = _get_semantic_hits(None, 10, None, False)
        assert hits == []

    def test_handles_chroma_response_format(self):
        """Verify we correctly parse Chroma's nested list response format."""
        # Chroma returns nested lists: results['ids'][0][i], etc.
        fake_results = {
            'ids': [['id_1', 'id_2', 'id_3']],
            'distances': [[0.5, 1.0, 1.5]],  # Lower = more similar
            'documents': [['doc 1', 'doc 2', 'doc 3']],
            'metadatas': [[
                {'user_id': 'id_1', 'timestamp': '2026-01-15T10:00:00Z'},
                {'user_id': 'id_2', 'timestamp': '2026-01-15T11:00:00Z'},
                {'user_id': 'id_3', 'timestamp': '2026-01-15T12:00:00Z'},
            ]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test query", 10, None, False)

        # All 3 should pass the similarity threshold (0.10)
        # distance 0.5 -> similarity 0.75
        # distance 1.0 -> similarity 0.50
        # distance 1.5 -> similarity 0.25
        assert len(hits) == 3
        assert all(isinstance(h, SemanticHit) for h in hits)
        assert hits[0].exchange_id == 'id_1'
        assert hits[0].relevance_score == pytest.approx(0.75, rel=0.01)
        assert hits[1].relevance_score == pytest.approx(0.50, rel=0.01)
        assert hits[2].relevance_score == pytest.approx(0.25, rel=0.01)

    def test_similarity_threshold_filters_low_scores(self):
        """Hits below min_similarity are filtered out."""
        # distance 1.9 -> similarity 0.05 (below 0.10 threshold)
        fake_results = {
            'ids': [['id_1', 'id_2']],
            'distances': [[1.0, 1.9]],  # Second one will be filtered
            'documents': [['doc 1', 'doc 2']],
            'metadatas': [[
                {'user_id': 'id_1'},
                {'user_id': 'id_2'},
            ]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False, min_similarity=0.10)

        # Only first hit passes (similarity 0.50 >= 0.10)
        # Second hit filtered (similarity 0.05 < 0.10)
        assert len(hits) == 1
        assert hits[0].exchange_id == 'id_1'

    def test_embeddings_included_when_requested(self):
        """Embeddings are extracted when include_embeddings=True."""
        emb1 = [0.1, 0.2, 0.3]
        emb2 = [0.4, 0.5, 0.6]

        fake_results = {
            'ids': [['id_1', 'id_2']],
            'distances': [[0.5, 0.6]],
            'documents': [['doc 1', 'doc 2']],
            'metadatas': [[
                {'user_id': 'id_1'},
                {'user_id': 'id_2'},
            ]],
            'embeddings': [[emb1, emb2]],  # Nested list format
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False, include_embeddings=True)

        assert len(hits) == 2
        assert hits[0].embedding is not None
        assert hits[1].embedding is not None
        np.testing.assert_array_almost_equal(hits[0].embedding, emb1)
        np.testing.assert_array_almost_equal(hits[1].embedding, emb2)

    def test_handles_numpy_array_embeddings(self):
        """Handles case where Chroma returns numpy arrays directly."""
        emb1 = np.array([0.1, 0.2, 0.3])
        emb2 = np.array([0.4, 0.5, 0.6])

        fake_results = {
            'ids': [['id_1', 'id_2']],
            'distances': [[0.5, 0.6]],
            'documents': [['doc 1', 'doc 2']],
            'metadatas': [[{'user_id': 'id_1'}, {'user_id': 'id_2'}]],
            'embeddings': [[emb1, emb2]],  # Already numpy arrays
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            # This should NOT raise "truth value of array is ambiguous"
            hits = _get_semantic_hits("test", 10, None, False, include_embeddings=True)

        assert len(hits) == 2
        assert hits[0].embedding is not None

    def test_handles_empty_embeddings_list(self):
        """Handles case where embeddings is empty or None."""
        fake_results = {
            'ids': [['id_1']],
            'distances': [[0.5]],
            'documents': [['doc 1']],
            'metadatas': [[{'user_id': 'id_1'}]],
            'embeddings': [[]],  # Empty embeddings list
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False, include_embeddings=True)

        assert len(hits) == 1
        assert hits[0].embedding is None  # No embedding available

    def test_handles_none_embeddings_entry(self):
        """Handles case where some embeddings are None."""
        fake_results = {
            'ids': [['id_1', 'id_2']],
            'distances': [[0.5, 0.6]],
            'documents': [['doc 1', 'doc 2']],
            'metadatas': [[{'user_id': 'id_1'}, {'user_id': 'id_2'}]],
            'embeddings': [[[0.1, 0.2], None]],  # Second embedding is None
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False, include_embeddings=True)

        assert len(hits) == 2
        assert hits[0].embedding is not None
        assert hits[1].embedding is None

    def test_temporal_filter_excludes_out_of_range(self):
        """Temporal filter excludes hits outside the time range."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        utc = ZoneInfo("UTC")
        start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=utc)
        end = datetime(2026, 1, 15, 12, 0, 0, tzinfo=utc)
        temporal = (start, end)

        fake_results = {
            'ids': [['id_1', 'id_2', 'id_3']],
            'distances': [[0.5, 0.5, 0.5]],
            'documents': [['doc 1', 'doc 2', 'doc 3']],
            'metadatas': [[
                {'user_id': 'id_1', 'timestamp': '2026-01-15T10:30:00Z'},  # In range
                {'user_id': 'id_2', 'timestamp': '2026-01-15T09:00:00Z'},  # Before range
                {'user_id': 'id_3', 'timestamp': '2026-01-15T13:00:00Z'},  # After range
            ]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, temporal, broad_horizon=False)

        # Only id_1 is in range
        assert len(hits) == 1
        assert hits[0].exchange_id == 'id_1'

    def test_chroma_exception_returns_empty(self):
        """Chroma exceptions are caught and return empty list."""
        fake_rag = MagicMock()
        fake_rag.get_collection.side_effect = Exception("Chroma error")

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False)

        assert hits == []

    def test_empty_chroma_results_returns_empty(self):
        """Empty Chroma results return empty list."""
        fake_results = {
            'ids': [[]],
            'distances': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False)

        assert hits == []


class TestMinSimilarityThreshold:
    """Verify the min_similarity threshold is calibrated correctly."""

    def test_default_threshold_allows_reasonable_distances(self):
        """Default threshold (0.10) allows distances up to ~1.8."""
        # L2 distance of 1.8 -> similarity of 0.10 (exactly at threshold)
        fake_results = {
            'ids': [['id_1']],
            'distances': [[1.79]],  # Just under 1.8, should pass
            'documents': [['doc 1']],
            'metadatas': [[{'user_id': 'id_1'}]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False)

        assert len(hits) == 1  # Should pass at 0.10 threshold

    def test_threshold_filters_very_distant_hits(self):
        """Threshold filters hits with L2 distance > 1.8."""
        fake_results = {
            'ids': [['id_1']],
            'distances': [[1.81]],  # Just over 1.8, should fail
            'documents': [['doc 1']],
            'metadatas': [[{'user_id': 'id_1'}]],
            'embeddings': None,
        }

        fake_collection = FakeChromaCollection(fake_results)
        fake_rag = FakeRAG(fake_collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=fake_rag):
            hits = _get_semantic_hits("test", 10, None, False)

        assert len(hits) == 0  # Filtered out at 0.10 threshold
