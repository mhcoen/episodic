"""
Integration tests for recall CLI behavior.

Tests the full recall pipeline with mocked Chroma responses.
Verifies that the similarity threshold correctly filters results.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from episodic.recall.pipeline import _get_semantic_hits, SemanticHit


class FakeChromaCollection:
    """Fake Chroma collection for controlled testing."""

    def __init__(self, query_responses):
        """
        Args:
            query_responses: dict mapping query text to response dict
        """
        self._responses = query_responses

    def query(self, query_texts, n_results, include):
        query = query_texts[0].lower()
        # Find best matching response
        for key, response in self._responses.items():
            if key.lower() in query:
                return response
        # Return empty response for unknown queries
        return {
            'ids': [[]],
            'distances': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'embeddings': None,
        }


class FakeRAG:
    """Fake RAG wrapper."""

    def __init__(self, collection):
        self._collection = collection

    def get_collection(self, collection_type):
        return self._collection


# Test data: simulates realistic Chroma responses
# Distances are L2 distances - lower = more similar
# Similarity = 1.0 - (distance / 2.0)
MOCK_RESPONSES = {
    # Coffee query - HAS relevant content (distances ~0.7-0.9 -> sims 0.55-0.65)
    'coffee': {
        'ids': [['coffee_1', 'coffee_2', 'coffee_3']],
        'distances': [[0.72, 0.85, 0.95]],  # sims: 0.64, 0.575, 0.525
        'documents': [[
            'User: Anything about coffee in our past chats?\nAssistant: We discussed coffee brewing...',
            'User: What kind of coffee do you recommend?\nAssistant: I like medium roast...',
            'User: Tell me about espresso\nAssistant: Espresso is a concentrated coffee...',
        ]],
        'metadatas': [[
            {'user_id': 'coffee_1', 'timestamp': '2026-01-15T10:00:00Z'},
            {'user_id': 'coffee_2', 'timestamp': '2026-01-15T11:00:00Z'},
            {'user_id': 'coffee_3', 'timestamp': '2026-01-15T12:00:00Z'},
        ]],
        'embeddings': None,
    },
    # Java query - NO relevant content (distances ~1.4-1.6 -> sims 0.20-0.30)
    'java': {
        'ids': [['unrelated_1', 'unrelated_2', 'unrelated_3']],
        'distances': [[1.36, 1.45, 1.55]],  # sims: 0.32, 0.275, 0.225 - all below 0.35
        'documents': [[
            'User: computer.\nAssistant: It sounds like you are interested in computers...',
            'User: Hello\nAssistant: Hi there! How can I help you today?',
            'User: What is in @README.md?\nAssistant: The README contains...',
        ]],
        'metadatas': [[
            {'user_id': 'unrelated_1', 'timestamp': '2026-01-15T10:00:00Z'},
            {'user_id': 'unrelated_2', 'timestamp': '2026-01-15T11:00:00Z'},
            {'user_id': 'unrelated_3', 'timestamp': '2026-01-15T12:00:00Z'},
        ]],
        'embeddings': None,
    },
    # Weather query - HAS relevant content (distances ~0.9-1.2 -> sims 0.40-0.55)
    'weather': {
        'ids': [['weather_1', 'weather_2']],
        'distances': [[0.90, 1.10]],  # sims: 0.55, 0.45
        'documents': [[
            'User: Do you know the weather?\nAssistant: I cannot check real-time weather...',
            'User: What is the weather like today?\nAssistant: I do not have access...',
        ]],
        'metadatas': [[
            {'user_id': 'weather_1', 'timestamp': '2026-01-16T10:00:00Z'},
            {'user_id': 'weather_2', 'timestamp': '2026-01-16T11:00:00Z'},
        ]],
        'embeddings': None,
    },
}


@pytest.fixture
def mock_rag():
    """Create mock RAG with test responses."""
    collection = FakeChromaCollection(MOCK_RESPONSES)
    return FakeRAG(collection)


class TestSimilarityThresholdFiltering:
    """Verify the min_similarity threshold (0.35) correctly filters results."""

    def test_high_similarity_matches_pass_threshold(self, mock_rag):
        """Queries with relevant content (sim >= 0.35) return results."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("coffee brewing", 10, None, False)

        # Coffee has sims 0.64, 0.575, 0.525 - all above 0.35
        assert len(hits) == 3
        assert all(h.relevance_score >= 0.35 for h in hits)
        assert 'coffee' in hits[0].text.lower()

    def test_low_similarity_matches_filtered_out(self, mock_rag):
        """Queries with no relevant content (sim < 0.35) return empty."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("java programming", 10, None, False)

        # Java has sims 0.32, 0.275, 0.225 - all below 0.35
        assert len(hits) == 0

    def test_mixed_similarity_filters_correctly(self, mock_rag):
        """Verify filtering works when some matches pass and some fail."""
        # Add a response with mixed similarities
        mixed_response = {
            'ids': [['good_1', 'bad_1', 'good_2', 'bad_2']],
            'distances': [[0.8, 1.5, 1.0, 1.8]],  # sims: 0.60, 0.25, 0.50, 0.10
            'documents': [[
                'Good match 1', 'Bad match 1', 'Good match 2', 'Bad match 2',
            ]],
            'metadatas': [[
                {'user_id': 'good_1'},
                {'user_id': 'bad_1'},
                {'user_id': 'good_2'},
                {'user_id': 'bad_2'},
            ]],
            'embeddings': None,
        }

        collection = FakeChromaCollection({'mixed': mixed_response})
        rag = FakeRAG(collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=rag):
            hits = _get_semantic_hits("mixed test", 10, None, False)

        # Only 2 hits should pass (sims 0.60 and 0.50)
        assert len(hits) == 2
        assert hits[0].exchange_id == 'good_1'
        assert hits[1].exchange_id == 'good_2'
        assert all(h.relevance_score >= 0.35 for h in hits)

    def test_threshold_boundary_at_035(self, mock_rag):
        """Verify the exact 0.35 threshold boundary."""
        # Create responses at exactly 0.35 and just below
        boundary_response = {
            'ids': [['at_threshold', 'below_threshold']],
            'distances': [[1.30, 1.31]],  # sims: 0.35 (exactly), 0.345 (just below)
            'documents': [['At threshold', 'Below threshold']],
            'metadatas': [[{'user_id': 'at_threshold'}, {'user_id': 'below_threshold'}]],
            'embeddings': None,
        }

        collection = FakeChromaCollection({'boundary': boundary_response})
        rag = FakeRAG(collection)

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=rag):
            hits = _get_semantic_hits("boundary test", 10, None, False)

        # Only the one at exactly 0.35 should pass
        assert len(hits) == 1
        assert hits[0].exchange_id == 'at_threshold'


class TestSemanticHitConstruction:
    """Verify SemanticHit objects are correctly constructed from Chroma responses."""

    def test_hit_contains_correct_fields(self, mock_rag):
        """SemanticHit has all required fields populated."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("weather", 10, None, False)

        assert len(hits) >= 1
        hit = hits[0]

        assert isinstance(hit, SemanticHit)
        assert hit.exchange_id == 'weather_1'
        assert 0.0 <= hit.relevance_score <= 1.0
        assert isinstance(hit.metadata, dict)
        assert 'weather' in hit.text.lower()

    def test_similarity_score_calculation(self, mock_rag):
        """Verify L2 distance to similarity conversion is correct."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("weather", 10, None, False)

        # Weather has distances [0.90, 1.10] -> sims [0.55, 0.45]
        assert len(hits) == 2
        assert abs(hits[0].relevance_score - 0.55) < 0.01
        assert abs(hits[1].relevance_score - 0.45) < 0.01


class TestEmptyAndEdgeCases:
    """Test handling of empty and edge case inputs."""

    def test_empty_query_returns_empty(self):
        """Empty target returns no hits without calling Chroma."""
        hits = _get_semantic_hits("", 10, None, False)
        assert hits == []

    def test_none_query_returns_empty(self):
        """None target returns no hits without calling Chroma."""
        hits = _get_semantic_hits(None, 10, None, False)
        assert hits == []

    def test_whitespace_query_returns_empty(self):
        """Whitespace-only target returns no hits."""
        hits = _get_semantic_hits("   ", 10, None, False)
        assert hits == []

    def test_chroma_exception_returns_empty(self):
        """Chroma exceptions are caught and return empty list."""
        mock_rag = MagicMock()
        mock_rag.get_collection.side_effect = Exception("Connection failed")

        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("test", 10, None, False)

        assert hits == []


class TestNoGarbageResults:
    """Verify queries don't return irrelevant garbage."""

    def test_java_query_returns_nothing_not_garbage(self, mock_rag):
        """Query for 'java' (no content) returns empty, not unrelated results."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("java", 10, None, False)

        # Should return empty, not garbage about computers/hello/README
        assert len(hits) == 0

    def test_nonexistent_topic_returns_empty(self, mock_rag):
        """Query for topic that doesn't exist returns empty."""
        with patch('episodic.recall.pipeline.get_multi_collection_rag', return_value=mock_rag):
            hits = _get_semantic_hits("quantum entanglement physics", 10, None, False)

        # No quantum physics content - should be empty
        assert len(hits) == 0
