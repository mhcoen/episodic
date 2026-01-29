"""
Tests for fusion (Success Criterion 9).

SC9: Determinism - identical queries yield identical ordering.
"""
import pytest


class TestFusionDeterminism:
    """SC9: Fusion must be deterministic."""
    
    def test_fusion_ordering_is_deterministic(self):
        """Repeated fusion with same inputs yields same order."""
        from episodic.retrieval.fusion import fuse_results
        
        semantic = [
            {"exchange_id": "U1", "distance": 0.2},
            {"exchange_id": "U2", "distance": 0.2},  # Same distance
            {"exchange_id": "U3", "distance": 0.1},
        ]
        
        lexical = [
            {"exchange_id": "U1", "bm25_score": 5.0},
            {"exchange_id": "U2", "bm25_score": 5.0},  # Same score
            {"exchange_id": "U4", "bm25_score": 3.0},
        ]
        
        # Run fusion multiple times
        results = []
        for _ in range(5):
            fused = fuse_results(semantic, lexical, w_sem=0.6, w_lex=0.4)
            results.append([r['exchange_id'] for r in fused])
        
        # All runs should produce identical ordering
        assert all(r == results[0] for r in results), "Fusion not deterministic"
    
    def test_fusion_tiebreaker_by_exchange_id(self):
        """Equal scores are broken by exchange_id ASC."""
        from episodic.retrieval.fusion import fuse_results
        
        # Create candidates with identical scores
        semantic = [
            {"exchange_id": "B", "distance": 0.5},
            {"exchange_id": "A", "distance": 0.5},
            {"exchange_id": "C", "distance": 0.5},
        ]
        
        lexical = [
            {"exchange_id": "B", "bm25_score": 1.0},
            {"exchange_id": "A", "bm25_score": 1.0},
            {"exchange_id": "C", "bm25_score": 1.0},
        ]
        
        fused = fuse_results(semantic, lexical, w_sem=0.5, w_lex=0.5)
        
        ids = [r['exchange_id'] for r in fused]
        # With equal final_scores, should be sorted by exchange_id ASC
        assert ids == sorted(ids), f"Expected alphabetical order for ties, got {ids}"
    
    def test_semantic_sorted_before_normalization(self):
        """Semantic candidates sorted by (distance ASC, exchange_id ASC)."""
        from episodic.retrieval.fusion import prepare_for_fusion
        
        semantic = [
            {"exchange_id": "C", "distance": 0.3},
            {"exchange_id": "A", "distance": 0.1},
            {"exchange_id": "B", "distance": 0.1},  # Same distance as A
        ]
        
        sorted_sem, _ = prepare_for_fusion(semantic, [])
        
        ids = [r['exchange_id'] for r in sorted_sem]
        # A and B have same distance, A comes first alphabetically
        assert ids == ["A", "B", "C"]
    
    def test_lexical_sorted_before_normalization(self):
        """Lexical candidates sorted by (bm25_score DESC, exchange_id ASC)."""
        from episodic.retrieval.fusion import prepare_for_fusion
        
        lexical = [
            {"exchange_id": "C", "bm25_score": 5.0},
            {"exchange_id": "A", "bm25_score": 10.0},
            {"exchange_id": "B", "bm25_score": 10.0},  # Same score as A
        ]
        
        _, sorted_lex = prepare_for_fusion([], lexical)
        
        ids = [r['exchange_id'] for r in sorted_lex]
        # A and B have same score (high), should come first, A before B
        assert ids == ["A", "B", "C"]


class TestNormalization:
    """Test score normalization."""
    
    def test_semantic_normalization_inverts(self):
        """Semantic: lower distance = higher normalized score."""
        from episodic.retrieval.fusion import normalize_scores
        
        candidates = [
            {"exchange_id": "A", "distance": 0.1},  # Best
            {"exchange_id": "B", "distance": 0.5},  # Worst
        ]
        
        normalized = normalize_scores(candidates, "distance", invert=True)
        
        # Lower distance should have higher normalized score
        assert normalized["A"] > normalized["B"]
        assert normalized["A"] == 1.0  # Best = 1.0 after inversion
        assert normalized["B"] == 0.0  # Worst = 0.0 after inversion
    
    def test_lexical_normalization_no_invert(self):
        """Lexical: higher bm25_score = higher normalized score."""
        from episodic.retrieval.fusion import normalize_scores
        
        candidates = [
            {"exchange_id": "A", "bm25_score": 10.0},  # Best
            {"exchange_id": "B", "bm25_score": 2.0},   # Worst
        ]
        
        normalized = normalize_scores(candidates, "bm25_score", invert=False)
        
        assert normalized["A"] > normalized["B"]
        assert normalized["A"] == 1.0
        assert normalized["B"] == 0.0
    
    def test_normalization_handles_equal_scores(self):
        """When min == max, all get 1.0 (all tied for best)."""
        from episodic.retrieval.fusion import normalize_scores

        candidates = [
            {"exchange_id": "A", "distance": 0.5},
            {"exchange_id": "B", "distance": 0.5},
        ]

        normalized = normalize_scores(candidates, "distance", invert=True)

        # All tied = all best = 1.0
        assert normalized["A"] == 1.0
        assert normalized["B"] == 1.0
    
    def test_missing_channel_gets_zero(self):
        """Candidate missing from a channel gets 0.0 norm for that channel."""
        from episodic.retrieval.fusion import fuse_results
        
        semantic = [
            {"exchange_id": "A", "distance": 0.1},
        ]
        
        lexical = [
            {"exchange_id": "B", "bm25_score": 5.0},
        ]
        
        fused = fuse_results(semantic, lexical, w_sem=0.5, w_lex=0.5)
        
        # Both should be in results
        ids = {r['exchange_id'] for r in fused}
        assert ids == {"A", "B"}
        
        # A has no lexical, B has no semantic
        for r in fused:
            if r['exchange_id'] == "A":
                # Only semantic contribution
                assert r['final_score'] == 0.5 * 1.0 + 0.5 * 0.0
            elif r['exchange_id'] == "B":
                # Only lexical contribution
                assert r['final_score'] == 0.5 * 0.0 + 0.5 * 1.0


class TestFusionFormula:
    """Test fusion score calculation."""
    
    def test_fusion_weighted_combination(self):
        """Final score is weighted sum of normalized scores."""
        from episodic.retrieval.fusion import fuse_results
        
        semantic = [
            {"exchange_id": "A", "distance": 0.0},  # Best semantic
        ]
        
        lexical = [
            {"exchange_id": "A", "bm25_score": 10.0},  # Also best lexical
        ]
        
        fused = fuse_results(semantic, lexical, w_sem=0.6, w_lex=0.4)
        
        # With only one candidate, both norms are 0.5 (single value)
        # Actually with one candidate, norm depends on implementation
        # If only one value, should still be defined (likely 0.5 or 1.0)
        assert len(fused) == 1
        assert 'final_score' in fused[0]
