"""
Distance and similarity functions for semantic embeddings.

This module provides various algorithms for measuring semantic distance
between embedding vectors.
"""

import math
from typing import List


class DistanceFunction:
    """Configurable distance/similarity function supporting multiple algorithms."""
    
    def __init__(self, algorithm: str = "cosine"):
        """
        Initialize distance function.
        
        Args:
            algorithm: Distance algorithm ("cosine", "euclidean", "dot_product", "manhattan")
        """
        self.algorithm = algorithm
        self._validate_algorithm()
    
    def _validate_algorithm(self):
        """Validate that the algorithm is supported."""
        supported = {"cosine", "euclidean", "dot_product", "manhattan"}
        if self.algorithm not in supported:
            raise ValueError(f"Unknown distance algorithm: {self.algorithm}. Supported: {supported}")
    
    def calculate(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate distance/similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            For cosine: similarity score (0.0 to 1.0, higher = more similar)
            For euclidean: distance (lower = more similar)
            For dot_product: similarity score (higher = more similar)
            For manhattan: distance (lower = more similar)
            
        Raises:
            ValueError: If embedding dimensions don't match
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(f"Embedding dimensions don't match: {len(embedding1)} vs {len(embedding2)}")
        
        if self.algorithm == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif self.algorithm == "euclidean":
            return self._euclidean_distance(embedding1, embedding2)
        elif self.algorithm == "dot_product":
            return self._dot_product(embedding1, embedding2)
        elif self.algorithm == "manhattan":
            return self._manhattan_distance(embedding1, embedding2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0  # No similarity if either vector is zero
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product between two vectors."""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def _manhattan_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Manhattan (L1) distance between two vectors."""
        return sum(abs(a - b) for a, b in zip(vec1, vec2))
    
    def is_similarity(self) -> bool:
        """
        Return True if the algorithm returns similarity (higher = more similar),
        False if it returns distance (lower = more similar).
        """
        return self.algorithm in {"cosine", "dot_product"}
    
    def is_distance(self) -> bool:
        """
        Return True if the algorithm returns distance (lower = more similar),
        False if it returns similarity (higher = more similar).
        """
        return not self.is_similarity()


# Convenience functions for direct use
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return DistanceFunction("cosine").calculate(vec1, vec2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return DistanceFunction("euclidean").calculate(vec1, vec2)