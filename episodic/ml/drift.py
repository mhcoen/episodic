"""
Conversational drift detection for DAG-based conversations.

This module provides functionality to measure semantic drift between
conversation nodes using embeddings and distance functions.
"""

from typing import List, Dict, Any, Optional, Tuple
from .embeddings import EmbeddingProvider
from .distance.functions import DistanceFunction


class ConversationalDrift:
    """
    Measures semantic drift between conversation nodes.
    
    Combines text embeddings with distance/similarity functions to quantify
    how much the conversation topic has shifted between nodes.
    """
    
    def __init__(
        self,
        embedding_provider: str = "sentence-transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        distance_algorithm: str = "cosine"
    ):
        """
        Initialize conversational drift calculator.
        
        Args:
            embedding_provider: Backend for generating embeddings
            embedding_model: Model name for the embedding provider
            distance_algorithm: Algorithm for measuring distance/similarity
        """
        self.embedding_provider = EmbeddingProvider(
            provider=embedding_provider,
            model=embedding_model
        )
        self.distance_function = DistanceFunction(algorithm=distance_algorithm)
        
        # Cache embeddings to avoid recomputation
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def calculate_drift(
        self,
        node1: Dict[str, Any],
        node2: Dict[str, Any],
        text_field: str = "message"
    ) -> float:
        """
        Calculate semantic drift between two conversation nodes.
        
        Args:
            node1: First conversation node (dict with text content)
            node2: Second conversation node (dict with text content)
            text_field: Field name containing the text to analyze
            
        Returns:
            Drift score - interpretation depends on distance algorithm:
            - For similarity measures (cosine, dot_product): 0.0 = identical, 1.0 = completely different
            - For distance measures (euclidean, manhattan): 0.0 = identical, higher = more different
            
        Raises:
            ValueError: If nodes are missing the specified text field
            RuntimeError: If embedding generation fails
        """
        # Extract text content from nodes
        text1 = node1.get(text_field)
        text2 = node2.get(text_field)
        
        if text1 is None or text2 is None:
            raise ValueError(f"Both nodes must have '{text_field}' field")
        
        if not isinstance(text1, str) or not isinstance(text2, str):
            raise ValueError(f"Field '{text_field}' must contain string values")
        
        # Generate embeddings (with caching)
        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)
        
        # Calculate raw distance/similarity
        raw_score = self.distance_function.calculate(embedding1, embedding2)
        
        # Normalize to drift score (0.0 = no drift, 1.0 = maximum drift)
        if self.distance_function.is_similarity():
            # For similarity: high similarity = low drift
            return 1.0 - max(0.0, min(1.0, raw_score))
        else:
            # For distance: normalize to 0-1 range
            # This is a simple normalization - could be improved with empirical bounds
            return min(1.0, raw_score / 2.0)
    
    def calculate_drift_sequence(
        self,
        nodes: List[Dict[str, Any]],
        text_field: str = "message"
    ) -> List[float]:
        """
        Calculate drift scores for a sequence of conversation nodes.
        
        Returns drift between each consecutive pair of nodes.
        
        Args:
            nodes: List of conversation nodes in chronological order
            text_field: Field name containing the text to analyze
            
        Returns:
            List of drift scores (length = len(nodes) - 1)
            
        Raises:
            ValueError: If fewer than 2 nodes provided
        """
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes to calculate drift sequence")
        
        drift_scores = []
        for i in range(len(nodes) - 1):
            drift = self.calculate_drift(nodes[i], nodes[i + 1], text_field)
            drift_scores.append(drift)
        
        return drift_scores
    
    def calculate_cumulative_drift(
        self,
        nodes: List[Dict[str, Any]],
        text_field: str = "message"
    ) -> List[float]:
        """
        Calculate cumulative drift from the first node to each subsequent node.
        
        Args:
            nodes: List of conversation nodes in chronological order
            text_field: Field name containing the text to analyze
            
        Returns:
            List of cumulative drift scores (length = len(nodes) - 1)
            First node is the reference point (drift = 0.0 implicitly)
        """
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes to calculate cumulative drift")
        
        reference_node = nodes[0]
        cumulative_drift = []
        
        for i in range(1, len(nodes)):
            drift = self.calculate_drift(reference_node, nodes[i], text_field)
            cumulative_drift.append(drift)
        
        return cumulative_drift
    
    def find_drift_peaks(
        self,
        nodes: List[Dict[str, Any]],
        threshold: float = 0.5,
        text_field: str = "message"
    ) -> List[Tuple[int, float]]:
        """
        Identify conversation points where drift exceeds a threshold.
        
        Args:
            nodes: List of conversation nodes in chronological order
            threshold: Minimum drift score to consider a "peak"
            text_field: Field name containing the text to analyze
            
        Returns:
            List of (node_index, drift_score) tuples for peaks
        """
        drift_scores = self.calculate_drift_sequence(nodes, text_field)
        
        peaks = []
        for i, drift in enumerate(drift_scores):
            if drift >= threshold:
                # i+1 because drift_scores[i] represents drift between nodes[i] and nodes[i+1]
                peaks.append((i + 1, drift))
        
        return peaks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_provider.embed(text)
        return self._embedding_cache[text]
    
    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._embedding_cache)