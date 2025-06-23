"""
Machine Learning components for conversational drift detection.

This package contains modular ML components:
- distance: Distance/similarity functions
- embeddings: Text embedding providers
- summarization: Branch summarization strategies
- drift: Conversational drift detection
"""

from .drift import ConversationalDrift
from .embeddings import EmbeddingProvider
from .distance.functions import DistanceFunction
from .peaks import PeakDetector

__all__ = ['ConversationalDrift', 'EmbeddingProvider', 'DistanceFunction', 'PeakDetector']