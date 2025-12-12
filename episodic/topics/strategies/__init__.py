"""
Topic strategies for pluggable detection and retrieval.

This package contains different strategy implementations that can
be swapped via configuration for experimentation.
"""

from episodic.topics.strategies.dual_window_strategy import DualWindowStrategy
from episodic.topics.strategies.keyword_strategy import KeywordStrategy
from episodic.topics.strategies.relative_embedding_strategy import RelativeEmbeddingStrategy
from episodic.topics.strategies.neural_strategy import NeuralStrategy
from episodic.topics.strategies.ensemble_strategy import EnsembleStrategy
from episodic.topics.strategies.commitment_strategy import (
    CommitmentPolicyStrategy,
    CommitmentPolicy,
    CommitmentState,
)

__all__ = [
    'DualWindowStrategy',
    'KeywordStrategy',
    'RelativeEmbeddingStrategy',
    'NeuralStrategy',
    'EnsembleStrategy',
    'CommitmentPolicyStrategy',
    'CommitmentPolicy',
    'CommitmentState',
]
