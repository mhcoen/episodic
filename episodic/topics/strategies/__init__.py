"""
Topic strategies for pluggable detection and retrieval.

This package contains different strategy implementations that can
be swapped via configuration for experimentation.
"""

from episodic.topics.strategies.dual_window_strategy import DualWindowStrategy

__all__ = ['DualWindowStrategy']
