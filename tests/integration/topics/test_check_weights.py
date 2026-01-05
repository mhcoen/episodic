#!/usr/bin/env python3
"""Check actual weights being used."""

if __name__ != "__main__":
    import pytest

    pytest.skip("Debug script; run directly", allow_module_level=True)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.topics_hybrid import HybridTopicDetector
from episodic.config import config

detector = HybridTopicDetector()

print("Weights in detector:", detector.scorer.weights)
print("Weights in config:", config.get("hybrid_topic_weights"))
print("Topic threshold:", detector.scorer.topic_change_threshold)
print("LLM threshold:", detector.scorer.llm_fallback_threshold)
