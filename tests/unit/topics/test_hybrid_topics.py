#!/usr/bin/env python3
"""
Test script for hybrid topic detection system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.topics_hybrid import HybridTopicDetector, TopicChangeSignals
from episodic.config import config

# Enable hybrid detection
config.set("use_hybrid_topic_detection", True)
config.set("debug", True)

# Set improved weights and thresholds
config.set("hybrid_topic_weights", {
    "semantic_drift": 0.6,
    "keyword_explicit": 0.25,
    "keyword_domain": 0.1,
    "message_gap": 0.025,
    "conversation_flow": 0.025
})
config.set("hybrid_topic_threshold", 0.55)
config.set("hybrid_llm_threshold", 0.3)


def test_semantic_drift():
    """Test semantic drift detection."""
    print("\n=== Testing Semantic Drift Detection ===\n")
    
    detector = HybridTopicDetector()
    
    # Test messages with clear topic change
    messages = [
        {"role": "user", "content": "How do I make pasta?"},
        {"role": "assistant", "content": "To make pasta..."},
        {"role": "user", "content": "What sauce goes well with it?"},
        {"role": "assistant", "content": "For pasta sauces..."},
    ]
    
    # This should NOT trigger topic change (same topic)
    new_msg1 = "How long should I cook it?"
    result1 = detector.detect_topic_change(messages, new_msg1)
    print(f"Test 1 - Same topic (cooking): {result1[0]} - {result1[2].get('explanation', '')}")
    
    # This SHOULD trigger topic change (different topic)
    new_msg2 = "Can you explain quantum computing?"
    result2 = detector.detect_topic_change(messages, new_msg2)
    print(f"Test 2 - Topic change (cooking→quantum): {result2[0]} - {result2[2].get('explanation', '')}")


def test_keyword_detection():
    """Test keyword-based transition detection."""
    print("\n=== Testing Keyword Detection ===\n")
    
    detector = HybridTopicDetector()
    
    messages = [
        {"role": "user", "content": "Tell me about Python programming"},
        {"role": "assistant", "content": "Python is..."},
    ]
    
    # Test explicit transition
    new_msg1 = "Let's change topics. What's the weather like?"
    result1 = detector.detect_topic_change(messages, new_msg1)
    print(f"Test 1 - Explicit transition: {result1[0]} - {result1[2].get('explanation', '')}")
    
    # Test domain shift
    messages2 = [
        {"role": "user", "content": "How do I debug this code?"},
        {"role": "assistant", "content": "To debug..."},
        {"role": "user", "content": "What about unit tests?"},
        {"role": "assistant", "content": "Unit tests..."},
    ]
    
    new_msg2 = "What's a good recipe for chocolate cake?"
    result2 = detector.detect_topic_change(messages2, new_msg2)
    print(f"Test 2 - Domain shift (tech→cooking): {result2[0]} - {result2[2].get('explanation', '')}")


def test_signal_combination():
    """Test how different signals combine."""
    print("\n=== Testing Signal Combination ===\n")
    
    from episodic.topics_hybrid import HybridScorer
    
    scorer = HybridScorer()
    
    # Test various signal combinations
    test_cases = [
        ("Low all signals", TopicChangeSignals(0.2, 0.1, 0.1, 0.0, 0.0)),
        ("High semantic only", TopicChangeSignals(0.8, 0.0, 0.0, 0.0, 0.0)),
        ("High keyword only", TopicChangeSignals(0.0, 0.9, 0.0, 0.0, 0.0)),
        ("Medium mixed", TopicChangeSignals(0.5, 0.0, 0.5, 0.1, 0.1)),
        ("High all signals", TopicChangeSignals(0.8, 0.8, 0.7, 0.2, 0.2)),
    ]
    
    for name, signals in test_cases:
        score, explanation = scorer.calculate_topic_change_score(signals)
        decision = "CHANGE" if score >= scorer.topic_change_threshold else "SAME"
        print(f"{name}: Score={score:.2f} Decision={decision} - {explanation}")


def test_real_conversation():
    """Test with a realistic conversation flow."""
    print("\n=== Testing Real Conversation Flow ===\n")
    
    detector = HybridTopicDetector()
    
    # Simulate a conversation that gradually drifts
    conversation = [
        ("How do I get started with machine learning?", False),
        ("What Python libraries should I use?", False),
        ("Can you explain neural networks?", False),
        ("How does backpropagation work?", False),
        ("Speaking of calculations, what's the best calculator app?", True),  # Soft transition
        ("I need something for my phone", False),
        ("What about other productivity apps?", False),
        ("By the way, how's the weather tomorrow?", True),  # Explicit transition
    ]
    
    messages = []
    for i, (msg, should_change) in enumerate(conversation):
        if i > 0:  # Need history
            result = detector.detect_topic_change(messages, msg)
            status = "✓" if result[0] == should_change else "✗"
            print(f"{status} Message {i}: '{msg[:50]}...' - Detected: {result[0]}, Expected: {should_change}")
            
        # Add to history
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": f"Response to: {msg}"})


if __name__ == "__main__":
    print("Hybrid Topic Detection Test Suite")
    print("=" * 50)
    
    test_semantic_drift()
    test_keyword_detection()
    test_signal_combination()
    test_real_conversation()
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    
    # Show current configuration
    print("\nCurrent Configuration:")
    print(f"  Hybrid detection enabled: {config.get('use_hybrid_topic_detection')}")
    print(f"  Topic change threshold: {config.get('hybrid_topic_threshold')}")
    print(f"  LLM fallback threshold: {config.get('hybrid_llm_threshold')}")
    print(f"  Weights: {config.get('hybrid_topic_weights')}")