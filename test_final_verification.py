#!/usr/bin/env python3
"""Final verification of hybrid topic detection system."""

import sys
sys.path.insert(0, '.')

from episodic.topics_hybrid import detect_topic_change_hybrid
from episodic.config import config

# Configure optimal settings
config.set("use_hybrid_topic_detection", True)
config.set("debug", False)  # Disable debug for cleaner output
config.set("hybrid_topic_weights", {
    "semantic_drift": 0.6,
    "keyword_explicit": 0.25,
    "keyword_domain": 0.1,
    "message_gap": 0.025,
    "conversation_flow": 0.025
})
config.set("hybrid_topic_threshold", 0.55)
config.set("hybrid_llm_threshold", 0.3)

print("Hybrid Topic Detection System - Final Verification")
print("="*60)

test_cases = [
    # (name, messages, new_message, expected_result)
    ("Continuing same topic", 
     [{"role": "user", "content": "How do I learn Python?"}, 
      {"role": "assistant", "content": "Start with basics..."}],
     "What IDE should I use?", 
     False),
    
    ("Natural topic drift",
     [{"role": "user", "content": "Tell me about Mars"}, 
      {"role": "assistant", "content": "Mars is the fourth planet..."},
      {"role": "user", "content": "How far is it from Earth?"},
      {"role": "assistant", "content": "The distance varies..."}],
     "What's a good recipe for lasagna?",
     True),
    
    ("Explicit transition",
     [{"role": "user", "content": "Explain recursion"}, 
      {"role": "assistant", "content": "Recursion is..."}],
     "Let's change topics. How's the weather?",
     True),
    
    ("Subtle transition",
     [{"role": "user", "content": "What's the capital of France?"}, 
      {"role": "assistant", "content": "Paris"}],
     "Speaking of cities, what's the best pizza in New York?",
     True),
    
    ("Related but different",
     [{"role": "user", "content": "How do I train a neural network?"}, 
      {"role": "assistant", "content": "Training involves..."},
      {"role": "user", "content": "What about the loss function?"},
      {"role": "assistant", "content": "Loss functions measure..."}],
     "Can you recommend a good statistics textbook?",
     False),  # Still in the ML/math domain
]

correct = 0
total = len(test_cases)

for name, messages, new_msg, expected in test_cases:
    changed, _, metadata = detect_topic_change_hybrid(messages, new_msg)
    
    status = "✓" if changed == expected else "✗"
    correct += (changed == expected)
    
    print(f"\n{status} {name}")
    print(f"   New message: '{new_msg[:50]}{'...' if len(new_msg) > 50 else ''}'")
    print(f"   Expected: {expected}, Got: {changed}")
    print(f"   Score: {metadata.get('score', 0):.2f}, Method: {metadata.get('method', 'hybrid')}")
    if changed != expected:
        print(f"   Explanation: {metadata.get('explanation', 'N/A')}")

print(f"\n{'='*60}")
print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("\nConfiguration:")
print(f"  Threshold: {config.get('hybrid_topic_threshold')}")
print(f"  LLM Fallback: {config.get('hybrid_llm_threshold')}")
print(f"  Weights: {config.get('hybrid_topic_weights')}")

# Test that it integrates with conversation.py
print("\nIntegration check:")
try:
    from episodic.conversation import ConversationManager
    print("✓ Can import ConversationManager")
    
    # Check config
    if config.get("use_hybrid_topic_detection"):
        print("✓ Hybrid detection is enabled in config")
    else:
        print("✗ Hybrid detection is NOT enabled in config")
        
except Exception as e:
    print(f"✗ Integration issue: {e}")

print("\nVerification complete!")