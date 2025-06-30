#!/usr/bin/env python3
"""
Test script for topic boundary analysis.

This script tests the new boundary analysis feature that finds
the actual topic transition point when a change is detected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.topic_boundary_analyzer import analyze_topic_boundary, find_transition_point_heuristic

# Test data: A conversation that transitions from cooking to programming
test_nodes = [
    {"id": "n1", "short_id": "A1", "role": "user", "content": "What's a good recipe for pasta?"},
    {"id": "n2", "short_id": "A2", "role": "assistant", "content": "Here's a simple pasta recipe: Boil water, add salt, cook pasta until al dente. For sauce, sauté garlic in olive oil, add tomatoes, basil, and season to taste."},
    {"id": "n3", "short_id": "A3", "role": "user", "content": "How long should I cook it for?"},
    {"id": "n4", "short_id": "A4", "role": "assistant", "content": "Most pasta takes 8-12 minutes. Check the package for specific times. Test a piece - it should be firm but not hard in the center (al dente)."},
    {"id": "n5", "short_id": "A5", "role": "user", "content": "Thanks! By the way, can you help me with Python?"},  # TRANSITION HERE
    {"id": "n6", "short_id": "A6", "role": "assistant", "content": "Of course! I'd be happy to help with Python. What specific aspect are you working on?"},
    {"id": "n7", "short_id": "A7", "role": "user", "content": "I need to sort a list of dictionaries"},
    {"id": "n8", "short_id": "A8", "role": "assistant", "content": "You can use the sorted() function with a key parameter. For example: sorted(list_of_dicts, key=lambda x: x['field_name'])"},
]

def test_heuristic_analysis():
    """Test the heuristic-based boundary finder."""
    print("Testing heuristic boundary analysis...")
    
    # Detection happened at n7
    boundary = find_transition_point_heuristic(test_nodes, "n7")
    
    print(f"Detection point: n7")
    print(f"Found boundary: {boundary}")
    print(f"Expected: n4 (before the transition)")
    
    # The boundary should be n4 (the last message before the transition)
    assert boundary == "n4", f"Expected n4, got {boundary}"
    print("✅ Heuristic test passed!\n")

def test_with_explicit_transition():
    """Test with explicit transition phrases."""
    nodes_explicit = [
        {"id": "e1", "short_id": "E1", "role": "user", "content": "Tell me about machine learning"},
        {"id": "e2", "short_id": "E2", "role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
        {"id": "e3", "short_id": "E3", "role": "user", "content": "What are neural networks?"},
        {"id": "e4", "short_id": "E4", "role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks."},
        {"id": "e5", "short_id": "E5", "role": "user", "content": "Let me ask about a different topic. What's the weather like?"},
        {"id": "e6", "short_id": "E6", "role": "assistant", "content": "I don't have real-time weather data, but I can help you understand weather patterns."},
    ]
    
    print("Testing with explicit transition phrase...")
    boundary = find_transition_point_heuristic(nodes_explicit, "e6")
    print(f"Found boundary: {boundary}")
    print(f"Expected: e4")
    assert boundary == "e4", f"Expected e4, got {boundary}"
    print("✅ Explicit transition test passed!\n")

def test_no_clear_transition():
    """Test when there's no clear transition point."""
    nodes_gradual = [
        {"id": "g1", "short_id": "G1", "role": "user", "content": "How do I cook rice?"},
        {"id": "g2", "short_id": "G2", "role": "assistant", "content": "Use a 2:1 water to rice ratio. Bring to boil, then simmer covered for 18 minutes."},
        {"id": "g3", "short_id": "G3", "role": "user", "content": "What about fried rice?"},
        {"id": "g4", "short_id": "G4", "role": "assistant", "content": "For fried rice, use day-old cooked rice. Heat oil in a wok, add rice and stir-fry."},
        {"id": "g5", "short_id": "G5", "role": "user", "content": "Do you know about rice in Asian cultures?"},
        {"id": "g6", "short_id": "G6", "role": "assistant", "content": "Rice is central to many Asian cultures, both as a staple food and cultural symbol."},
        {"id": "g7", "short_id": "G7", "role": "user", "content": "What about wheat in Western cultures?"},
    ]
    
    print("Testing gradual transition (no clear boundary)...")
    # Detection happens at g7 (user message where topic change detected)
    boundary = find_transition_point_heuristic(nodes_gradual, "g7")
    print(f"Found boundary: {boundary}")
    print(f"Expected: g5 (default: 2 messages back from g7)")
    # Should default to 2 messages before detection: g7 is at index 6, so 6-2=4, which is g5
    expected = "g5"  # 2 messages before g7
    assert boundary == expected, f"Expected {expected}, got {boundary}"
    print("✅ Gradual transition test passed!\n")

def main():
    """Run all tests."""
    print("Topic Boundary Analysis Tests\n" + "="*40 + "\n")
    
    test_heuristic_analysis()
    test_with_explicit_transition() 
    test_no_clear_transition()
    
    print("All tests passed! 🎉")
    
    # Note about LLM-based analysis
    print("\nNote: LLM-based analysis (analyze_topic_boundary) requires")
    print("a running LLM and is not tested here. It would provide more")
    print("accurate boundary detection by understanding conversation context.")

if __name__ == "__main__":
    main()