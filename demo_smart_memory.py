#!/usr/bin/env python3
"""Demo the smart memory context detection (Milestone 2)"""

import asyncio
from episodic.config import config
from episodic.rag_memory_smart import smart_detector, set_memory_thresholds

def setup_demo():
    """Configure for demo"""
    config.set("enable_memory_rag", True)
    config.set("enable_smart_memory", True)
    config.set("memory_show_indicators", True)
    config.set("debug", False)  # Less noise for demo
    print("✅ Smart memory enabled\n")

def test_implicit_detection():
    """Test implicit reference detection"""
    print("=== Testing Implicit Reference Detection ===\n")
    
    test_cases = [
        # Clear implicit references
        ("And how do I activate it?", "Continuation with 'and'"),
        ("What about on Windows?", "Follow-up with 'what about'"),
        ("Try it with a different approach", "Assumed context with 'it'"),
        ("Make it faster", "Command with unclear antecedent"),
        ("But that didn't work", "Contradiction/clarification"),
        
        # Not implicit
        ("How do I create a Python list?", "Complete new question"),
        ("Explain object-oriented programming", "Self-contained request"),
        
        # Edge cases  
        ("OK, now what?", "Transition phrase"),
        ("Continue", "Explicit continuation"),
        ("More examples", "Request for more"),
        ("Actually, never mind", "Correction/cancellation"),
    ]
    
    for query, description in test_cases:
        is_implicit, confidence = smart_detector.detect_implicit_reference(query)
        status = "✓" if is_implicit else "✗"
        print(f"{status} '{query}'")
        print(f"   {description}")
        print(f"   Implicit: {is_implicit}, Confidence: {confidence:.2f}\n")

def test_smart_injection():
    """Test the full smart injection logic"""
    print("\n=== Testing Smart Context Injection ===\n")
    
    # Simulate conversation states
    test_scenarios = [
        {
            'query': "And how do I install dependencies?",
            'state': {
                'current_topic_name': 'python-virtualenv',
                'messages_since_topic_change': 5,
                'total_messages': 20
            },
            'description': "Implicit continuation in ongoing topic"
        },
        {
            'query': "What's the command for that?",
            'state': {
                'current_topic_name': 'git-basics',
                'messages_since_topic_change': 2,
                'total_messages': 10
            },
            'description': "Vague reference after topic change"
        },
        {
            'query': "pip install requests",
            'state': {
                'current_topic_name': 'python-packages',
                'messages_since_topic_change': 8,
                'total_messages': 30
            },
            'description': "Topic-specific keyword trigger"
        },
        {
            'query': "How do I create a function?",
            'state': {
                'current_topic_name': None,
                'messages_since_topic_change': 0,
                'total_messages': 0
            },
            'description': "New conversation, no context"
        }
    ]
    
    for scenario in test_scenarios:
        should_inject, confidence, reason = smart_detector.should_inject_context(
            scenario['query'], 
            scenario['state']
        )
        
        print(f"Query: '{scenario['query']}'")
        print(f"Context: {scenario['description']}")
        print(f"Topic: {scenario['state']['current_topic_name'] or 'None'}")
        print(f"Decision: {'INJECT' if should_inject else 'NO INJECT'}")
        print(f"Confidence: {confidence:.2f}, Reason: {reason}\n")

def test_threshold_tuning():
    """Demo threshold configuration"""
    print("\n=== Threshold Configuration ===\n")
    
    print("Default thresholds:")
    print("  Explicit: 70%, Implicit: 50%, Relevance: 70%")
    
    print("\nSetting aggressive thresholds...")
    set_memory_thresholds(explicit=0.5, implicit=0.3, relevance=0.6)
    
    print("\nSetting conservative thresholds...")
    set_memory_thresholds(explicit=0.8, implicit=0.6, relevance=0.8)

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Smart Memory Context - Milestone 2 Demo             ║
║                                                              ║
║  Intelligent context injection without explicit references    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    setup_demo()
    test_implicit_detection()
    test_smart_injection()
    test_threshold_tuning()
    
    print("""
✅ Smart Memory Features:

1. **Implicit Reference Detection**
   - Detects continuations: "And...", "But...", "Also..."
   - Catches vague references: "it", "that", "the command"
   - Identifies follow-ups: "What about...", "How about..."

2. **Contextual Intelligence**
   - Adjusts confidence based on conversation length
   - Considers topic changes and boundaries
   - Uses topic-specific keywords

3. **Visual Indicators**
   - 🧠 Strong connection (80%+ confidence)
   - 💭 Moderate connection (60-79%)
   - 💡 Weak connection (below 60%)

4. **Configurable Thresholds**
   - Tune sensitivity for your workflow
   - Balance precision vs recall
   - Per-deployment customization
""")

if __name__ == "__main__":
    main()