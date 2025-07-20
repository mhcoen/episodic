#!/usr/bin/env python3
"""Enable smart memory context detection (Milestone 2)"""

from episodic.config import config
from episodic.rag_memory_smart import set_memory_thresholds

def main():
    print("=== Enabling Smart Memory Context ===\n")
    
    # Enable memory systems
    config.set("enable_memory_rag", True)
    config.set("enable_smart_memory", True)
    config.set("memory_show_indicators", True)
    print("✓ Smart memory enabled")
    
    # Set default thresholds
    set_memory_thresholds(
        explicit=0.7,   # 70% confidence for explicit references
        implicit=0.5,   # 50% confidence for implicit references
        relevance=0.7   # 70% relevance score threshold
    )
    
    print("\n✅ Smart Memory is now active!")
    print("\nFeatures enabled:")
    print("- Automatic detection of implicit references")
    print("- Context injection without explicit 'remember when'")
    print("- Visual indicators showing memory usage")
    print("- Intelligent confidence scoring")
    
    print("\nExamples that will trigger smart memory:")
    print('- "And how do I..." (continuation)')
    print('- "What about..." (follow-up)')
    print('- "Make it faster" (unclear antecedent)')
    print('- "Try that" (vague reference)')
    print('- "But wait..." (clarification)')
    
    print("\nTo disable: config.set('enable_smart_memory', False)")
    print("To hide indicators: config.set('memory_show_indicators', False)")

if __name__ == "__main__":
    main()