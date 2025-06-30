#!/usr/bin/env python3
"""Test document context inclusion"""

from episodic.documents_poc import DocumentManagerPOC
from episodic.commands.documents import doc_commands

# Test 1: Check what contexts are found
print("=== Testing Document Context ===\n")

doc_manager = DocumentManagerPOC()
success, msg = doc_manager.load_pdf('example.pdf')
print(f"Load result: {msg}\n")

# Test different queries
test_queries = [
    "translate hello into french",
    "what is episodic",
    "how does topic detection work"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    
    # Find contexts
    contexts = doc_manager.find_relevant_context(query, k=3, score_threshold=0.7)
    print(f"  Contexts found (0.7 threshold): {len(contexts)}")
    
    # Test enhancement
    enhanced = doc_manager.enhance_prompt_with_context(query)
    is_enhanced = (enhanced != query)
    print(f"  Prompt enhanced: {is_enhanced}")
    
    if is_enhanced:
        print(f"  Enhanced length: {len(enhanced)} chars (original: {len(query)})")
        # Show a bit of the enhanced prompt
        if "Context from project documentation:" in enhanced:
            print("  Contains documentation context!")

# Test 2: Check doc_commands integration
print("\n\n=== Testing doc_commands ===")
print(f"Context enabled: {doc_commands.context_enabled}")

# Test enhancement through doc_commands
test_msg = "translate hello into french"
enhanced_msg = doc_commands.enhance_message_if_enabled(test_msg)
print(f"\nOriginal: '{test_msg}'")
print(f"Enhanced: {enhanced_msg == test_msg and 'NOT enhanced' or 'WAS enhanced'}")