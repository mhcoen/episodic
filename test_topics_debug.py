#!/usr/bin/env python3
"""Debug topics command."""

import sys
sys.path.insert(0, '.')

# Test direct call
print("Testing direct topics function call...")
print("=" * 60)

try:
    from episodic.commands.topics import topics
    from episodic.db import get_recent_topics
    
    # First check if we have any topics
    topic_list = get_recent_topics(limit=10)
    print(f"Found {len(topic_list)} topics in database")
    
    # Call the function directly
    print("\nCalling topics() function:")
    topics(limit=10, all=False, verbose=False)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("If you see topics above, the function works but CLI routing may be broken.")
print("If no topics shown, check benchmark_operation or display_pending_benchmark.")