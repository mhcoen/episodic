#!/usr/bin/env python3
"""
Debug what the actual streaming content looks like
"""

import sys
import os
sys.path.insert(0, '/Users/mhcoen/proj/episodic')

# Import required modules
from episodic.unified_streaming import unified_stream_response
from episodic.config import config

# Create a mock stream that matches the user's output
class MockDelta:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)

class MockChunk:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

def mock_stream_generator():
    # Recreate the exact text the user saw
    text = '''Mystery/Thriller:
- "The Silent Patient" by Alex Michaelides: This psychological thriller revolves around Alicia Berenson'''
    
    # Stream character by character to match real streaming
    for char in text:
        yield MockChunk(char)

print("Testing with user's exact output:")
print("=" * 50)

# Enable debug
config.set("debug", True)
config.set("stream_rate", 0)

try:
    result = unified_stream_response(
        stream_generator=mock_stream_generator(),
        model="test-model"
    )
    print("\n" + "=" * 50)
    print("Debug complete. Check output above.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()