#!/usr/bin/env python3
"""
Standalone test file for testing unified streaming output formatting.
Reads from test_streaming.txt and prints it using the unified streaming module.
"""

import sys
import os
# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from episodic.unified_streaming import unified_stream_response
from episodic.config import config

# Set up minimal config
config._config = {
    "stream_rate": 0,  # Immediate mode for testing
    "stream_constant_rate": False,
    "stream_natural_rhythm": False,
    "stream_char_mode": False,
    "text_wrap": True,
    "debug": False
}

def test_streaming():

    # Read the test file from the same directory
    test_file = os.path.join(os.path.dirname(__file__), "test_streaming_clean.txt")
    with open(test_file, "r") as f:
        content = f.read()
    
    # Create a mock chunk object that matches LiteLLM's structure
    class MockDelta:
        def __init__(self, content):
            self.content = content
    
    class MockChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)
    
    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    # Create a simple generator that yields the content in chunks
    def content_generator():
        # Simulate streaming by yielding in chunks with proper structure
        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            yield MockChunk(content[i:i+chunk_size])
    
    # Stream the content using unified streaming
    # Use cyan like muse mode does
    # Force wrap_width to None to see what happens
    print(f"[TEST] text_wrap={config.get('text_wrap')}")
    print(f"[TEST] Calling unified_stream_response with wrap_width=None")
    unified_stream_response(
        content_generator(),
        model="test",
        prefix="",
        color="cyan",
        wrap_width=None  # Let it auto-detect
    )

if __name__ == "__main__":
    test_streaming()