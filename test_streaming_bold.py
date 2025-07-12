#!/usr/bin/env python3
"""Test streaming bold functionality directly."""

# Initialize config first
from episodic.config import config
config.initialize()

from episodic.unified_streaming import unified_stream_response
import time

def mock_stream_generator():
    """Mock a stream response with bold markers."""
    response_text = "Here are some book recommendations:\n\n**Mystery/Thriller:**\n- *The Silent Patient* by Alex Michaelides\n- **Gone Girl** by Gillian Flynn\n\n**Science Fiction:**\n- The Martian by Andy Weir"
    
    # Simulate streaming by yielding character by character
    for char in response_text:
        yield char
        time.sleep(0.01)  # Small delay to simulate real streaming

def test_streaming_bold():
    """Test the streaming output with bold formatting."""
    print("Testing streaming bold formatting...")
    print("=" * 50)
    
    # Test the unified streaming function
    unified_stream_response(
        stream_generator=mock_stream_generator(),
        model="test-model",
        prefix="",
        preserve_formatting=True
    )
    
    print("\n" + "=" * 50)
    print("Test complete. Check if headers (Mystery/Thriller:, Science Fiction:) and book titles are bold.")

if __name__ == "__main__":
    test_streaming_bold()