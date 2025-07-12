#!/usr/bin/env python3
"""Test bold formatting in streaming"""

from episodic.response_streaming import ResponseStreamer, StreamConfig, StreamingMode
from episodic.config import config

# Enable debug
config.set("debug_streaming_verbose", True)

# Create a mock stream generator
def mock_stream():
    chunks = [
        "Peru's winter season, which typically runs from June to August, varies significantly\n",
        "depending on the region due to its diverse geography. Here's a general overview:\n\n",
        "1. **Coastal Areas (including Lima):** \n",
        "- The coast experiences a mild winter with cooler temperatures ranging from about\n",
        "12°C to 18°C (54°F to 64°F).\n"
    ]
    for chunk in chunks:
        yield chunk

# Test streaming
streamer = ResponseStreamer()
config_obj = StreamConfig(
    mode=StreamingMode.IMMEDIATE,
    words_per_second=15.0,
    enable_wrapping=True,
    enable_bold=True
)

print("Testing bold formatting in streaming:")
print("=" * 50)

# Process the mock stream
full_response = ""
for chunk in mock_stream():
    streamer._process_chunk(chunk, [], config_obj, OutputHandler())
    full_response += chunk

print("\n\nFull response:")
print(full_response)