#!/usr/bin/env python3
"""Test different streaming modes"""

from episodic.response_streaming import ResponseStreamer
from episodic.config import config

# Test text with bold markers
test_text = """Peru's winter season varies significantly:

1. **Coastal Areas (including Lima):** 
- The coast experiences mild winter.

2. **Andean Highlands:**
- Winter is dry and sunny."""

# Test 1: Immediate mode
print("=== TEST 1: IMMEDIATE MODE ===")
config.set('stream_natural_rhythm', False)
config.set('stream_constant_rate', False)

def mock_stream():
    for line in test_text.split('\n'):
        yield line + '\n'

streamer = ResponseStreamer()
result = streamer.stream_response(
    stream_generator=mock_stream(),
    model="test",
    stream_rate=0,
    use_constant_rate=False,
    use_natural_rhythm=False
)

print("\n=== TEST 2: NATURAL RHYTHM MODE ===")
# This is what's actually being used
streamer2 = ResponseStreamer()
result2 = streamer2.stream_response(
    stream_generator=mock_stream(),
    model="test",
    stream_rate=15.0,
    use_constant_rate=False,
    use_natural_rhythm=True
)

print("\nDone.")