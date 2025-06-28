#!/usr/bin/env python3
"""Debug why mars-rover is being extracted for wrong topics."""

import re

# Read the test output
with open('test_three_topics.txt', 'r') as f:
    content = f.read()

# Find all topic name extractions
pattern = r"Extracting name for previous topic '([^']+)'.*?Topic has (\d+) nodes.*?Extracted topic name: ([^\n]+)"
matches = re.findall(pattern, content, re.DOTALL)

print("Topic name extractions:")
for prev_name, node_count, extracted_name in matches:
    print(f"  Previous: '{prev_name}' ({node_count} nodes) → Extracted: '{extracted_name}'")