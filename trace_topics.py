#!/usr/bin/env python3
"""Trace topic extensions to find the bug."""

import re

with open('test_three_topics.txt', 'r') as f:
    lines = f.readlines()

current_topic = None
topic_changes = []
extensions = []

for i, line in enumerate(lines):
    # Track current topic changes
    if "Current topic set to" in line:
        match = re.search(r"Current topic set to '([^']+)'", line)
        if match:
            current_topic = match.group(1)
            topic_changes.append((i, current_topic))
    
    # Track extensions
    if "Extended topic" in line:
        match = re.search(r"Extended topic '([^']+)'", line)
        if match:
            topic = match.group(1)
            extensions.append((i, topic, current_topic))
    
    # Track when topics are closed
    if "Updated topic name:" in line or "Queued topic" in line:
        match = re.search(r"'([^']+)'", line)
        if match:
            topic = match.group(1)
            if "mars-rover" in topic:
                print(f"Line {i}: mars-rover activity: {line.strip()}")

print("\nTopic changes:")
for line_no, topic in topic_changes:
    print(f"  Line {line_no}: Changed to '{topic}'")

print("\nExtensions of mars-rover:")
for line_no, extended_topic, current in extensions:
    if extended_topic == "mars-rover":
        print(f"  Line {line_no}: Extended mars-rover (current was: {current})")