#\!/usr/bin/env python3
"""Run realistic conversation test and analyze drift scores."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# First, initialize fresh database
from episodic.db import initialize_db
db_path = os.path.expanduser("~/.episodic/episodic.db")
if os.path.exists(db_path):
    os.remove(db_path)
initialize_db(migrate=True)

# Disable automatic topic detection
from episodic.config import config
config.set("automatic_topic_detection", False)

# Run the script
from episodic.cli import execute_script
print("Running realistic conversation script...")
execute_script("scripts/realistic-conversation-test.txt")

# Now run manual index detection
print("\n" + "="*80)
print("Running manual topic detection with window size 3...")
from episodic.commands.index_topics import index_topics
index_topics(window_size=3, apply=False, verbose=False)

# Check the scores
print("\n" + "="*80)
print("Analyzing drift scores at topic boundaries...")
from episodic.db import get_manual_index_scores

scores = get_manual_index_scores(window_size=3)

# Group by expected topics
expected_topics = [
    (0, 5, "Vacation planning"),
    (5, 11, "Learning programming"),
    (11, 17, "Home cooking"),
    (17, 23, "Fitness and exercise"),
    (23, 29, "Personal finance"),
    (29, 35, "Plant care"),
    (35, 41, "Technology and gadgets")
]

print(f"\n{'Msg#':^6} | {'Short ID':^8} | {'Drift':^6} | {'Topic Boundary?':^15} | {'Expected Topic':^25}")
print("-" * 75)

for i, score in enumerate(scores):
    # Find which topic this should be in
    expected_topic = "Unknown"
    is_boundary = False
    for start, end, topic in expected_topics:
        if i + 1 == start and start > 0:  # i+1 because we're 0-indexed but checking 1-indexed boundaries
            is_boundary = True
            expected_topic = topic
            break
        elif start <= i + 1 < end:
            expected_topic = topic
            break
    
    boundary_marker = "→ YES" if is_boundary else ""
    drift_marker = "***" if score['drift_score'] > 0.9 else ""
    
    print(f"{i+2:^6} | {score['user_node_short_id']:^8} | {score['drift_score']:6.3f} | {boundary_marker:^15} | {expected_topic:25} {drift_marker}")

print("-" * 75)
print("\nSummary:")
print(f"Messages with drift > 0.9: {sum(1 for s in scores if s['drift_score'] > 0.9)}")
print(f"Expected topic boundaries: {len(expected_topics) - 1}")
