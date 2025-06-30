#\!/usr/bin/env python3
"""Verify that windows don't overlap."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import get_manual_index_scores

scores = get_manual_index_scores()

print("Verifying window boundaries (no overlap should exist):")
print("-" * 80)
print(f"{'Node':^6} | {'Window A':^20} | {'Window B':^20} | {'Overlap?':^10}")
print("-" * 80)

for score in scores[:10]:  # Show first 10
    # Check if Window A end equals Window B start
    overlap = "YES\!" if score['window_a_end_short_id'] == score['user_node_short_id'] else "No"
    
    print(f"{score['user_node_short_id']:^6} | [{score['window_a_start_short_id']}] → [{score['window_a_end_short_id']}] | [{score['window_b_start_short_id']}] → [{score['window_b_end_short_id']}] | {overlap:^10}")

print("-" * 80)
print("\nExplanation:")
print("- user_node_short_id should be the first node of Window B")
print("- Window A should end just before Window B starts")
print("- There should be NO overlap between windows")
