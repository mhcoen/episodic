#\!/usr/bin/env python3
"""Check manual index scores in the database."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import get_manual_index_scores

# Get all scores
scores = get_manual_index_scores()

print(f"Total scores stored: {len(scores)}")
print("\nManual Index Scores (indexed by Window B start node):")
print("-" * 100)
print(f"{'Node ID':^8} | {'Content Preview':^40} | {'Drift':^6} | {'Keyword':^7} | {'Combined':^8} | {'Boundary':^8}")
print("-" * 100)

for score in scores:
    preview = score['content'][:37] + '...' if len(score['content']) > 40 else score['content']
    boundary = "YES" if score['is_boundary'] else "no"
    
    print(f"[{score['user_node_short_id']:^6}] | {preview:40} | {score['drift_score']:6.3f} | {score['keyword_score']:7.3f} | {score['combined_score']:8.3f} | {boundary:^8}")
    
    if score['transition_phrase']:
        print(f"         | → Transition: '{score['transition_phrase']}'")

print("-" * 100)
print(f"\nThreshold used: {scores[0]['threshold_used'] if scores else 'N/A'}")
