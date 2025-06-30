#\!/usr/bin/env python3
"""Compare window sizes 2 and 3 for topic detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.config import config
config.set("automatic_topic_detection", False)

from episodic.commands.index_topics import index_topics
from episodic.db import get_manual_index_scores, clear_manual_index_scores

# Expected topic boundaries
expected_topics = [
    (0, 5, "Vacation planning"),
    (5, 11, "Learning programming"),  
    (11, 17, "Home cooking"),
    (17, 23, "Fitness and exercise"),
    (23, 29, "Personal finance"),
    (29, 35, "Plant care"),
    (35, 41, "Technology and gadgets")
]

def analyze_window_size(window_size, threshold=0.9):
    """Run analysis for a specific window size."""
    # Clear previous scores
    clear_manual_index_scores()
    
    # Run detection
    print(f"\n{'='*80}")
    print(f"Testing window size {window_size}")
    print(f"{'='*80}")
    
    index_topics(window_size=window_size, apply=False, verbose=False)
    
    # Get scores
    scores = get_manual_index_scores(window_size=window_size)
    
    # Analyze results
    print(f"\nDrift scores at topic boundaries (threshold={threshold}):")
    print(f"{'Msg#':^6} | {'ID':^6} | {'Drift':^6} | {'Expected Boundary':^20} | {'Detected':^10}")
    print("-" * 60)
    
    detected_boundaries = []
    false_positives = []
    
    for i, score in enumerate(scores):
        # Check if this is an expected boundary
        is_expected_boundary = False
        boundary_name = ""
        for start, end, topic in expected_topics:
            if i + 2 == start and start > 0:  # i+2 because scores are 0-indexed and we skip first message
                is_expected_boundary = True
                boundary_name = f"→ {topic}"
                break
        
        is_detected = score['drift_score'] >= threshold
        if is_detected:
            if is_expected_boundary:
                detected_boundaries.append((i+2, score['drift_score']))
            else:
                false_positives.append((i+2, score['drift_score']))
        
        if is_expected_boundary or is_detected:
            marker = "✓" if (is_expected_boundary and is_detected) else "✗" if is_expected_boundary else "FP"
            print(f"{i+2:^6} | {score['user_node_short_id']:^6} | {score['drift_score']:6.3f} | {boundary_name:^20} | {marker:^10}")
    
    # Summary
    print(f"\nSummary for window size {window_size}:")
    print(f"- Expected boundaries: {len(expected_topics) - 1}")
    print(f"- Detected boundaries: {len(detected_boundaries)} {[f'(msg {m}, drift {d:.3f})' for m, d in detected_boundaries]}")
    print(f"- False positives: {len(false_positives)} {[f'(msg {m}, drift {d:.3f})' for m, d in false_positives]}")
    print(f"- Detection rate: {len(detected_boundaries)}/{len(expected_topics)-1} = {len(detected_boundaries)/(len(expected_topics)-1)*100:.0f}%")
    
    return scores

# Test both window sizes
print("Comparing Window Sizes for Topic Detection")
print("=" * 80)

# Window size 2
scores_2 = analyze_window_size(2, threshold=0.9)

# Window size 3  
scores_3 = analyze_window_size(3, threshold=0.9)

# Direct comparison at boundaries
print("\n" + "="*80)
print("Direct Comparison at Expected Topic Boundaries")
print("="*80)
print(f"{'Topic Transition':^30} | {'Window 2':^12} | {'Window 3':^12} | {'Better':^8}")
print("-" * 70)

boundary_positions = [5, 11, 17, 23, 29, 35]  # Message positions where topics change
boundary_names = [
    "Vacation → Programming",
    "Programming → Cooking", 
    "Cooking → Fitness",
    "Fitness → Finance",
    "Finance → Plants",
    "Plants → Technology"
]

for i, (pos, name) in enumerate(zip(boundary_positions, boundary_names)):
    # Find scores for this position (remember offset)
    score_2 = next((s['drift_score'] for s in scores_2 if s == scores_2[pos-2]), None) if pos-2 < len(scores_2) else None
    score_3 = next((s['drift_score'] for s in scores_3 if s == scores_3[pos-2]), None) if pos-2 < len(scores_3) else None
    
    if score_2 and score_3:
        better = "W2" if score_2 > score_3 else "W3" if score_3 > score_2 else "="
        print(f"{name:^30} | {score_2:^12.3f} | {score_3:^12.3f} | {better:^8}")

# Test different thresholds
print("\n" + "="*80)
print("Performance at Different Thresholds")
print("="*80)
print(f"{'Threshold':^10} | {'Window 2 Detection':^20} | {'Window 3 Detection':^20}")
print("-" * 55)

for threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    # Count detections for each window size
    w2_detected = sum(1 for i, s in enumerate(scores_2) 
                     if s['drift_score'] >= threshold 
                     and any(i+2 == start for start, _, _ in expected_topics[1:]))
    w3_detected = sum(1 for i, s in enumerate(scores_3)
                     if s['drift_score'] >= threshold
                     and any(i+2 == start for start, _, _ in expected_topics[1:]))
    
    print(f"{threshold:^10.2f} | {w2_detected}/6 ({w2_detected/6*100:>3.0f}%){' '*10} | {w3_detected}/6 ({w3_detected/6*100:>3.0f}%){' '*10}")

