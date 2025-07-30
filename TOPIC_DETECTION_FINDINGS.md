# Topic Detection Evaluation Findings

## Executive Summary

After extensive evaluation, cosine similarity with a (4,2) window configuration outperforms all other approaches, achieving 93.88% F1 score for topic boundary detection.

## Key Results

### Approach Comparison

1. **Current Implementation (User-to-User)**: 
   - F1: 73.8%
   - Too many false positives (41.5% FPR)
   - Only compares consecutive user messages

2. **Best Configuration ((4,2) Window)**:
   - F1: 93.88%
   - Compares last 4 messages with next 2 messages
   - Includes assistant responses for better context

3. **High Precision Option ((4,1) Window)**:
   - F1: 86.4%
   - 95% precision, 79.2% recall
   - Good for high-confidence boundary detection

### Why Cosine Similarity Works Better Than Fine-tuning

- Pre-trained sentence embeddings (all-mpnet-base-v2) already capture semantic relationships
- Fine-tuned models achieved only 71.4% F1 vs 91.67% for cosine similarity
- No distance metric outperformed cosine similarity in our tests

## Recommended Implementation

### Two-Tier System
1. **Primary Detection**: Use (4,1) window for memory updates
   - High precision (95%) ensures few false positives
   - Triggers actual context/memory changes

2. **Safety Net**: Use (4,2) window for monitoring
   - Catches boundaries the primary might miss
   - Could trigger lighter-weight responses

### Optimal Thresholds
- (4,2) window: 0.249 (similarity below this = boundary)
- (4,1) window: 0.137 (more conservative)

## Implementation Notes

- Current user-to-user comparison in `conversation.py` could be improved
- Assistant responses provide valuable context for boundary detection
- Window size matters: more context generally improves performance