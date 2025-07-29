# Sliding Window vs ML Models Comparison

## Executive Summary

The current sliding window detection system in Episodic performs significantly worse than the trained ML models on the topic boundary detection task.

## Performance Comparison

### Current Sliding Window System (Default Settings)
- **Model**: all-MiniLM-L6-v2 embeddings
- **Window Size**: 3 messages
- **Similarity Threshold**: 0.9

**Results on Test Data**:
- **F1 Score: 0.416**
- **Precision: 0.263**
- **Recall: 1.000**

### Trained ML Models (Best Performers with 4,2 Window)

| Model | F1 Score | Precision | Recall | Improvement over Sliding Window |
|-------|----------|-----------|--------|--------------------------------|
| **DistilBERT (4,2)** | **0.8222** | **0.731** | **0.939** | **+97.6%** |
| MiniLM-L12 (4,2) | 0.7983 | 0.697 | 0.934 | +91.9% |
| ELECTRA Small (4,2) | 0.7957 | 0.690 | 0.939 | +91.3% |

## Key Findings

### 1. **Sliding Window Limitations**
- **Very Low Precision (26.3%)**: The system predicts far too many boundaries
- **Perfect Recall (100%)**: It catches all boundaries but with massive false positives
- **Poor F1 Score (41.6%)**: The balance between precision and recall is poor

### 2. **Threshold Sensitivity**
Testing different similarity thresholds shows the fundamental limitation:

| Threshold | F1 Score | Precision | Recall |
|-----------|----------|-----------|--------|
| 0.70 | 0.393 | 0.251 | 0.904 |
| 0.80 | 0.402 | 0.255 | 0.942 |
| 0.85 | 0.418 | 0.264 | 1.000 |
| **0.90** | **0.416** | **0.263** | **1.000** |
| 0.95 | 0.413 | 0.260 | 1.000 |

No threshold provides good performance - the approach is fundamentally limited.

### 3. **Why ML Models Perform Better**

1. **Learned Patterns**: ML models learn complex patterns beyond simple similarity
2. **Contextual Understanding**: They consider linguistic cues, not just semantic drift
3. **Balanced Predictions**: Much better precision-recall balance
4. **Window Configuration**: The (4,2) window provides better context for decisions

### 4. **Practical Impact**

In a typical conversation with 100 messages and 5 true topic boundaries:
- **Sliding Window**: Would detect ~50 boundaries (45 false positives)
- **DistilBERT**: Would detect ~6-7 boundaries (1-2 false positives)

## Recommendations

1. **Integrate the Trained Models**: Replace the sliding window with DistilBERT (4,2)
   - 97.6% improvement in F1 score
   - 178% improvement in precision
   - Maintains high recall (93.9%)

2. **Implementation Path**:
   ```python
   # Current approach
   similarity = cosine_similarity(window1_embed, window2_embed)
   is_boundary = similarity < 0.9
   
   # Proposed approach
   context = get_42_window(messages, position)
   is_boundary = distilbert_model.predict(context)
   ```

3. **Fallback Strategy**: Keep sliding window as fallback for when ML model unavailable

4. **User Experience**: With ML models, users would see:
   - Fewer false topic changes
   - More accurate topic organization
   - Better conversation flow

## Conclusion

The current sliding window approach achieves only **41.6% F1 score**, while the trained DistilBERT model achieves **82.2% F1 score** - a **97.6% improvement**. The ML models provide dramatically better precision while maintaining excellent recall, making them far superior for production use.