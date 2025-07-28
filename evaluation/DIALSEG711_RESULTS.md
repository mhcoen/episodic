# DialSeg711 Evaluation Results

## Dataset Comparison

We evaluated Episodic's sliding window detector on both SuperDialseg and DialSeg711 datasets to understand performance across different dialogue types.

## DialSeg711 Overview

- **Total conversations**: 711
- **Total utterances**: 19,350
- **Total boundaries**: 2,754
- **Average boundaries per conversation**: ~3.9

## Results Summary

### DialSeg711 Performance (100 dialogues tested)

| Threshold | Precision | Recall | F1    | F1(w=3) | WindowDiff | Pk    |
|-----------|-----------|--------|-------|---------|------------|-------|
| 0.3       | 0.297     | 0.826  | 0.430 | 0.511   | 0.607      | 0.558 |
| 0.5       | 0.313     | 0.789  | 0.442 | 0.554   | 0.550      | 0.520 |
| **0.7**   | **0.326** | **0.468** | **0.372** | **0.649** | **0.431** | **0.438** |

### Comparison with SuperDialseg

| Dataset     | Best Threshold | F1    | F1(w=3) | WindowDiff |
|-------------|----------------|-------|---------|------------|
| SuperDialseg | 0.3           | 0.562 | 0.714   | 0.409      |
| DialSeg711  | 0.7           | 0.372 | 0.649   | 0.431      |

## Key Findings

1. **Different Optimal Thresholds**: 
   - SuperDialseg performs best with threshold 0.3
   - DialSeg711 performs best with threshold 0.7
   - This suggests different conversation characteristics between datasets

2. **DialSeg711 Characteristics**:
   - Lower thresholds lead to over-segmentation (high recall, low precision)
   - The dialogues appear to have more subtle topic transitions
   - Requires higher confidence (0.7) to avoid false positives

3. **Performance Differences**:
   - SuperDialseg: Better exact F1 (0.562 vs 0.372)
   - Similar WindowDiff scores (~0.41-0.43)
   - Both achieve good windowed F1 scores (0.65-0.71)

## Analysis

The performance difference suggests that:

1. **Domain Differences**: DialSeg711 may contain more domain-specific dialogues where topic transitions are less semantically distinct.

2. **Annotation Differences**: The datasets may have different annotation guidelines for what constitutes a topic boundary.

3. **Conversation Types**: 
   - SuperDialseg appears to have clearer topic boundaries
   - DialSeg711 may have more gradual topic shifts

## Recommendations

1. **Adaptive Thresholds**: Implement dataset or domain-specific threshold selection.

2. **Feature Analysis**: Investigate what makes DialSeg711 boundaries harder to detect:
   - Analyze boundary examples
   - Compare semantic drift distributions
   - Look for dataset-specific patterns

3. **Ensemble Approach**: Different datasets might benefit from different detection strategies.

4. **Calibration**: Consider threshold calibration based on initial conversation analysis.

## Conclusion

While our sliding window approach works reasonably well on both datasets, the significant threshold difference (0.3 vs 0.7) highlights the importance of:
- Dataset-specific tuning
- Understanding domain characteristics
- Adaptive detection strategies

The windowed F1 scores (0.649-0.714) suggest our detector is often "close" to correct boundaries, making it suitable for practical applications where exact boundary precision is less critical than general topic organization.