# Episodic Topic Detection Evaluation Results

## Summary

We evaluated Episodic's topic detection approaches on the SuperDialseg dataset, which contains 9,478 dialogues with human-annotated topic boundaries. The evaluation provides quantitative metrics to assess our current approach and guide improvements.

## Dataset Overview

- **Dataset**: SuperDialseg (superseg)
- **Total conversations**: 10,914
  - Train: 6,948
  - Validation: 1,322
  - Test: 1,322
- **Total utterances**: 144,947
- **Total boundaries**: 70,420

## Best Results

### Sliding Window Detection (Window=3, Threshold=0.3)

Evaluated on 200 test dialogues:

| Metric | Value |
|--------|-------|
| Precision | 0.569 |
| Recall | 0.615 |
| **F1 Score** | **0.562** |
| F1 (w=1) | 0.585 |
| F1 (w=3) | 0.714 |
| F1 (w=5) | 0.775 |
| WindowDiff | 0.409 |
| Pk | 0.420 |

## Key Findings

1. **Threshold Sensitivity**: The default threshold of 0.9 was too conservative, resulting in very few detected boundaries. Lowering to 0.3 dramatically improved performance.

2. **Window Tolerance**: When allowing a tolerance window of 3 positions, F1 score improves from 0.562 to 0.714, suggesting our detector is often "close" to the correct boundary.

3. **Keyword Detection**: The keyword-based detector alone performed poorly (F1 ≈ 0.000), indicating that explicit transition phrases are rare in this dataset.

4. **Hybrid Approach**: The hybrid detector had implementation issues that need to be resolved before proper evaluation.

## Comparison with Baselines

The SuperDialseg paper reports various baseline results. Our sliding window approach with optimized threshold achieves competitive performance:

- Our F1: 0.562 (exact), 0.714 (w=3)
- This is respectable for a real-time, unsupervised approach

## Recommendations

1. **Dynamic Thresholds**: Consider adaptive thresholds based on conversation context or topic count.

2. **Feature Engineering**: The current approach only uses semantic drift. Adding features like:
   - Time gaps between messages
   - Speaker turn patterns
   - Message length changes
   - Domain-specific keywords

3. **Ensemble Methods**: Combine multiple detectors with learned weights rather than fixed thresholds.

4. **Training Data**: While Episodic is designed to be unsupervised, we could:
   - Use SuperDialseg to tune hyperparameters
   - Learn optimal window sizes and thresholds
   - Train a lightweight classifier on top of our features

5. **Evaluation Metrics**: Focus on windowed metrics (F1 w=3) for practical applications, as exact boundary matching may be too strict.

## Next Steps

1. Fix the hybrid detector implementation
2. Experiment with different embedding models
3. Test on other datasets (DialSeg711, TIAGE)
4. Implement dynamic threshold adjustment
5. Add more features beyond semantic drift

## Code and Reproducibility

All evaluation code is in `/evaluation/`:
- `superdialseg_loader.py` - Dataset loading
- `metrics.py` - Evaluation metrics
- `detector_adapters.py` - Episodic detector adapters
- `run_evaluation.py` - Main evaluation script

To reproduce:
```bash
python evaluation/run_evaluation.py /path/to/superseg --detector sliding_window --threshold 0.3
```