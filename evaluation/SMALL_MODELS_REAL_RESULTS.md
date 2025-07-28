# Small Instruct Models - Real Dataset Evaluation

## Executive Summary

We evaluated the smallest available instruct models on real dialogue segmentation datasets (SuperDialseg and TIAGE). The results show that while these models are much smaller than traditional LLMs, their performance on real data is lower than on simple test cases.

## Results Summary

| Model | Size | SuperDialseg F1 | TIAGE F1 | Avg Speed | Notes |
|-------|------|-----------------|----------|-----------|--------|
| **qwen2:0.5b** | 352 MB | 0.250 | **0.333** | 1.41s | Fastest, best on TIAGE |
| tinyllama:latest | 637 MB | 0.183 | 0.255 | 6.24s | Slowest, lowest accuracy |
| qwen2:1.5b | 934 MB | 0.244 | 0.312 | 1.91s | Balanced performance |
| **Sentence-BERT** | ~400 MB | **0.571** | 0.222 | 0.01s | Previous best |

## Key Findings

### 1. Size vs Performance Trade-off
- Qwen2:0.5b (352 MB) achieves surprisingly good results given its tiny size
- It actually **outperforms Sentence-BERT on TIAGE** (0.333 vs 0.222)
- However, it underperforms on SuperDialseg (0.250 vs 0.571)

### 2. Speed Characteristics
- Qwen2 models are much faster than TinyLlama
- Qwen2:0.5b: ~1.4s per dialogue
- TinyLlama: ~6.2s per dialogue (4.4x slower)
- Still much slower than embedding methods (~0.01s)

### 3. Model-Specific Observations

#### Qwen2:0.5b (352 MB)
- **Pros**: Smallest size, fastest LLM, best on TIAGE
- **Cons**: Lower accuracy on SuperDialseg
- **Best for**: Extreme resource constraints, TIAGE-like datasets

#### TinyLlama (637 MB)
- **Pros**: Better on simple test cases
- **Cons**: Slow and lower accuracy on real data
- **Not recommended** for production use

#### Qwen2:1.5b (934 MB)
- **Pros**: More balanced performance
- **Cons**: Still underperforms Sentence-BERT on SuperDialseg
- **Best for**: When you need slightly better accuracy

## Threshold Sensitivity

The models showed high sensitivity to threshold settings:
- Qwen2:0.5b: Best with threshold=0.5 (lower than most)
- TinyLlama: Best with threshold=0.7
- Qwen2:1.5b: Best with threshold=0.6

## Comparison with Test Results

| Model | Test Dialogue F1 | Real Data Avg F1 | Degradation |
|-------|------------------|------------------|-------------|
| qwen2:0.5b | 0.571 | 0.292 | -49% |
| tinyllama | 1.000 | 0.219 | -78% |
| qwen2:1.5b | 0.667 | 0.278 | -58% |

All models showed significant performance degradation on real data compared to simple test dialogues.

## Recommendations

1. **For TIAGE-like datasets**: Use qwen2:0.5b (best F1=0.333)
2. **For SuperDialseg-like datasets**: Use Sentence-BERT (F1=0.571)
3. **For extreme size constraints**: qwen2:0.5b is viable (352 MB)
4. **For speed-critical applications**: Avoid instruct models

## Conclusion

While qwen2:0.5b at 352 MB is remarkably small and achieves competitive performance on some datasets (beating Sentence-BERT on TIAGE), the instruct-based approach generally underperforms embedding methods on dialogue segmentation tasks. The trade-off is:

- **Instruct models**: More interpretable, no training needed, but slower and less accurate
- **Embedding methods**: Faster and more accurate, but require training/fine-tuning

For production systems, a hybrid approach might be optimal:
1. Use Sentence-BERT for initial fast detection
2. Use qwen2:0.5b for verification or edge cases
3. Use larger instruct models for explainability when needed