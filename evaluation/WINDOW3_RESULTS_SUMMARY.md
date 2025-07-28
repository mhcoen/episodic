# Small Instruct Models with Window Size 3 - Results Summary

## Fair Comparison Results (All methods using window_size=3)

### TIAGE Dataset Results

| Method | Window Size | F1 Score | Notes |
|--------|-------------|----------|-------|
| **Sentence-BERT** | 3 | 0.222 | Previous best |
| Sliding Window | 3 | 0.219 | Baseline |
| **tinyllama (t=0.7)** | 3 | **0.308** | Best small model |
| qwen2:0.5b (t=0.5) | 3 | 0.305 | Close second |
| qwen2:1.5b (t=0.6) | 3 | 0.297 | Third place |

### Key Findings

1. **Small models outperform Sentence-BERT on TIAGE** when using the same window size (3)
   - TinyLlama: F1=0.308 vs Sentence-BERT: F1=0.222 (38% improvement!)
   - Even the smallest model (qwen2:0.5b) beats Sentence-BERT (0.305 vs 0.222)

2. **Window size matters significantly**
   - With window_size=1: qwen2:0.5b got F1=0.333
   - With window_size=3: qwen2:0.5b got F1=0.305
   - The additional context actually hurt performance slightly

3. **Speed trade-offs**
   - qwen2:0.5b: ~1.7s per dialogue
   - tinyllama: ~25s per dialogue (15x slower)
   - qwen2:1.5b: ~2.5s per dialogue

4. **Model behavior with windows**
   - Many models produced verbose explanations instead of just numbers
   - The window-based prompts confused some models
   - This suggests instruct models may work better with simpler prompts

## Comparison Summary

When comparing apples to apples (all using window_size=3):

- **TIAGE**: Small instruct models WIN (F1=0.308 vs 0.222)
- **SuperDialseg**: Unable to test due to loader issue, but previous results with window_size=1 showed Sentence-BERT winning

## Conclusion

The fair comparison with window_size=3 shows that small instruct models can actually outperform specialized embedding methods on certain datasets like TIAGE. However:

1. They are still much slower (1.7-25s vs 0.01s per dialogue)
2. They struggle with complex window-based prompts
3. Performance varies significantly by dataset

For TIAGE specifically, even the tiniest model (qwen2:0.5b at 352 MB) is competitive, achieving F1=0.305 compared to Sentence-BERT's 0.222.