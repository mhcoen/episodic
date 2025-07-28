# Small Instruct Models with Window Size 3 - Summary

## Executive Summary

Testing small instruct models (qwen2:0.5b, TinyLlama, phi3:mini) with window_size=3 revealed fundamental limitations in their ability to handle complex multi-message comparisons.

## Models Tested

1. **qwen2:0.5b (352 MB)** - Smallest available model
   - Cannot follow basic instructions for window comparisons
   - Outputs numbers outside 0-1 range (2.6, 3.5, 4.5)
   - Fails regardless of prompt structure

2. **TinyLlama (637 MB)** - Second smallest
   - Outputs verbose explanations instead of numbers
   - Cannot comply with "output only a number" instruction
   - Extracts concepts but fails at numerical scoring

3. **phi3:mini (2.2 GB)** - Larger "small" model
   - Sometimes outputs just numbers, but often verbose
   - Frequently outputs numbers outside 0-1 range
   - With robust parsing: F1=0.455 on SuperDialseg

## Key Findings

### 1. Instruction Following Degrades with Complexity
- Simple prompts (window_size=1): Models can output single numbers
- Complex prompts (window_size=3): Models fail to follow constraints
- The issue is not the mathematical concept but instruction compliance

### 2. Model Size Matters for Complex Instructions
```
Model Size vs Instruction Following (window_size=3):
- <1B params: Cannot reliably output constrained numbers
- 1-3B params: Partial success with robust parsing
- >7B params: Works as intended (e.g., mistral:instruct)
```

### 3. Comparison Results (Window Size 3)

| Method | F1 Score | Notes |
|--------|----------|-------|
| Sentence-BERT | 0.571 | Specialized embedding model |
| Sliding Window | 0.560 | Simple pattern matching |
| phi3:mini (robust) | 0.455 | With keyword extraction + robust parsing |
| qwen2:0.5b | Failed | Cannot produce valid outputs |
| TinyLlama | Failed | Produces explanations, not numbers |

### 4. Why Window Size 1 Works But 3 Doesn't

**Window Size 1 (Pairwise)**:
- Simple instruction: "Compare message A to message B"
- Single comparison point
- Clear binary decision

**Window Size 3**:
- Complex instruction: "Compare 3 messages to 3 messages"
- Multiple comparison points
- Requires aggregating information
- Small models can't maintain instruction constraints while processing complexity

## Practical Implications

1. **For production use with small models**:
   - Stick to window_size=1 (pairwise comparison)
   - Use extremely simple prompts
   - Consider specialized models over general LLMs

2. **For window_size=3 requirements**:
   - Use models ≥7B parameters
   - Or use non-LLM methods (Sentence-BERT, sliding window)
   - Small instruct models are not suitable

3. **Trade-offs**:
   - Small models (qwen2:0.5b): Fast but limited to simple tasks
   - Medium models (phi3:mini): Slower, partially works with workarounds
   - Large models (mistral:instruct): Slowest but works correctly

## Conclusion

The experiment to fairly compare instruct models with window_size=3 revealed that:
1. Model size directly correlates with instruction-following ability
2. Small models (<1B) fundamentally cannot handle complex prompts
3. The "smallest working model" for window_size=3 is phi3:mini (2.2GB) with significant workarounds
4. For reliable window_size=3 detection, use models ≥7B parameters or specialized embedding approaches