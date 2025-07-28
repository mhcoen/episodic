# Window Size 3 Problem Analysis

## The Issue

When using window_size=3 with small instruct models on SuperDialseg, we're seeing:

1. **Models output numbers outside 0-1 range**
   - Example: qwen2:0.5b outputs "2.6376684950319335" and "3.5"
   - The models don't understand the constraint properly

2. **Verbose responses instead of numbers**
   - Many models produce long explanations
   - Some even try to write Python code to solve the problem

3. **Context confusion**
   - The window-based prompt with multiple messages confuses small models
   - They perform better with simple pairwise comparison

## Root Cause

The small models (qwen2:0.5b, tinyllama) have limited instruction-following capabilities compared to larger models like mistral:instruct. When given complex prompts with:
- Multiple messages to compare
- Specific output constraints (0.0 to 1.0)
- Window-based context

They struggle to:
1. Understand the numeric constraint
2. Compare multiple messages simultaneously
3. Output only a number without explanation

## Why It Works on TIAGE but Not SuperDialseg

TIAGE results were actually collected, but SuperDialseg failed because:
- The prompt parsing issues compound over many dialogues
- SuperDialseg has longer, more complex conversations
- The models may have been trained on data similar to TIAGE's simpler format

## Conclusion

The window_size=3 comparison reveals that:

1. **Small models work better with simple prompts** (window_size=1)
2. **The (3,3) window approach is too complex** for models under 1B parameters
3. **Larger instruct models (mistral, llama3) handle complex prompts better**

This explains why:
- Sentence-BERT (specialized model) performs well with window_size=3
- Small general-purpose instruct models struggle with the same setup
- The comparison isn't quite fair because the models can't properly process the window-based prompts

## Recommendation

For small instruct models:
- Use window_size=1 (simple pairwise comparison)
- Keep prompts extremely simple
- Consider them only for datasets where they've shown success (like TIAGE with window_size=1)