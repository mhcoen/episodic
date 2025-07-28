# Smallest Instruct Models for Topic Detection

## Model Size Comparison

| Model | Size | Speed (per transition) | F1 Score | Notes |
|-------|------|----------------------|----------|--------|
| **qwen2:0.5b** | **352 MB** | 0.10s | 0.571 | NEW SMALLEST! Ultra-fast, decent accuracy with threshold=0.5 |
| tinyllama:latest | 637 MB | 0.72s | 1.000 | Best accuracy with threshold=0.7 |
| qwen2:1.5b | 934 MB | 0.20s | 0.667 | Good balance of speed and accuracy |
| gemma:2b-instruct | 1.6 GB | 0.15s | 0.000 | Very fast but poor accuracy on test |
| phi3:instruct | 2.2 GB | 2.60s | 0.800 | Good accuracy but slower |
| mistral:instruct | 4.1 GB | 1-3s | 1.000 | Excellent accuracy |
| llama3:instruct | 4.7 GB | 1-3s | 1.000 | Excellent accuracy |

## Winner: Qwen2:0.5b (352 MB)

Qwen2:0.5b is the smallest available model capable of topic detection at just 352 MB. It's 45% smaller than TinyLlama and still achieves decent accuracy (F1=0.571) with proper threshold tuning.

### Key Advantages:
1. **Smallest footprint**: Only 352 MB (8% of mistral:instruct)
2. **Blazing fast**: ~0.10s per transition (7x faster than TinyLlama)
3. **Decent accuracy**: F1=0.571 with threshold=0.5
4. **Minimal resource usage**: Runs on extremely limited hardware

### Performance Characteristics:
- **Best threshold**: 0.5 (lower than most models)
- **Speed**: Fastest of all tested models
- **Trade-off**: Lower accuracy than TinyLlama but much faster/smaller

### When to Use Each:
- **Qwen2:0.5b**: When size and speed are critical, accuracy is secondary
- **TinyLlama**: When you need better accuracy but can afford 637 MB
- **Qwen2:1.5b**: Best balance of size (934 MB), speed, and accuracy (F1=0.667)

## Other Small Model Options

### Attempted but Not Available:
- `qwen:0.5b` - Not in Ollama repository
- `tinyllama:instruct` - Only base model available
- Nano/micro variants - Not found

### Quantized Options:
The quantized models (q2, q3, q4) available are still larger than TinyLlama:
- `mistral:7b-instruct-q4_0`: 4.1 GB
- `llama3.1:8b-instruct-q4_0`: 4.7 GB

## Recommendations

1. **For smallest size & fastest speed**: Use **qwen2:0.5b** (352 MB)
2. **For best accuracy in small package**: Use **tinyllama:latest** (637 MB)
3. **For balanced performance**: Use **qwen2:1.5b** (934 MB)
4. **For production use**: Test on your specific dataset to find optimal model/threshold

## Installation

```bash
# For the smallest model (352 MB)
ollama pull qwen2:0.5b

# For better accuracy (637 MB)
ollama pull tinyllama:latest

# For balanced performance (934 MB)
ollama pull qwen2:1.5b
```

## Usage Example

```python
from evaluation.ollama_instruct_detector import OllamaInstructDetector

# For smallest/fastest (352 MB)
detector = OllamaInstructDetector(
    model_name="qwen2:0.5b",
    threshold=0.5,  # Note: lower threshold
    window_size=1,
    verbose=False
)

# For best accuracy (637 MB)
detector = OllamaInstructDetector(
    model_name="tinyllama:latest",
    threshold=0.7,
    window_size=1,
    verbose=False
)

boundaries = detector.detect_boundaries(messages)
```

## Conclusion

**Qwen2:0.5b at 352 MB is the absolute smallest model** capable of topic detection. Key findings:

1. **Qwen2:0.5b (352 MB)**:
   - 45% smaller than TinyLlama
   - 7x faster (0.10s vs 0.72s per transition)
   - F1=0.571 with threshold=0.5
   - Best for extreme resource constraints

2. **TinyLlama (637 MB)**:
   - Better accuracy (F1=1.000 on test)
   - Still very small
   - Best for accuracy-focused deployments

3. **Qwen2:1.5b (934 MB)**:
   - Best balance overall
   - F1=0.667, reasonably fast
   - Good middle ground option

Choose based on your constraints: extreme size (qwen2:0.5b), accuracy (tinyllama), or balance (qwen2:1.5b).