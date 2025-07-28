# Topic Detection Evaluation Summary

## Overview
This document summarizes the comprehensive evaluation of topic detection methods for the Episodic memory system, including the development and evaluation of multiple approaches culminating in a fine-tuned model achieving F1=0.667.

## Datasets Used

### 1. SuperDialseg
- **Source**: Primary dialogue segmentation dataset
- **Size**: ~9,400 conversations with topic boundaries
- **Splits**: train/validation/test
- **Format**: Multi-turn dialogues with annotated topic transitions

### 2. TIAGE (Tech, Insurance, Airlines)
- **Source**: Industry-specific customer service dialogues
- **Size**: ~600 conversations across 3 domains
- **Splits**: train/validation/test
- **Format**: Customer-agent interactions with topic segments

### 3. DialSeg711
- **Source**: 711 annotated dialogues
- **Size**: 711 conversations
- **Usage**: Additional validation data
- **Format**: General dialogues with topic boundaries

### 4. MP2D (Multi-Party to Dialogue)
- **Source**: EMNLP 2024 dataset samples
- **Size**: Sample data (~130KB)
- **Usage**: Additional training examples
- **Format**: Multi-party conversations converted to dialogues

## Methods Evaluated

### 1. Traditional Methods
- **TF-IDF with Cosine Similarity**
  - Window size: 3
  - F1 Score: 0.560
  - Speed: 5,418 messages/second
  
- **Sentence-BERT Embeddings**
  - Model: all-MiniLM-L6-v2
  - Window size: 3
  - F1 Score: 0.571
  - Speed: 390 messages/second

### 2. Statistical Methods
- **Bayesian Changepoint Detection**
  - F1 Score: 0.341 (poor performance)
  - Not suitable for dialogue topic detection

### 3. LLM-Based Methods
- **Small Instruct Models (Ollama)**
  - Models tested: qwen2:0.5b, TinyLlama, phi3:mini
  - Window size: 1 (couldn't handle window_size=3)
  - F1 Scores: 0.305-0.455
  - Speed: ~10 messages/second
  - Issue: Models <1B parameters cannot follow complex instructions

### 4. Fine-Tuned Model (Best Performance)
- **Model**: microsoft/xtremedistil-l6-h256-uncased
- **Parameters**: 13M
- **Training Data**: 55,657 examples from all 4 datasets
- **F1 Score**: 0.667
- **Precision**: 0.719
- **Recall**: 0.621
- **Speed**: 197 messages/second
- **Training Time**: 9.1 minutes on Apple Silicon

## Key Findings

1. **Window Size Matters**: Comparing methods with different window sizes (1 vs 3) is meaningless. All methods must use the same window size for fair comparison.

2. **Small LLMs Struggle**: Models under 1B parameters cannot reliably follow instructions for complex window-based comparisons. They output invalid responses outside the 0-1 range.

3. **Fine-Tuning Wins**: A small fine-tuned model (13M params) significantly outperforms all other methods, including much larger instruct models.

4. **Speed vs Accuracy Trade-off**:
   - TF-IDF: Fastest but lowest accuracy
   - Fine-tuned model: Good balance (197 msg/s, F1=0.667)
   - LLM prompting: Slowest and poor accuracy

5. **Multi-Dataset Training**: Training on all available datasets improved generalization compared to single-dataset approaches.

## Implementation Files

### Core Evaluation Framework
- `evaluation/evaluate_comprehensive.py` - Main evaluation script
- `evaluation/eval_metrics.py` - Metrics implementation (Pk, WindowDiff, F1)
- `evaluation/base_detector.py` - Base class for all detectors

### Dataset Loaders
- `evaluation/superdialseg_loader.py` - SuperDialseg dataset loader
- `evaluation/tiage_loader.py` - TIAGE dataset loader
- `evaluation/mp2d_loader.py` - MP2D dataset loader
- `evaluation/dialseg_loader.py` - DialSeg711 loader

### Detection Methods
- `evaluation/sliding_window_detector.py` - TF-IDF implementation
- `evaluation/sentence_bert_detector.py` - Sentence-BERT implementation
- `evaluation/ollama_instruct_detector.py` - LLM prompting approach
- `evaluation/bayesian_detector.py` - Bayesian changepoint detection

### Fine-Tuning Pipeline
- `evaluation/prepare_all_datasets.py` - Comprehensive data preparation
- `evaluation/finetune_full_simple.py` - Full-scale fine-tuning script
- `evaluation/benchmark_inference_speed_simple.py` - Speed benchmarking

### Models
- `evaluation/finetuned_models/topic_detector_full.pt` - Final trained model (51MB)

## Practical Implications

For a 1-hour conversation (~600 messages):
- **TF-IDF**: 0.111s (real-time capable)
- **Fine-tuned Model**: 3.0s (real-time capable)
- **Sentence-BERT**: 1.5s (real-time capable)
- **Ollama LLM**: 60s (too slow for real-time)

## Recommendations

1. **Use the Fine-Tuned Model** for production:
   - Best accuracy (F1=0.667)
   - Fast enough for real-time (197 msg/s)
   - Small model size (51MB)
   - Direct PyTorch integration

2. **Fallback Options**:
   - TF-IDF for ultra-low latency requirements
   - Sentence-BERT for systems already using embeddings

3. **Avoid**:
   - Small instruct LLMs (unreliable outputs)
   - Bayesian methods (poor performance on dialogues)
   - Window sizes > 1 for LLMs < 1B parameters

## Future Work

1. Test larger fine-tuned models (e.g., BERT-base)
2. Implement ensemble methods combining multiple approaches
3. Explore few-shot learning with larger LLMs
4. Add online learning capabilities
5. Evaluate on more diverse dialogue types

## Usage Example

```python
# Load the fine-tuned model
from evaluation.topic_detector import TopicDetector

detector = TopicDetector(
    model_path="evaluation/finetuned_models/topic_detector_full.pt",
    device="mps"  # or "cpu"
)

# Detect topic boundaries
messages = [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "It's sunny and 75°F today."},
    {"role": "user", "content": "Can you help me debug Python code?"},
    # ...
]

boundaries = detector.detect_boundaries(messages)
print(f"Topic changes at positions: {boundaries}")
```

## Conclusion

The fine-tuned XtremDistil model provides the best balance of accuracy, speed, and resource efficiency for topic detection in conversational systems. It processes 197 messages per second with F1=0.667, making it suitable for real-time applications while maintaining good boundary detection performance.