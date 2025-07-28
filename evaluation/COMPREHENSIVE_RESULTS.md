# Comprehensive Topic Detection Evaluation Results

## Executive Summary

We evaluated multiple topic detection approaches on four dialogue segmentation datasets:
- **SuperDialseg**: 500 dialogues (9,478 total available)
- **DialSeg711**: 300 dialogues (711 total available)
- **TIAGE**: 100 dialogues (100 total available)
- **MP2D**: Limited sample (structured Q&A format)

Additionally, we analyzed the EMNLP 2024 approach which uses fine-tuned language models (T5, Flan-T5, T0) for topic segmentation.

### Best Performers by Dataset

**SuperDialseg**:
1. **Sentence-BERT (t=0.3)**: F1=0.571, WindowDiff=0.424
2. **Sliding Window (t=0.3)**: F1=0.560, WindowDiff=0.437
3. **Bayesian BOCPD**: F1=0.474, WindowDiff=0.453

**DialSeg711**:
1. **Sentence-BERT (t=0.5)**: F1=0.467, WindowDiff=0.545
2. **Bayesian BOCPD**: F1=0.416, WindowDiff=0.465
3. **Sliding Window (t=0.7)**: F1=0.372, WindowDiff=0.431

**TIAGE**:
1. **Sentence-BERT (t=0.5)**: F1=0.222, WindowDiff=0.585
2. **Sliding Window (t=0.3)**: F1=0.219, WindowDiff=0.597
3. **Bayesian BOCPD (t=0.2)**: F1=0.199, WindowDiff=0.559

## Detailed Results

### SuperDialseg Dataset (500 dialogues)

| Method | Config | Precision | Recall | F1 | F1(w=3) | WindowDiff | Pk | Speed |
|--------|--------|-----------|--------|-------|---------|------------|-------|-------|
| **Sentence-BERT** | t=0.3 | 0.580 | 0.611 | **0.571** | 0.728 | **0.424** | 0.431 | 42.2s |
| Sliding Window | t=0.3, w=3 | 0.574 | 0.596 | 0.560 | **0.729** | 0.437 | 0.443 | 67.4s |
| Bayesian BOCPD | t=0.25, w=3 | 0.537 | 0.477 | 0.474 | 0.629 | 0.453 | 0.460 | 49.2s |
| Hybrid | t=0.9 | 0.368 | 0.278 | 0.285 | 0.456 | 0.490 | 0.495 | ~100s |
| Keywords | t=0.5 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | 0.493 | 0.497 | Fast |

### DialSeg711 Dataset (300 dialogues)

| Method | Config | Precision | Recall | F1 | F1(w=3) | WindowDiff | Pk | Speed |
|--------|--------|-----------|--------|-------|---------|------------|-------|-------|
| **Sentence-BERT** | t=0.5 | 0.336 | 0.814 | **0.467** | 0.549 | 0.545 | 0.517 | 80.2s |
| Bayesian BOCPD | t=0.3 | 0.299 | 0.764 | 0.416 | **0.560** | **0.465** | **0.472** | ~35s |
| Sliding Window | t=0.7 | 0.299 | 0.639 | 0.372 | 0.554 | 0.431 | 0.411 | ~50s |
| Keywords | t=0.5 | 0.145 | 0.223 | 0.158 | 0.241 | 0.310 | 0.302 | Fast |

### TIAGE Dataset (100 dialogues)

| Method | Config | Precision | Recall | F1 | F1(w=3) | WindowDiff | Pk | Speed |
|--------|--------|-----------|--------|-------|---------|------------|-------|-------|
| **Sentence-BERT** | t=0.5 | 0.192 | 0.305 | **0.222** | - | 0.585 | 0.569 | 14.4s |
| Sliding Window | t=0.3 | 0.180 | 0.315 | 0.219 | - | 0.597 | 0.575 | 24.2s |
| Bayesian BOCPD | t=0.2 | 0.170 | 0.271 | 0.199 | - | **0.559** | - | ~14s |
| Sliding Window | t=0.7 | 0.189 | 0.193 | 0.177 | - | 0.502 | 0.516 | 23.1s |

## Key Insights

### 1. Model Performance Patterns

**Sentence-BERT** shows the strongest overall performance:
- Best F1 scores on both datasets
- Particularly effective on subtle transitions (DialSeg711)
- Consistent performance across different dialogue types
- Benefits from pre-trained semantic understanding

**Sliding Window** performs well on structured dialogues:
- Close second on SuperDialseg (F1=0.560)
- Optimal threshold varies significantly by dataset (0.3 vs 0.7)
- Fast and simple to implement
- Less effective on subtle topic changes

**Bayesian BOCPD** offers balanced performance:
- More consistent across datasets
- Best WindowDiff on DialSeg711 (0.465)
- Handles uncertainty well
- Computationally efficient

### 2. Dataset Characteristics

**SuperDialseg**:
- Lower optimal thresholds (0.3) work best
- Clear topic boundaries
- All methods show reasonable performance
- Sentence embeddings provide marginal gains

**DialSeg711**:
- Higher thresholds needed (0.5-0.7)
- More subtle topic transitions
- Sentence-BERT shows significant advantage
- Keyword-based methods struggle

**TIAGE**:
- Very casual conversations with subtle transitions
- Lower F1 scores across all methods (max 0.222)
- Topic boundaries often occur mid-conversation flow
- All methods struggle with precision/recall balance
- Represents real-world challenge of informal dialogue

### 3. Threshold Sensitivity

Optimal thresholds vary by dataset and method:
- SuperDialseg: 0.25-0.3 across methods
- DialSeg711: 0.3-0.7 depending on approach
- TIAGE: 0.2-0.5 (lower thresholds for Bayesian)
- Sentence-BERT most robust to threshold choice
- Production systems need adaptive thresholding

### 4. Computational Trade-offs

| Method | Speed (dialogues/sec) | Memory | Real-time Capable |
|--------|----------------------|---------|-------------------|
| Keywords | >100 | Minimal | Yes |
| Sliding Window | ~8 | Low | Yes |
| Bayesian BOCPD | ~10 | Low | Yes |
| Sentence-BERT | ~12 | 50MB model | Yes |

## Recommendations

### For Production Use

1. **Primary Recommendation: Sentence-BERT**
   - Best overall F1 scores
   - Handles diverse dialogue types
   - Pre-trained knowledge transfers well
   - Acceptable computational cost

2. **Alternative: Sliding Window**
   - When simplicity is paramount
   - For well-structured dialogues
   - Minimal dependencies
   - Fastest inference

3. **Special Cases: Bayesian BOCPD**
   - When uncertainty quantification matters
   - For systems needing probability estimates
   - Good balance of performance and interpretability

### Implementation Strategy

1. **Start with Sentence-BERT** (t=0.3-0.5)
   - Use `all-MiniLM-L6-v2` for efficiency
   - Consider `all-mpnet-base-v2` for better quality

2. **Adaptive Thresholding**
   - Monitor dialogue characteristics
   - Adjust thresholds based on domain
   - Consider ensemble approaches

3. **Optimization Opportunities**
   - Fine-tune on domain-specific data
   - Combine multiple signals
   - Add context beyond 3-message window

## Future Work

1. **Ensemble Methods**: Combine Sentence-BERT with Bayesian inference
2. **Domain Adaptation**: Fine-tune on specific dialogue types
3. **Online Learning**: Adapt thresholds during deployment
4. **Larger Models**: Test with more powerful transformers
5. **Multi-modal**: Incorporate speaker changes, timing, etc.
6. **Prompt Engineering**: Explore the EMNLP 2024 approach of using LLMs with specific prompts
7. **Synthetic Data**: Generate training data for challenging domains like TIAGE

## MP2D Dataset and EMNLP 2024 Approach

The MP2D dataset represents a different paradigm:
- **Structured Q&A Format**: Generated from passages rather than natural conversation
- **Clear Topic Labels**: Each segment has explicit topic names
- **EMNLP Approach**: Fine-tuned T5/Flan-T5 models achieve high performance

Key insights:
1. **Task Formulation Matters**: The EMNLP approach frames segmentation as sequence-to-sequence
2. **Data Quality**: MP2D's structured format is easier than natural conversations
3. **Trade-offs**: Fine-tuning offers accuracy but requires computational resources

For Episodic's use case, the unsupervised Sentence-BERT approach remains optimal for:
- Real-time processing
- Domain generalization
- Resource efficiency

However, the EMNLP approach suggests opportunities for:
- Domain-specific fine-tuning when accuracy is critical
- Prompt engineering with existing LLMs
- Hybrid approaches combining unsupervised detection with LLM verification

## Conclusion

Sentence-BERT emerges as the best overall approach, achieving F1 scores of 0.571 on SuperDialseg, 0.467 on DialSeg711, and 0.222 on TIAGE. It successfully balances performance, generalization, and computational efficiency across diverse dialogue types.

Key findings:
1. **Performance varies dramatically by dataset**: F1 scores range from 0.571 (SuperDialseg) to 0.222 (TIAGE)
2. **TIAGE reveals limitations**: All methods struggle with casual, informal conversations
3. **Sentence-BERT is most robust**: Consistently best or near-best across all datasets
4. **Threshold adaptation is critical**: Optimal values vary from 0.2 to 0.7 depending on dataset

For Episodic's use case, we recommend:
- **Primary**: Sentence-BERT with adaptive thresholding
- **Fallback**: Sliding window for resource-constrained environments
- **Future work**: Domain-specific fine-tuning for casual conversations like TIAGE