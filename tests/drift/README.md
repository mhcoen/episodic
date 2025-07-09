# Topic Drift Detection Test Suite

This directory contains test scripts for evaluating topic detection algorithms without the overhead of LLM responses.

## Test Scripts

### topic_drift_test_01.txt - Basic Test
- **Purpose**: Test basic topic detection with clear boundaries
- **Topics**: 5 distinct topics (Python, Fitness, Photography, Finance, Gardening)
- **Messages**: 10 messages per topic
- **Expected**: Clear topic boundaries at messages 11, 21, 31, 41

### topic_drift_test_02_gradual.txt - Gradual Transitions
- **Purpose**: Test detection of gradual vs sharp topic transitions
- **Topics**: 3 main topics with varying transition types
- **Features**:
  - Web Dev → Cloud Computing (gradual transition)
  - Data Science → Machine Learning (natural progression)
  - Cooking → Restaurant Management (sharp transition)
- **Expected**: Different drift scores based on transition sharpness

### topic_drift_test_03_challenging.txt - Edge Cases
- **Purpose**: Test challenging scenarios for topic detection
- **Challenges**:
  1. Very similar topics (Python vs JavaScript)
  2. Returning to previous topic (Fitness → Cooking → Fitness)
  3. Ambiguous transitions (Health/Nutrition/Cooking overlap)
  4. Rapid topic switches (below threshold)
- **Expected**: Tests algorithm robustness and edge case handling

### topic_drift_test_04_algorithms.txt - Algorithm Comparison
- **Purpose**: Compare different detection algorithms on identical input
- **Algorithms**:
  1. Sliding Window Detection (default)
  2. Hybrid Detection
  3. LLM-based Detection
- **Usage**: Run multiple times with different settings
- **Expected**: Reveals strengths/weaknesses of each approach

## Usage

### Basic Usage
```bash
# Run a test script
python -m episodic --execute tests/drift/topic_drift_test_01.txt

# Or interactively
python -m episodic
> /script tests/drift/topic_drift_test_01.txt
```

### Configuration Options
Key settings for testing:
- `skip_llm_response`: Skip LLM calls for faster testing
- `debug`: Show detailed detection information
- `show_drift`: Display semantic drift scores
- `show_topics`: Show topic evolution
- `min_messages_before_topic_change`: Threshold for topic changes (default: 8)

### Algorithm Selection
```bash
# Sliding Window Detection (default)
/set use_sliding_window_detection true
/set use_hybrid_topic_detection false

# Hybrid Detection
/set use_sliding_window_detection false
/set use_hybrid_topic_detection true

# LLM-based Detection
/set use_sliding_window_detection false
/set use_hybrid_topic_detection false
/set skip_llm_response false  # Must allow LLM calls
```

## Analyzing Results

After running a test:
1. Check `/topics` to see detected topics and boundaries
2. Check `/topics stats` for detailed statistics
3. Look for semantic drift scores in the output
4. Compare actual vs expected boundaries

## Creating New Tests

When creating new test scripts:
1. Start with `/init --erase` for a clean slate
2. Set configuration options at the beginning
3. Group messages by intended topic (8+ messages per topic)
4. Add comments marking expected boundaries
5. Reset settings at the end
6. Include analysis commands (`/topics`, `/topics stats`)

## Tips

- Use clear topic transitions for initial testing
- Test edge cases like returning topics or ambiguous boundaries
- Run the same test with different algorithms to compare
- Watch the semantic drift scores during transitions
- Use debug mode to understand detection decisions