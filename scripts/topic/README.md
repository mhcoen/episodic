# Topic Detection Test Scripts

This directory contains test scripts for validating topic detection behavior in Episodic.

## Purpose

These scripts test various conversational patterns to ensure topic detection correctly identifies when conversations shift to new subjects vs. when they're exploring different aspects of the same topic.

## Test Scripts

### test-gradual-progression.txt
- **Expected:** 1 topic
- **Tests:** Natural progression within a single domain (Python basics → advanced)
- **Validates:** Related concepts stay together

### test-explicit-transitions.txt
- **Expected:** 4 topics
- **Tests:** Recognition of explicit transition phrases ("I have a different question", "Changing topics")
- **Validates:** Clear verbal cues trigger topic changes

### test-related-domains.txt
- **Expected:** 2-3 topics
- **Tests:** Boundaries between related but distinct fields (human medicine → veterinary → biology)
- **Validates:** Different expertise domains are separated

### test-depth-exploration.txt
- **Expected:** 1 topic
- **Tests:** Deep dive from general to extremely specific (ML basics → specific CNN architectures)
- **Validates:** Depth exploration doesn't trigger false splits

### test-conversation-flow.txt
- **Expected:** 3 topics
- **Tests:** Natural conversation with soft transitions (Japan travel → Japanese cooking → nutrition)
- **Validates:** Natural flow with logical but distinct topics

### test-mixed-patterns.txt
- **Expected:** 4-5 topics
- **Tests:** Various transition types in one conversation
- **Validates:** Different transition patterns are handled correctly

### test-ambiguous-transitions.txt
- **Expected:** 2-3 topics
- **Tests:** Edge cases where topic boundaries are unclear (France geography → French cuisine → wine regions)
- **Validates:** Ambiguous cases are handled reasonably

## Usage

Run any test script with:
```bash
/script scripts/topic/test-name.txt
```

Then check results with:
```bash
/topics
```

## Configuration

These tests assume:
- Topic detection model: `gpt-3.5-turbo` (can be changed with `/set topic_detection_model`)
- Topic detection v3 prompt is enabled (default)
- Default minimum messages before topic change threshold