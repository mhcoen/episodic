#!/bin/bash
# Test neural topic detection in Episodic CLI

echo "Testing neural topic detection in Episodic..."
echo

# Set debug mode and neural detection
echo "Enabling debug mode and neural detection..."
python -m episodic set debug true
python -m episodic set neural_topic_detection true

echo
echo "Starting conversation about programming..."
echo

# Programming conversation
echo "How do I implement a binary search tree?" | python -m episodic talk
echo
echo "What's the time complexity of search operations?" | python -m episodic talk
echo
echo "Can you show me code for balancing the tree?" | python -m episodic talk
echo

echo "NOW CHANGING TOPIC TO COOKING..."
echo
echo "What's a good recipe for chocolate cake?" | python -m episodic talk
echo

echo "Checking topics..."
python -m episodic topics

# Cleanup
echo
echo "Disabling debug mode..."
python -m episodic set debug false