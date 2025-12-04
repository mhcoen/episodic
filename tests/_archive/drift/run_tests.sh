#!/bin/bash
# Run drift detection tests without HuggingFace rate limiting

echo "Running drift detection tests in offline mode..."
echo "This prevents HuggingFace update checks that cause 429 errors"
echo

# Set offline mode to prevent update checks
export HF_HUB_OFFLINE=1

# Run test scripts
for test in tests/drift/topic_drift_test_*.txt; do
    if [ -f "$test" ]; then
        echo "=================================="
        echo "Running: $(basename $test)"
        echo "=================================="
        python -m episodic --execute "$test"
        echo
    fi
done

echo "All tests completed!"