#!/bin/bash
# Initialize database and test dual-window detection

echo "=== Initializing Database and Testing Dual-Window Detection ==="
echo

# Clean up
echo "1. Cleaning up old database..."
rm -rf ~/.episodic
mkdir -p ~/.episodic

# Initialize by running episodic once
echo -e "\n2. Initializing database..."
echo "/exit" | python -m episodic >/dev/null 2>&1

# Configure settings using separate python script
echo -e "\n3. Configuring dual-window detection..."
python -c "
from episodic.config import config
config.set('debug', True)
config.set('use_dual_window_detection', True)
config.set('use_sliding_window_detection', False)
config.set('dual_window_high_precision_threshold', 0.2)
config.set('dual_window_safety_net_threshold', 0.25)
config.set('skip_llm_response', True)
config.set('automatic_topic_detection', True)
config.set('min_messages_before_topic_change', 2)  # Lower for testing
print('Configuration set successfully')
"

# Run the populate script
echo -e "\n4. Running populate script with dual-window detection..."
echo -e "/script scripts/populate_database.txt\n/topics\n/exit" | python -m episodic 2>&1 | grep -E "(Message |🔍|Topic |Current topic|High precision|Safety net|TOPIC CHANGED|Created topic)"

# Show final topics
echo -e "\n5. Final topic list:"
echo "/topics" | python -m episodic 2>&1 | grep -A50 "📚 Topics in this conversation"