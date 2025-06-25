# CLAUDE.md

## Current Session Context

### Last Working Session (2025-06-25)
Fixed topic naming to analyze content when topics close, rather than using the triggering message. Re-implemented the /summary command after it was lost during code revert.

### Key System Understanding

#### Topic Detection Flow
1. User sends message → Topic detection runs (ollama/llama3)
2. If topic change detected → Close previous topic with proper name
3. Previous topic's content is analyzed to extract appropriate name
4. New topic starts as "ongoing-discussion" until it too is closed

#### Database Functions
- `store_topic()` - Creates new topic entry
- `update_topic_end_node()` - Extends topic boundary
- `update_topic_name()` - Renames topic (newly added)
- `get_recent_topics()` - Retrieves topic list

#### Important Code Locations
- Topic detection: `episodic/topics.py:detect_topic_change_separately()`
- Topic naming: `episodic/conversation.py:387-442` (in handle_chat_message)
- Summary command: `episodic/cli.py:1593-1701`
- Command parsing: `episodic/cli.py:2039-2056`

### Configuration Options
- `topic_detection_model` - Default: ollama/llama3
- `running_topic_guess` - Default: True (not yet implemented)
- `min_messages_before_topic_change` - Default: 8
- `show_topics` - Shows topic evolution in responses
- `debug` - Shows detailed topic detection info

### Recent Discoveries
- Topic boundary issues occur when nodes branch (non-linear history)
- The `--` prefix in topic names (like "--space") comes from the prompt response
- First topic creation has timing issues - may not trigger properly
- `get_ancestry()` returns nodes in reverse chronological order

### Test Scripts
- `scripts/test-complex-topics.txt` - 21 queries across multiple topics
- `scripts/test-topic-naming.txt` - Simple topic transitions
- `scripts/test-final-topic.txt` - Tests final topic handling

### Common Commands for Testing
```bash
# Full test with debug
echo -e "/init --erase\n/set debug on\n/script scripts/test-complex-topics.txt\n/topics\n/exit" | python -m episodic

# Check topics after test
python -m episodic
> /topics
> /exit

# Test summary
> /summary
> /summary 5
> /summary all
```