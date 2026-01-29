# Test Scripts for Episodic

This directory contains test scripts for exercising the query understanding and memory retrieval systems.

## Quick Start

```bash
# 1. Set up test fixtures (creates test database with known conversations)
/test setup

# 2. Enable test mode (switch to test database)
/test on

# 3. Run a test script
/scripts run tests/scripts/test_temporal_queries.txt

# 4. When done, switch back to production
/test off
```

## Test Database Structure

The test fixtures create conversations at known temporal offsets relative to a fixed reference time (2026-01-26 12:00 UTC):

| Temporal Reference | Topic Name | Content |
|-------------------|------------|---------|
| yesterday | machine-learning-basics | Supervised/unsupervised learning, overfitting |
| 3 days ago | python-asyncio | asyncio, gather vs wait |
| last week | database-indexing | B-trees, hash indexes |
| last month | quantum-computing | Qubits, coherence, error correction |

## Available Test Scripts

### test_temporal_queries.txt
Tests temporal reference parsing and resolution:
- "what did we discuss yesterday" → machine-learning-basics
- "what did we discuss 3 days ago" → python-asyncio
- "what did we discuss last week" → database-indexing
- "what did we discuss last month" → quantum-computing

### test_topic_queries.txt
Tests topic/segment reference parsing:
- Explicit topic syntax (`in topic: quantum-computing`)
- Topic disambiguation
- Ongoing vs closed topics

### test_speaker_queries.txt
Tests speaker reference parsing:
- "what did I say" → user messages only
- "what did you say" → assistant messages only
- "what did we discuss" → all messages

### test_discussion_queries.txt
Tests DiscussionQuery AST recognition:
- "when did we discuss X"
- "have we talked about X before"
- "did I mention X"
- Broadness cues (ever, before, previously)

## Test Commands

```bash
/test              # Show test mode status
/test on           # Switch to test database
/test off          # Switch to production database
/test setup        # Initialize test database with fixtures
/test status       # Show detailed test DB status
/test destroy      # Delete test database
```

## Writing New Test Scripts

Test scripts are plain text files with:
- Lines starting with `#` are comments
- Lines starting with `/` are commands
- All other lines are chat messages (queries to test)

Example:
```
# Test temporal queries
# Expected: should retrieve yesterday's content

what did we discuss yesterday

# This should find machine-learning content
did I ask about overfitting yesterday
```

## Running Tests Programmatically

```python
from episodic.test_fixtures import setup_test_environment, teardown_test_environment
from datetime import datetime
from zoneinfo import ZoneInfo

# Use fixed reference time for reproducibility
ref_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

# Set up test environment
manager = setup_test_environment(ref_time)

# Run your tests...
conn = manager.get_connection()
# ...

# Clean up
teardown_test_environment(manager)
```

## Integration with pytest

```bash
# Run query understanding tests (no DB required)
pytest tests/unit/test_parser.py tests/unit/test_lexer.py tests/unit/test_resolver.py -v

# Run integration tests with fixtures
pytest tests/integration/test_query_retrieval.py -v
```

## Notes

- The test database is stored at `~/.episodic/episodic_test.db`
- Test mode is a config flag; restart may be needed for full effect
- Test fixtures use deterministic timestamps for reproducibility
- Production database is never modified by test operations
