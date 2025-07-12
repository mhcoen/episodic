# DEBUG.md - Debugging Guide for Episodic

This file documents debugging flags, troubleshooting steps, and development utilities for the Episodic project.

## Debug Flags

### General Debug Mode
```bash
/set debug on
```
Enables general debugging output including:
- Topic detection decisions
- Configuration loading
- Basic operation flow
- Error details with stack traces

### Specialized Debug Flags

#### LLM API Debugging
```bash
/set debug_llm_api on
```
Shows detailed LLM API information:
- Thread and call tracking (`[LLM API] Thread X: Call #Y`)
- Model and streaming mode
- Calling function location
- Message counts
- Cost calculations
- Response lengths
- API errors

#### Streaming Output Debugging
```bash
/set debug_streaming_verbose on
```
Shows word-by-word streaming details:
- Each word with its parsing state
- Line start, header, list detection
- Format preservation decisions
- Terminal width calculations
- Streaming mode selection

#### Web Search Debugging
The regular `debug` flag shows web search fallback:
- Provider selection
- Fallback attempts
- API errors (missing keys, rate limits)

## Common Debugging Scenarios

### Topic Detection Issues
1. Enable debug mode: `/set debug on`
2. Send messages to trigger topic detection
3. Look for:
   - "Topic detection decision" messages
   - Topic scores and thresholds
   - Topic boundary detection

### LLM Response Issues
1. Enable LLM debugging: `/set debug_llm_api on`
2. Look for:
   - Model being used
   - API errors
   - Cost calculation failures

### Streaming/Formatting Issues
1. Enable streaming debug: `/set debug_streaming_verbose on`
2. Look for:
   - Word parsing decisions
   - Bold/header detection
   - Line wrapping behavior

### Web Search Issues
1. Enable debug: `/set debug on`
2. Run a web search: `/ws test query`
3. Look for:
   - "🔍 Searching with [Provider]..."
   - "⚠️ [Provider] failed: [error]"
   - "Trying next provider..."

## Database Debugging

### Check Database Location
```python
from episodic.db import get_db_path
print(get_db_path())  # Should be ~/.episodic/episodic.db
```

### View Recent Nodes
```bash
/list --count 10  # Show last 10 nodes
/show <node_id>   # Show specific node details
```

### Topic Information
```bash
/topics           # List all topics
/topic-scores     # Show topic detection scores
```

## Configuration Debugging

### View Current Configuration
```bash
/set              # Show all settings
/verify           # Verify database and config
/config-docs      # Show configuration documentation
```

### Check Model Configuration
```bash
/model            # Show current chat model
/model list       # Show all models for all contexts
/mset             # Show all model parameters
```

### Web Search Configuration
```bash
/web              # Show web search status
/websearch config # Detailed web search config
```

## Performance Debugging

### Benchmarking
```bash
/benchmark        # Show performance statistics
```

### API Usage
```bash
/compression api-stats  # Show LLM API usage by operation
/cost                  # Show session costs
```

## Test Scripts for Debugging

### Topic Detection Test
```bash
python -m episodic --execute scripts/testing/three-topics-test.txt
```

### Web Search Test
```bash
python scripts/test_web_fallback.py
```

## Environment Variables

For debugging environment variable issues:
```bash
# Check if variables are set
echo $GOOGLE_API_KEY
echo $GOOGLE_SEARCH_ENGINE_ID
echo $EPISODIC_DB_PATH
```

## Development Utilities

### Database Safeguards
The system prevents creating databases in the project directory:
- `episodic/db_safeguards.py` - Validates database paths
- Raises `ValueError` if attempting to create DB in project root

### Debug Output Functions
- `debug_print()` from `episodic/debug_utils.py` - Consolidated debug output
- Respects the `debug` flag automatically

## Troubleshooting Checklist

1. **Commands not working**: Check `/help all` and verify command syntax
2. **Database issues**: Run `/verify` and check `~/.episodic/` exists
3. **Model errors**: Verify API keys are set in environment
4. **Web search failing**: Check provider configuration and API keys
5. **Formatting issues**: Try `/set text_wrap off` temporarily

## Adding New Debug Flags

When adding new debug functionality:
1. Add the flag to `config_defaults.py`
2. Use `config.get('debug_flag_name', False)` to check
3. Document the flag in this file
4. Consider if it should be part of general `debug` or separate

## Log File Locations

Currently, Episodic doesn't write log files. All debug output goes to stderr.
Future enhancement: Add file logging with `/set log_file <path>`