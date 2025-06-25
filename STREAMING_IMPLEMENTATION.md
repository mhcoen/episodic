# Streaming Implementation for Episodic

## Overview
This implementation adds streaming support for LLM responses in the episodic conversation system. Responses now stream in real-time as they are generated, providing a more responsive user experience.

## Key Features Implemented

### 1. Streaming Infrastructure (`llm.py`)
- Modified `_execute_llm_query()` to support a `stream` parameter
- Added `query_with_context()` streaming support
- Created `process_stream_response()` generator to yield content chunks from the stream

### 2. Streaming Display Handler (`conversation.py`)
- Updated `handle_chat_message()` to check for streaming preference
- Implemented real-time display with:
  - Proper color formatting using `get_llm_color()`
  - Word wrapping that works with partial lines
  - Line-by-line processing for clean output
  - Smart line breaking for long lines without newlines

### 3. Cost Calculation
- Maintained accurate cost tracking by making a non-streaming call after streaming
- This ensures we get proper token counts and cost information
- Future improvement: Extract usage data from streaming chunks when LiteLLM supports it

### 4. Configuration
- Added `stream_responses` config option (default: True)
- Added `/set stream on/off` command to toggle streaming
- Updated status display to show streaming state
- Configuration persists across sessions

## Usage

### Enable/Disable Streaming
```bash
# Toggle streaming
/set stream

# Enable streaming
/set stream on

# Disable streaming  
/set stream off

# Check current status
/set
```

### How It Works
1. When a user sends a message, the system checks if streaming is enabled
2. If enabled, it requests a streaming response from the LLM
3. As chunks arrive, they are:
   - Displayed immediately with proper formatting
   - Accumulated for database storage
   - Word-wrapped intelligently
4. After streaming completes, the full response is stored in the database
5. Cost information is calculated and displayed if enabled

## Technical Details

### Streaming Flow
1. `handle_chat_message()` checks `config.get("stream_responses", True)`
2. Calls `query_with_context(..., stream=True)` to get a generator
3. Processes chunks through `process_stream_response()`
4. Displays chunks with color and wrapping
5. Stores complete response in database

### Word Wrapping During Streaming
- Accumulates content until a newline is found
- Wraps complete lines respecting terminal width
- For very long lines without newlines, breaks at word boundaries
- Maintains proper indentation for wrapped lines

### Color Consistency
- The robot emoji and response text use `get_llm_color()`
- Maintains visual consistency with non-streaming responses
- System messages use `get_system_color()`

## Testing
Run the included test script to verify streaming functionality:
```bash
python test_streaming.py
```

## Future Improvements
1. Extract token usage directly from streaming chunks when LiteLLM adds support
2. Add progress indicators for very long responses
3. Support for interrupting streaming responses
4. Optimize the temporary non-streaming call for cost calculation