# Episodic User's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Mode Switching](#mode-switching)
4. [Getting Started](#getting-started)
5. [Basic Usage](#basic-usage)
6. [Topic Management](#topic-management)
7. [Knowledge Base (RAG)](#knowledge-base-rag)
8. [Web Search](#web-search)
9. [Muse Mode](#muse-mode)
10. [Configuration](#configuration)
11. [Advanced Features](#advanced-features)
12. [Experimental Features](#experimental-features)

## Introduction

Episodic is a conversational memory system that creates persistent, navigable conversations with language models. It automatically organizes conversations into topics, compresses old topics to manage context, and provides tools for searching both local knowledge and the web.

### Key Features
- **Persistent Conversations**: All conversations are stored in a local SQLite database
- **Automatic Topic Detection**: Conversations are automatically segmented into topics
- **Context Management**: Old topics are compressed to stay within LLM context limits
- **Knowledge Base (RAG)**: Index documents and search them during conversations
- **Web Search**: Search the web for current information
- **Multi-Model Support**: Works with OpenAI, Anthropic, Google, Ollama, and more

## Core Concepts

### Conversation DAG
Conversations are stored as a Directed Acyclic Graph (DAG), though currently only linear conversations are implemented. Each message exchange creates nodes in the graph.

### Topics
Topics are semantic segments of conversation. Episodic automatically detects when the conversation shifts to a new topic and creates boundaries. Topics can be:
- **Ongoing**: Currently active topic (no end boundary yet)
- **Closed**: Completed topic with defined start and end
- **Compressed**: Summarized topic to save context space

### Context Window
LLMs have limited context windows. Episodic manages this by:
1. Including recent messages in full
2. Including compressed summaries of older topics
3. Showing context usage percentage

## Mode Switching

Episodic operates in two primary modes that you can toggle between:

### Chat Mode (Default)
**Standard LLM conversation mode** - your messages go directly to the AI model for normal conversation.

```bash
/chat    # Switch to chat mode
```

- Normal conversational AI interaction
- Uses your configured chat model
- All standard Episodic features (topics, compression, etc.) available
- Default mode when starting Episodic

### Muse Mode 
**Web search synthesis mode** - like Perplexity AI, your messages become web search queries that are synthesized into comprehensive answers.

```bash
/muse    # Switch to muse mode  
```

- All input becomes web search queries
- Results are synthesized using AI into comprehensive answers
- Includes citations and source links
- Maintains conversational context for follow-up questions
- Great for research and current information

### Mode Status
```bash
# Check current mode (both commands show status when called without arguments)
/muse    # Shows if muse mode is active
/chat    # Shows if chat mode is active
```

### Quick Mode Examples

**Chat Mode Example:**
```bash
> /chat
💬 Chat mode active - conversation with AI

> Explain quantum computing
🤖 Quantum computing is a revolutionary approach to computation that...
```

**Muse Mode Example:**
```bash
> /muse  
🎭 Muse mode active - web search synthesis

> Latest developments in quantum computing
🔍 Searching web for: latest developments quantum computing
📚 Found 8 relevant sources
✨ Based on recent developments, here are the latest advances:

1. **IBM's 1000-qubit Condor processor** (December 2023)...
2. **Google's quantum error correction breakthrough**...

📄 Sources: Nature, IBM Research, Google AI...
```

### Configuration
Both modes respect your model and parameter configurations:
- **Chat mode**: Uses your `/model chat` setting
- **Muse mode**: Uses your `/model synthesis` setting (or chat model if not set)

## Getting Started

### Installation
```bash
pip install -e .
```

### First Run
```bash
python -m episodic
> /init  # Initialize the database
```

### Basic Conversation
Just start typing! No command prefix needed:
```
> What is the capital of France?
🤖 The capital of France is Paris.

> Tell me more about it
🤖 Paris is the capital and largest city of France...
```

## Basic Usage

### Starting a Conversation
Simply type your message and press Enter. Episodic will:
1. Detect if this is a new topic
2. Query the configured LLM
3. Stream the response
4. Save everything to the database

### Interrupting Responses
You can interrupt long streaming responses using Ctrl-C:
- **Single Ctrl-C**: Interrupts the current response and returns to the prompt
- **Double Ctrl-C** (within 1 second): Exits Episodic

When you interrupt a response:
- The partial response is saved to the conversation history
- You'll see "⚡ Response interrupted" message
- The system returns cleanly to the prompt for your next input

### Navigation Commands

| Command | Description |
|---------|-------------|
| `/list` | Show recent messages |
| `/show <node_id>` | Show details of a specific node |
| `/head` | Show current conversation head |
| `/ancestry` | Show the conversation history |

### Model Selection
```bash
/model  # Show all four models in use
/model list  # Show available models with pricing
/model chat gpt-4  # Set chat (main conversation) model
/model detection ollama/llama3  # Set topic detection model
/model compression gpt-3.5-turbo  # Set compression model
/model synthesis claude-3-haiku  # Set web synthesis model
```

## Topic Management

### Automatic Topic Detection
Episodic uses semantic drift detection to identify topic changes. When drift exceeds a threshold (default 0.9), a new topic is created.

### Topic Commands

| Command | Description |
|---------|-------------|
| `/topics` | List all topics |
| `/topics rename` | Rename ongoing topics |
| `/topics compress` | Manually compress current topic |
| `/topics stats` | Show topic statistics |

### Topic Detection Methods

#### 1. Sliding Window (Default)
Compares groups of messages to detect semantic shifts:
```bash
/set topic-window true
/set window-size 3  # Compare 3-message windows
/set drift 0.9    # Threshold for topic change
```

#### 2. Hybrid Detection (Experimental)
Combines multiple signals:
```bash
/set hybrid-topics true
```
Signals include:
- Semantic drift (embedding similarity)
- Explicit keywords ("let's talk about", "changing topics")
- Domain-specific keywords
- Message gaps (time between messages)
- Conversation flow patterns

#### 3. Manual Detection
Disable automatic detection and control topics manually:
```bash
/set topic-auto false
/topics index 5  # Manually run detection on last 5 messages
```

### Topic Compression
When topics end, they can be automatically compressed:
```bash
/set comp-auto true
/set comp-min 10  # Minimum messages before compression
/compression stats  # View compression queue
```

## Knowledge Base (RAG)

### Enabling RAG
```bash
/rag on  # Enable RAG functionality
```

### Indexing Documents
```bash
/index document.txt  # Index a file
/index --text "Important information to remember"  # Index text directly
/i document.pdf  # Short form (PDFs require additional dependencies)
```

### Searching
```bash
/search climate change  # Search the knowledge base
/s climate change      # Short form
```

### Document Management
```bash
/docs              # List all indexed documents
/docs show 1       # Show content of document ID 1
/docs remove 1     # Remove document ID 1
/docs clear        # Remove all documents
```

### RAG Configuration
```bash
/set rag-auto true         # Auto-search on each message
/set rag-threshold 0.7     # Minimum relevance score
/set rag-results 5         # Max results to include
/set rag-chunk 500         # Words per document chunk
```

## Web Search

### Mode Switching
```bash
/muse     # Switch to muse mode - all input becomes web searches
/chat     # Switch to chat mode - normal LLM conversation
```

Episodic has two main modes that you can toggle between:
- **Chat Mode**: Standard LLM conversation (default)
- **Muse Mode**: Web search synthesis - like Perplexity AI

### Using Muse Mode
```bash
# Simply type your questions - they become web searches automatically
latest AI developments
what's new in quantum computing
```

### Web Search Features
- **Auto-Enhancement**: Automatically search when local knowledge insufficient
- **Result Indexing**: Web results can be indexed into RAG
- **Synthesis**: Combine multiple results into comprehensive answer
- **Automatic Fallback**: Try multiple providers if one fails

### Provider Configuration (New Shorter Syntax)
```bash
# Set single provider
/set web.provider google

# Set provider order for automatic fallback
/set web.providers google,bing,duckduckgo

# Use only free providers
/set web.providers duckduckgo,searx

# Configure fallback behavior
/set web.fallback true              # Enable automatic fallback
/set web.fallback_cache_minutes 5   # Cache working provider
```

### Other Configuration Options
```bash
# Shorter syntax with web. prefix
/set web.enabled true           # Enable web search
/set web.max_results 5          # Number of results
/set web.synthesize true        # Synthesize results
/set web.extract true           # Extract page content
/set web.cache 3600             # Cache duration (seconds)
/set web.rate_limit 60          # Max searches per hour

# Legacy syntax still works
/set web-auto true              # Auto-search when needed
/set web-results 5              # Number of results
```

### Search Providers
1. **DuckDuckGo** (Default, no API key needed)
   - Always available as fallback
   - No configuration required
   
2. **Google** (Requires credentials)
   - Set `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`
   - 100 queries/day on free tier
   
3. **Bing** (Requires Azure API key)
   - Set `BING_API_KEY`
   - Better for certain types of queries
   
4. **Searx** (Privacy-focused, can self-host)
   - Configure with `/set searx_instance_url`
   - No API key needed

### How Fallback Works
When you search, Episodic will:
1. Try providers in your configured order
2. Skip providers without proper credentials
3. Automatically fallback on quota/auth errors
4. Cache the working provider for faster subsequent searches
5. Always ensure DuckDuckGo is available as last resort

Example flow with `/set web.providers google,bing,duckduckgo`:
- First tries Google → If quota exceeded
- Falls back to Bing → If not configured
- Falls back to DuckDuckGo → Always works

## Muse Mode

Muse mode transforms Episodic into a Perplexity-like conversational web search tool where all input is automatically treated as web search queries.

### Enabling Muse Mode
```bash
/muse  # Enable muse mode
/chat  # Return to normal chat mode
```

### How Muse Mode Works
1. All your input becomes web search queries
2. Multiple web results are fetched and synthesized
3. Answers include citations and sources
4. Follow-up questions maintain context

### Example Usage
```
> /muse
🎭 Muse mode ENABLED

> latest breakthroughs in quantum computing
[Web search synthesis with citations]

> tell me more about the IBM announcement
[Contextual follow-up with new search]
```

## Configuration

### Viewing Settings
```bash
/set  # Show all current settings
/set <param> <value>  # Set a parameter
/verify  # Verify configuration integrity
/cost  # Show session costs
/mset  # Show all model parameters for all contexts
/config-docs  # Show parameter documentation
```

### Common Settings

#### Display Settings
```bash
/set stream true              # Enable response streaming
/set stream-rate 15          # Words per second
/set color full              # Color mode (none/basic/full)
/set wrap true               # Enable text wrapping
```

#### Response Formatting
```bash
# Global response style (affects length and detail level)
/style                       # Show current style
/style concise               # Brief, direct responses
/style standard              # Clear, well-structured responses
/style comprehensive         # Thorough, detailed responses
/style custom                # Use model-specific max_tokens

# Global response format (affects presentation structure)
/format                      # Show current format
/format paragraph            # Flowing prose with markdown headers
/format bullet-points        # Bullet points and lists
/format mixed                # Mix of paragraphs and bullet points
/format academic             # Formal academic style with citations
```

#### Model Parameters
```bash
/mset  # Show all model parameters
/mset chat  # Show chat model parameters
/mset chat.temperature 0.7  # Set chat temperature
/mset detection.temperature 0.3  # Set detection temperature
/mset compression.max_tokens 500  # Set compression max tokens
/mset synthesis.temperature 0.5  # Set synthesis temperature
```

#### Performance
```bash
/set cache true              # Enable context caching
/set benchmark true          # Enable performance tracking
/benchmark                   # Show performance stats
```

### Configuration Files
- `~/.episodic/config.json` - User configuration
- `~/.episodic/config.default.json` - Default values reference

### Environment Variables
All configuration values can be set via environment variables:
```bash
# Web Search
export EPISODIC_WEB_PROVIDER=google
export GOOGLE_API_KEY=your-key
export GOOGLE_SEARCH_ENGINE_ID=your-id
export EPISODIC_WEB_ENABLED=true
export EPISODIC_WEB_AUTO=true

# RAG
export EPISODIC_RAG_ENABLED=true
export EPISODIC_RAG_AUTO=true
export EPISODIC_RAG_THRESHOLD=0.7

# Topic Detection
export EPISODIC_TOPIC_DETECTION_MODEL=ollama/llama3
export EPISODIC_TOPIC_AUTO=true
export EPISODIC_TOPIC_MIN=8

# Display
export EPISODIC_COLOR_MODE=full
export EPISODIC_STREAM_RATE=20
export EPISODIC_SHOW_COST=true

# Core
export EPISODIC_DEBUG=true
export EPISODIC_CACHE=true
```

## Advanced Features

### System Prompts
```bash
/prompt              # List available prompts
/prompt creative     # Switch to creative prompt
/prompt show         # Show current prompt content
/prompt custom "Be concise"  # Set custom prompt
```

### Session Management
```bash
/cost  # Show session token usage and costs
/summary  # Summarize recent conversation
/visualize  # Start web-based conversation visualizer
```

### Reset Configuration
```bash
/reset              # Show reset options
/reset all          # Reset all settings to defaults
/reset <param>      # Reset specific parameter
/reset all --save   # Reset and save to config file
```

## Experimental Features

### Alternative Topic Detection Methods

#### 1. Embedding-Based Drift Detection
The ML module (`episodic/ml/`) contains experimental embedding-based drift detection:
- Uses sentence transformers for embeddings
- Calculates cosine similarity between messages
- Includes peak detection for topic boundaries

#### 2. Keyword-Based Detection
The `TransitionDetector` in `episodic/topics/keywords.py` identifies:
- Explicit transition phrases ("let's move on to")
- Question patterns indicating topic shifts
- Domain changes (technical → casual)

#### 3. Boundary Analysis
When enabled, analyzes where topics actually change:
```bash
/set topic-boundaries true
/set topic-llm-analysis true  # Use LLM for analysis
```

### Hybrid Scoring System
Combines multiple signals with configurable weights:
```python
"hybrid_topic_weights": {
    "semantic_drift": 0.6,      # Embedding similarity
    "keyword_explicit": 0.25,    # Transition phrases
    "keyword_domain": 0.1,       # Domain shifts
    "message_gap": 0.025,        # Time gaps
    "conversation_flow": 0.025   # Question/answer patterns
}
```

### Running Topic Prediction
```bash
/set running_topic_guess true  # Not implemented yet
```
Planned feature to show topic predictions in real-time.

## Common Workflows

### Research Assistant
```bash
# Index research papers
/rag on
/index paper1.pdf
/index paper2.pdf

# Enable web search for current info
/muse
/set web-auto true                  # Enable automatic web search fallback

# Ask questions - will search both local docs and web
What are the latest developments in quantum computing?
```

### Long Conversation Management
```bash
# Enable automatic topic management
/set automatic_topic_detection true  # or: /set topic-auto true
/set auto_compress_topics true       # or: /set comp-auto true
/set show_topics true                # See topic evolution

# Monitor context usage
/set show_cost true
```

### Offline Usage with Ollama
```bash
# Use local models for all contexts
/model chat ollama/llama3
/model detection ollama/llama3
/model compression ollama/llama3
/model synthesis ollama/llama3
```

## Scripting and Automation

### Using Scripts

Execute commands from a script file to automate repetitive tasks:
```bash
/script scripts/my-workflow.txt
```

Script files are plain text files that can contain:
- **Commands**: Lines starting with `/`
- **Chat messages**: Lines without a prefix
- **Comments**: Lines starting with `#`
- **Empty lines**: Ignored

### Common Use Cases

#### 1. Parameter Profiles
Create different configuration profiles for various scenarios:

**scripts/dev-profile.txt:**
```bash
# Development settings - fast iteration
/set debug true
/set stream false
/set cost true
/set main.temperature 0.7
/set main.max_tokens 500
/model chat gpt-3.5-turbo
```

**scripts/creative-profile.txt:**
```bash
# Creative writing settings
/set main.temperature 1.2
/set main.max_tokens 2000
/set main.top_p 0.95
/model chat gpt-4
/prompt creative
```

#### 2. Test Scenarios
Create reproducible test cases:

**scripts/test-topics.txt:**
```bash
/init --erase
/set min_messages_before_topic_change 2
Tell me about space exploration.
What are the challenges of Mars colonization?
Now let's discuss Italian cooking.
What's your favorite pasta recipe?
/topics
```

#### 3. Daily Workflow
Automate your common setup:

**scripts/daily-setup.txt:**
```bash
# My daily research setup
/rag on
/muse
/set web-auto true
/set topics true
/model chat gpt-4
/muse
```

### Saving Sessions

Save your current session's commands for later replay:
```bash
/save my-session        # Saves to scripts/my-session.txt
/save profiles/research # Saves to scripts/profiles/research.txt
```

The saved script includes all commands from your session (except the `/save` command itself).

### Tips

- **Nested directories**: Organize scripts in subdirectories (e.g., `scripts/profiles/`, `scripts/tests/`)
- **Combine scripts**: Load base settings then specific overrides
- **Version control**: Track your scripts in git for team sharing
- **No nesting**: Scripts cannot call other scripts (not supported)

## Troubleshooting

### Common Issues

1. **"Config file corrupted"**
   - Check `~/.episodic/config.json` for valid JSON
   - Use `/reset all` to restore defaults

2. **Topic detection too sensitive/insensitive**
   - Adjust `/set drift_threshold 0.9` (higher = less sensitive)
   - Try `/set min_messages_before_topic_change 8`

3. **Context window exceeded**
   - Enable compression: `/set auto_compress_topics true`
   - Reduce context depth: `/set depth 3`

### Debug Mode
```bash
/set debug true  # Enable detailed debug output
```

## Deprecated Commands

The following commands are deprecated but still work with warnings. They will be removed in v0.5.0:

### Topic Commands (use `/topics` instead)
- `/rename-topics` → `/topics rename`
- `/compress-current-topic` → `/topics compress`
- `/index` → `/topics index` (Note: `/index` for RAG still works)
- `/topic-scores` → `/topics scores`

### Compression Commands (use `/compression` instead)
- `/compression-stats` → `/compression stats`
- `/compression-queue` → `/compression queue`
- `/api-stats` → `/compression api-stats`
- `/reset-api-stats` → `/compression reset-api`

## Architecture Notes

### Database Schema
- **nodes**: Stores all messages
- **topics**: Topic boundaries and metadata  
- **compressions_v2**: Compressed topic summaries
- **rag_documents**: Indexed documents
- **configuration**: Key-value settings

### Module Organization
- `episodic/cli.py` - Main CLI interface
- `episodic/conversation.py` - Conversation management
- `episodic/topics/` - Topic detection implementations
- `episodic/rag.py` - Knowledge base functionality
- `episodic/web_search.py` - Web search integration

## Contributing

Episodic has several experimental features that could use improvement:

1. **Non-linear DAG conversations** - Currently only linear
2. **Running topic prediction** - Real-time topic guessing
3. **Embedding provider alternatives** - Currently uses sentence-transformers
4. **Additional web search providers** - More search engines
5. **Topic detection improvements** - Better algorithms

See `ADAPTIVE_TOPIC_DETECTION_PLAN.md` for planned improvements.