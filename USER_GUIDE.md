# Episodic User's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Topic Management](#topic-management)
6. [Knowledge Base (RAG)](#knowledge-base-rag)
7. [Web Search](#web-search)
8. [Muse Mode](#muse-mode)
9. [Configuration](#configuration)
10. [Advanced Features](#advanced-features)
11. [Experimental Features](#experimental-features)

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

### Enabling Web Search
```bash
/websearch on  # Enable web search
```

### Searching the Web
```bash
/websearch latest AI developments  # Search the web
/ws latest AI developments        # Short form
```

### Web Search Features
- **Auto-Enhancement**: Automatically search when local knowledge insufficient
- **Result Indexing**: Web results can be indexed into RAG
- **Synthesis**: Combine multiple results into comprehensive answer

### Configuration
```bash
/set web-auto true      # Auto-search when needed
/set web-results 5      # Number of results
/set web-synthesize true        # Synthesize results
/set web-extract true   # Extract page content
```

### Search Providers
1. **DuckDuckGo** (Default, no API key needed)
2. **Searx** (Privacy-focused, can self-host)
3. **Google** (Requires API key and search engine ID)
4. **Bing** (Requires Azure API key)

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
/settings  # Show all current settings (or /set)
/settings show  # Same as above
/settings set <param> <value>  # Set a parameter
/settings verify  # Verify configuration integrity
/settings cost  # Show session costs
/settings params  # Show model parameters
/settings docs  # Show parameter documentation
```

### Common Settings

#### Display Settings
```bash
/set stream true              # Enable response streaming
/set stream-rate 15          # Words per second
/set color full              # Color mode (none/basic/full)
/set wrap true               # Enable text wrapping
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
export EPISODIC_TOPIC_MODEL=ollama/llama3
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
/websearch on
/set web_search_auto_enhance true

# Ask questions - will search both local docs and web
What are the latest developments in quantum computing?
```

### Long Conversation Management
```bash
# Enable automatic topic management
/set automatic_topic_detection true
/set auto_compress_topics true
/set show_topics true  # See topic evolution

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