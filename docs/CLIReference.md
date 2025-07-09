# CLI Reference

This document provides a comprehensive reference for all Episodic CLI commands.

## Starting the Application

```bash
# Interactive mode (default)
python -m episodic

# Execute a script non-interactively
python -m episodic --execute scripts/my-script.txt
python -m episodic -e scripts/my-script.txt

# Specify a model at startup
python -m episodic --model gpt-4
python -m episodic -m ollama/llama3

# Disable streaming output
python -m episodic --no-stream

# Combine options
python -m episodic -m gpt-4 -e scripts/test.txt
```

## Command Structure

In Episodic's talk mode:
- Type messages without any prefix to chat with the LLM
- Use "/" prefix for commands

## Navigation Commands

### /init
Initialize or reset the database
```bash
/init              # Initialize database
/init --erase      # Erase existing database and reset
```

### /add
Add a new message node
```bash
/add Your message here
/add --parent 01 Your message here  # Add as child of specific node
```

### /show
Display node details
```bash
/show <node_id>    # Show specific node
```

### /print
Print node content
```bash
/print             # Print current node
/print <node_id>   # Print specific node
```

### /head
Show or change current node
```bash
/head              # Show current head
/head <node_id>    # Set new head
```

### /list
List recent nodes
```bash
/list              # List recent nodes
/list --count 10   # Show 10 most recent
```

### /ancestry
Show conversation thread
```bash
/ancestry <node_id> # Show ancestry of node
```

## Configuration Commands

### /set
Legacy settings command (still works)
```bash
/set                         # Show all settings
/set <parameter>             # Show specific value
/set <parameter> <value>     # Set value

# Common settings:
/set debug true              # Enable debug mode
/set cache false             # Disable caching
/set stream true             # Enable streaming
/set wrap true               # Enable text wrapping
/set cost true               # Show cost info
```

### /reset
Reset parameters to defaults
```bash
/reset                      # Show reset help
/reset all                  # Reset all to defaults
/reset <param>              # Reset specific param
/reset all --save           # Reset and save to file
```

### /verify
Verify configuration integrity
```bash
/verify                     # Check database and config
```

### /cost
Show session costs
```bash
/cost                       # Display token usage and costs
```

### /mset
View/set model parameters
```bash
/mset                       # Show all model parameters
/mset chat                  # Show chat model params
/mset detection             # Show detection model params
/mset chat.temperature 0.8  # Set chat temperature
/mset detection.max_tokens 100  # Set detection max tokens
/mset compression.temperature 0.5  # Set compression temperature
/mset synthesis.top_p 0.9   # Set synthesis nucleus sampling
```

### Configuration Documentation
Access configuration docs through settings command:
```bash
/config-docs                # List all params with documentation
```

## Topic Management

### /topics
Unified topic management
```bash
/topics                     # List topics (default)
/topics list --all          # Show all topics
/topics rename              # Rename ongoing topics
/topics compress            # Compress current topic
/topics index 5             # Manual detection on last 5
/topics scores              # Show detection scores
/topics stats               # Topic statistics
```

## Knowledge Base (RAG)

### /rag
Enable/disable RAG
```bash
/rag                        # Show RAG status
/rag on                     # Enable RAG
/rag off                    # Disable RAG
```

### /index, /i
Index documents
```bash
/index document.txt         # Index a file
/i research.pdf             # Short form
/index --text "Important info"  # Index text directly
```

### /search, /s
Search knowledge base
```bash
/search climate change      # Search documents
/s climate change           # Short form
```

### /docs
Manage documents
```bash
/docs                       # List all documents
/docs list                  # Same as above
/docs show 1                # Show document content
/docs remove 1              # Remove document
/docs clear                 # Remove all documents
/docs clear web             # Remove only web-sourced docs
```

## Web Search

### /websearch, /ws
Search the web
```bash
/websearch latest AI news   # Search the web
/ws latest AI news          # Short form
/websearch on               # Enable web search
/websearch off              # Disable web search
/websearch config           # Show configuration
/websearch stats            # Show search statistics
/websearch cache clear      # Clear search cache
```

## Compression

### /compression
Unified compression management
```bash
/compression                # Show stats (default)
/compression stats          # Compression statistics
/compression queue          # Show pending jobs
/compression compress       # Manual compression
/compression api-stats      # API usage stats
/compression reset-api      # Reset API counters
```

### /compress
Manual compression
```bash
/compress                   # Compress to current head
/compress --node <id>       # Compress to specific node
/compress --dry-run         # Preview only
```

## Conversation Commands

### /model
Manage language models for all contexts
```bash
/model                      # Show all four models in use
/model list                 # Show available models with pricing
/model chat gpt-4           # Set chat (main) model
/model detection ollama/llama3  # Set topic detection model
/model compression gpt-3.5-turbo  # Set compression model
/model synthesis claude-3-haiku  # Set web synthesis model
```

### /prompt, /prompts
Manage system prompts
```bash
/prompt                     # List available prompts
/prompt creative            # Switch to prompt
/prompt show                # Show current prompt
/prompt custom "Be brief"   # Set custom prompt
```

### /summary
Summarize conversation
```bash
/summary                    # Last 5 messages
/summary 10                 # Last 10 messages
/summary all                # Entire conversation
```

## Mode Commands

### /muse
Enable Perplexity-like web search mode
```bash
/muse                       # Enable muse mode (all input → web search)
/muse on                    # Same as above
/muse off                   # Disable muse mode
```

### /chat
Return to normal chat mode
```bash
/chat                       # Enable chat mode (normal LLM conversation)
/chat on                    # Same as above
/chat off                   # Disable chat mode (enables muse)
```

## Utility Commands

### /visualize
Generate conversation graph
```bash
/visualize                  # Open in browser
/visualize --output out.html # Save to file
/visualize --no-browser     # Don't open browser
/visualize --port 5001      # Custom port
```

### /benchmark
Show performance stats
```bash
/benchmark                  # Display benchmarks
```

### /help
Show available commands
```bash
/help                       # List all commands
```

### /exit, /quit
Exit the application
```bash
/exit                       # Exit Episodic
/quit                       # Same as exit
```

## Model Parameters

Use /mset command to manage model-specific parameters:

```bash
# Chat (main conversation) parameters
/mset chat.temperature 0.7      # Temperature
/mset chat.max_tokens 2000      # Max tokens
/mset chat.top_p 0.9            # Top-p sampling

# Topic detection parameters
/mset detection.temperature 0.3  # Lower for consistency
/mset detection.max_tokens 50    # Minimal tokens needed

# Compression parameters
/mset compression.temperature 0.5     # Balanced temperature
/mset compression.presence_penalty 0.1 # Reduce repetition

# Web synthesis parameters
/mset synthesis.temperature 0.7  # Balanced for web synthesis
/mset synthesis.max_tokens 1000  # Good length for summaries
```

## Configuration Examples

### Research Mode
```bash
/rag on
/websearch on
/set web_search_auto_enhance true  # or: /set web-auto true
/set rag_auto_search true          # or: /set rag-auto true
```

### Long Conversation Mode
```bash
/set automatic_topic_detection true  # or: /set topic-auto true
/set auto_compress_topics true       # or: /set comp-auto true
/set show_topics true
/set show_cost true
```

### Offline Mode
```bash
/model chat ollama/llama3
/model detection ollama/llama3
/model compression ollama/llama3
/model synthesis ollama/llama3
/rag off
/websearch off
```

### Debug Mode
```bash
/set debug true
/set show_drift true
/set benchmark true
/set benchmark_display true
```

## Script Execution

Scripts can contain both messages and commands:

```bash
# Create a script file (example.txt):
/init
/model gpt-4
What is quantum computing?
/topics
/cost

# Execute the script:
python -m episodic -e example.txt
```

## Short Node IDs

Episodic uses 2-character alphanumeric IDs for easy reference:

```bash
> /add Hello world
Added node a1 (UUID: 3a7e46c9-...)

> /show a1
Node ID: a1
Content: Hello world

> /ancestry a1
a1: Hello world
```

## Tips

1. **Tab Completion**: Use Tab for command completion
2. **History**: Use Up/Down arrows for command history
3. **Shortcuts**: Many commands have short forms (/s for /search, /ws for /websearch)
4. **Help**: Type /help for quick command reference
5. **Settings**: Settings persist in the database automatically

## Deprecated Commands

The following commands are deprecated but still work with warnings (will be removed in v0.5.0):

| Deprecated Command | Use Instead |
|-------------------|-------------|
| `/rename-topics` | `/topics rename` |
| `/compress-current-topic` | `/topics compress` |
| `/topic-scores` | `/topics scores` |
| `/compression-stats` | `/compression stats` |
| `/compression-queue` | `/compression queue` |
| `/api-stats` | `/compression api-stats` |
| `/reset-api-stats` | `/compression reset-api` |

Note: `/index` for topic detection is deprecated, but `/index` for RAG document indexing remains active.