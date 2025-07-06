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
Configure parameters
```bash
/set                         # Show all settings
/set <parameter>             # Show specific value
/set <parameter> <value>     # Set value (session only)

# Common settings (also support short aliases):
/set debug on                # Enable debug mode
/set cache off              # Disable caching
/set stream true            # Enable streaming
/set wrap true              # Enable text wrapping
/set cost on                # Show cost info
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

### /model-params
View/set model parameters
```bash
/model-params               # Show all parameters
/model-params main          # Show main params only
/set main.temp 0.8          # Set temperature
/set topic.max 100          # Set max tokens
```

### /config-docs
Show configuration documentation
```bash
/config-docs                # List all params with docs
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
Select language model
```bash
/model                      # Show available models
/model gpt-4                # Switch to specific model
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

Use dot notation to set model-specific parameters:

```bash
# Main conversation parameters
/set main.temp 0.7          # Temperature
/set main.max 2000          # Max tokens
/set main.top_p 0.9         # Top-p sampling

# Topic detection parameters
/set topic.temp 0.3         # Lower for consistency
/set topic.max 50           # Minimal tokens needed

# Compression parameters
/set comp.temp 0.5          # Balanced temperature
/set comp.presence 0.1      # Reduce repetition
```

## Configuration Examples

### Research Mode
```bash
/rag on
/websearch on
/set web_search_auto_enhance true
/set rag_auto_search true
```

### Long Conversation Mode
```bash
/set automatic_topic_detection true
/set auto_compress_topics true
/set show_topics true
/set show_cost true
```

### Offline Mode
```bash
/model ollama/llama3
/set topic_detection_model ollama/llama3
/set compression_model ollama/llama3
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
5. **Settings**: Changes via /set are session-only unless using /reset --save