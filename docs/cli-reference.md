# CLI Reference

This document provides a comprehensive reference for all Episodic CLI commands.

## Starting the Application

```bash
# Interactive mode (default)
python -m episodic
# Database automatically created at ~/.episodic/episodic.db on first run

# Execute a script non-interactively
python -m episodic --execute scripts/my-script.txt
python -m episodic -e scripts/my-script.txt

# Specify a model at startup
python -m episodic --model gpt-4.1-2025-04-14
python -m episodic -m ollama/llama3

# Disable streaming output
python -m episodic --no-stream

# Combine options
python -m episodic -m gpt-4.1-2025-04-14 -e scripts/test.txt
```

## Command Structure

In Episodic's talk mode:
- Type messages without any prefix to chat with the LLM
- Use "/" prefix for commands

## Keyboard Shortcuts

### Tab Completion
- **Tab completion is enabled by default** - Press Tab to complete commands, parameters, and values
- **Commands**: Type `/` and press Tab to see all available commands
- **Parameters**: Type `/set ` and press Tab to see all configuration parameters
- **Models**: Type `/model chat ` and press Tab to see available models
- **Files**: Type `/load ` and press Tab to browse markdown files for import
- **Type hints**: Parameters show their expected value type (boolean, number, choice, etc.)
- To disable: `/set enable-tab-completion false`

### Interrupting Responses (Ctrl-C)
- **During streaming**: Press Ctrl-C once to interrupt the LLM response
- **At the prompt**: Press Ctrl-C twice quickly (within 1 second) to exit Episodic
- Interrupted responses are saved with a "[Response interrupted by user]" marker

### User Input Display
- User input is displayed in a styled box after pressing Enter
- Box automatically sizes to fit content (up to 80 characters wide)
- Long inputs wrap within the box
- Disable with: `/set show-input-box false`
- Use ASCII boxes for basic terminals: `/set use-unicode-boxes false`

### Exit Options
- `/exit` or `/quit` - Normal exit with cleanup
- `Ctrl-D` - Exit immediately (EOF)
- `Ctrl-C Ctrl-C` - Double Ctrl-C to exit from anywhere

## Interface Mode Commands

### /simple
Switch to simple mode - streamlined interface with essential commands only
```bash
/simple            # Switch to simple mode
```

### /advanced
Switch to advanced mode - full access to all 50+ commands
```bash
/advanced          # Switch to advanced mode
```

### /theme
Change the color theme for the interface
```bash
/theme             # Show available themes and current theme
/theme dark        # Switch to dark theme
/theme light       # Switch to light theme
/theme ocean       # Switch to ocean theme
```

### /detail
Set the detail level for responses (affects all modes)
```bash
/detail            # Show current detail level
/detail minimal    # Brief, concise responses
/detail moderate   # Standard balanced responses
/detail detailed   # Comprehensive responses
/detail maximum    # Most detailed responses
```

## Navigation Commands

### /init
Initialize or reset the database (rarely needed - database auto-initializes)
```bash
/init              # Initialize database (if not already initialized)
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
Configure various parameters
```bash
/set                         # Show common settings
/set all                     # Show all settings
/set <parameter>             # Show specific value
/set <parameter> <value>     # Set value

# Common settings:
/set cost true               # Show API costs
/set depth 10                # Conversation context depth
/set stream false            # Disable streaming output
/set topics true             # Show topic information
/set color-mode basic        # Switch to basic colors (full/basic/none)

# Other useful settings:
/set debug true              # Enable debug mode
/set cache false             # Disable prompt caching
/set wrap true               # Enable text wrapping
/set stream-rate 25          # Words per second for streaming
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
/mset chat.temperature default  # Reset to default value
/mset detection.top_p default   # Remove override, use default
```

### /style
Set global response style (affects length and detail level)
```bash
/style                      # Show current style
/style concise              # Brief, direct responses (1-2 sentences when possible)
/style standard             # Clear, well-structured responses with appropriate detail
/style comprehensive        # Thorough, detailed responses with examples and context
/style custom               # Use model-specific max_tokens settings for fine control
```

### /format
Set global response format (affects presentation structure)
```bash
/format                     # Show current format
/format paragraph           # Flowing prose in paragraph form with markdown headers
/format bulleted       # Bullet points and lists for all information
/format mixed               # Mix of paragraphs and bullet points as appropriate
/format academic            # Formal academic style with proper citations [Source N]
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
/topics list all            # Show all topics
/topics rename              # Rename ongoing topics
/topics compress            # Compress current topic
/topics index 5             # Manual detection on last 5
/topics scores              # Show detection scores
/topics stats               # Topic statistics
/topics reanalyze           # Re-detect topics using full conversation context
/topics reanalyze apply     # Re-detect and save to database
/topics reanalyze verbose   # Re-detect with detailed merge history
```

The `reanalyze` subcommand uses hierarchical clustering with elbow detection to find natural topic boundaries across the entire conversation, rather than the real-time sliding window approach used during chat.

### Topic Segmentation Settings

Configure neural topic boundary detection:

```bash
# Granularity controls how many topic boundaries are detected
/set topic-granularity fine      # Many boundaries, micro-shifts (threshold: 0.3)
/set topic-granularity medium    # Balanced detection (threshold: 0.5, default)
/set topic-granularity coarse    # Major themes only (threshold: 0.7)

# Temperature controls prediction confidence calibration
/set topic-temperature 1.0       # Default (no scaling)
/set topic-temperature 0.7       # Sharper predictions (more confident)
/set topic-temperature 1.5       # Softer predictions (less confident)
```

**Granularity Levels:**
| Level | Threshold | Description |
|-------|-----------|-------------|
| `fine` | 0.3 | Detects subtle topic shifts, many boundaries |
| `medium` | 0.5 | Balanced detection (recommended default) |
| `coarse` | 0.7 | Only major theme changes, few boundaries |

**Temperature Scaling:**
- `T < 1.0`: Sharper predictions (model is more confident in boundary decisions)
- `T = 1.0`: No scaling (default)
- `T > 1.0`: Softer predictions (model is less confident, more uniform probabilities)

Temperature calibration is useful for fine-tuning when the model is under-confident or over-confident on your specific conversation style.

## Markdown Operations

### /save
Export conversations to markdown
```bash
/save                       # Export current topic
/save 3                     # Export topic 3
/save 1-5 summary.md        # Export topics 1-5 to file
/save all backup.md         # Export entire conversation
```

### /load
Import markdown conversations
```bash
/load conversation.md       # Import markdown file
/load exports/backup.md     # Import from exports directory
/load ~/Documents/chat.md   # Import from path
```

### /files, /ls
List markdown files
```bash
/files                      # List markdown files in current dir
/ls                         # Unix-style alias
/files exports              # List files in exports directory
/ls ~/Documents             # List files in specific path
```

## Knowledge Base (RAG)

### /rag
Enable/disable RAG and view statistics
```bash
/rag                        # Show RAG statistics (default)
/rag on                     # Enable RAG
/rag off                    # Disable RAG
/rag stats                  # Show RAG statistics
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

### /muse
Enable muse mode for synthesized web search answers (like Perplexity)
```bash
/muse                       # Switch to muse mode - all input becomes web searches
                           # In this mode, everything you type is treated as a search query
                           # Results are fetched from the web and synthesized into answers
```

### /web
Configure web search providers
```bash
/web                        # Show current provider and status
/web list                   # List all available providers
/web provider google        # Set primary provider
/web provider duckduckgo    # Set primary provider
/web reset                  # Reset to default configuration
```

**Configuration via settings (alternative syntax):**
```bash
# Set single provider
/set web.provider google

# Set provider order for automatic fallback
/set web.providers google,bing,duckduckgo

# Configure fallback behavior
/set web.fallback true              # Enable automatic fallback
/set web.fallback_cache_minutes 5   # Cache working provider for 5 minutes

# Other web search settings
/set web.enabled true               # Enable web search
/set web.max_results 5              # Number of results to retrieve
/set web.cache 3600                 # Cache results for 1 hour
/set web.rate_limit 60              # Max searches per hour
```

**Available Providers:**
- `duckduckgo` - Free, no API key required (default)
- `google` - Requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID
- `bing` - Requires BING_API_KEY
- `searx` - Requires searx_instance_url configuration

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

## Script and Automation

### /script
Execute commands from a script file
```bash
/script scripts/my-script.txt  # Execute commands from file
```

Script files are plain text files that can contain:
- Commands (lines starting with `/`)
- Chat messages (lines without prefix)
- Comments (lines starting with `#`)
- Empty lines (ignored)

**Use Cases:**
- **Parameter profiles**: Load groups of settings for different scenarios
- **Test scenarios**: Reproducible conversation sequences
- **Configuration presets**: Apply multiple settings without modifying config files
- **Automated workflows**: Run sequences of commands and queries

**Example script for parameter profile:**
```bash
# Development profile - fast responses, debug info
/set debug true
/set stream-responses false
/set show-cost true
/set main.temperature 0.7
/set main.max_tokens 500
/model chat gpt-3.5-turbo
```

### /save
Save current session commands to a script file
```bash
/save my-session           # Saves to scripts/my-session.txt
/save configs/prod         # Saves to scripts/configs/prod.txt
```

The saved script will include all commands from the current session (excluding the /save command itself), which can be replayed later with `/script`.

## Conversation Commands

### /model
Manage language models for all contexts
```bash
/model                      # Show all five models in use
/model list                 # Show available models with pricing
/model chat gpt-4.1-2025-04-14  # Set chat (main) model
/model detection custom/topic-boundary-distilbert  # Set detection model (local or API)
/model compression ollama/phi4   # Set compression model
/model synthesis ollama/phi4     # Set web synthesis model
/model critic claude-opus-4-5-20251101  # Set critic model for /critique

# Model pricing is shown per 1M tokens (not 1K)
# Pricing info comes from models.json or litellm database
```

Example `/model` output:
```
Current models:
  Chat         [C]  openai/gpt-4o-mini (8B)
  Detection    [D]  custom/topic-boundary-distilbert
  Compression  [I]  ollama/phi4
  Synthesis    [I]  ollama/phi4
  Critic       [CI] anthropic/claude-opus-4-5-20251101

Model Types:
  [D]  = Detection model (local, boundary detection)
  [C]  = Chat model (best for conversations)
  [I]  = Instruct model (best for detection/compression/synthesis)
  [CI] = Chat & Instruct model (works for both)

Use '/model list' to see available models
```

**Custom Detection Models**: You can use local models for topic boundary detection:
```bash
/model detection custom/topic-boundary-distilbert  # Local fine-tuned model
```
See [Model Configuration Guide](models-configuration.md#custom-local-models) for setup.

### /prompt, /prompts
Manage system prompts
```bash
/prompt                     # List available prompts
/prompt creative            # Switch to prompt
/prompt show                # Show current prompt
/prompt custom "Be brief"   # Set custom prompt
```

### /reflect
Enable multi-step reflection and reasoning
```bash
/reflect                    # Enable reflection mode for next message
/reflect "problem"          # Reflect on specific problem
/reflect "problem" --steps 5  # Custom number of reflection steps
/reflect off                # Disable reflection mode
```

### /critique
Have another LLM critique the last assistant response
```bash
/critique                   # Critique the last response
/critique <short_id>        # Critique a specific response by ID
```

The critic model analyzes responses for:
- Factual accuracy and potential errors
- Logical gaps or questionable assumptions
- Missing nuances or oversimplifications
- Areas that could be improved

Default model: `anthropic/claude-opus-4-5-20251101` (change with `/model critic <model>`)

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
/muse                       # Switch to muse mode (all input → web search)
                           # Web results are synthesized into comprehensive answers
```

### /chat
Return to normal chat mode
```bash
/chat                       # Switch to chat mode (normal LLM conversation)
                           # This is the default mode when starting Episodic
```

### /memory
Manage the conversation memory system (always on by default)
```bash
/memory                     # Show memory system status and search interface
/memory <query>             # Search for specific memories
/memory-stats               # Show memory system statistics
/forget <query>             # Remove specific memories from the system
```

### /new
Start a new conversation branch
```bash
/new                        # Create a new conversation branch from current node
/new --from <node_id>       # Branch from a specific node
```

### /clear
Clear current conversation context
```bash
/clear                      # Clear the current conversation context
/clear --confirm            # Clear without confirmation prompt
```

### /debug
Toggle debug mode for troubleshooting
```bash
/debug                      # Toggle debug mode on/off
/debug on                   # Enable debug mode
/debug off                  # Disable debug mode
/debug on topic             # Enable debug for topic detection specifically
/debug on memory            # Enable debug for memory system
/debug on rag               # Enable debug for RAG operations
/debug on web               # Enable debug for web search
```

### /dev
Development and testing commands (advanced users)
```bash
/dev                        # Show development commands
/dev inspect <node_id>      # Inspect node internals
/dev graph                  # Show conversation graph statistics
/dev cache                  # Show cache statistics
/dev reload                 # Reload configuration
```

### /migrate
Database migration utilities
```bash
/migrate                    # Check migration status
/migrate latest             # Migrate to latest version
/migrate check              # Check for pending migrations
/migrate backup             # Backup before migration
```

## Utility Commands

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
/muse
/set web-auto true                  # Enable automatic web search fallback
/set rag-auto true                  # Enable automatic RAG search
```

### Long Conversation Mode
```bash
/set automatic-topic-detection true  # or: /set topic-auto true
/set auto-compress-topics true       # or: /set comp-auto true
/set show-topics true
/set show-cost true
```

### Offline Mode
```bash
/mode local    # Switch all models to local, disable online features
/chat          # Start chatting offline
```

### Debug Mode
```bash
/set debug true
/set show-drift true
/set benchmark true
/set benchmark-display true
```

## Script Execution

Scripts can contain both messages and commands:

```bash
# Create a script file (example.txt):
/init
/model chat gpt-4.1-2025-04-14
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
3. **Shortcuts**: Many commands have short forms (/s for /search, /i for /index)
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