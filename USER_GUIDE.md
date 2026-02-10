# Episodic User's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Mode Switching](#mode-switching)
4. [Getting Started](#getting-started)
5. [Basic Usage](#basic-usage)
6. [Topic Management](#topic-management)
7. [Knowledge Base (RAG)](#knowledge-base-rag)
8. [Knowledge Graph](#knowledge-graph)
9. [Web Search](#web-search)
10. [Muse Mode](#muse-mode)
11. [Configuration](#configuration)
12. [Saving and Loading Conversations](#saving-and-loading-conversations)
13. [Advanced Features](#advanced-features)
14. [Experimental Features](#experimental-features)

## Introduction

Episodic is a conversational memory system that helps prevent the "I already told you that" problem in chatbots.

Instead of stuffing the model with lots of old messages, Episodic tracks what you're talking about, leaves irrelevant history out of the context, and injects only the most relevant prior facts. It can also connect related facts through a knowledge graph (multi-hop traversal) to recover context you didn't explicitly repeat.

Episodic has two modes:

* Simple mode: chat normally, with topic organization and context handled automatically. Conversations are stored as readable markdown files.
* Advanced mode: for developers, researchers, and anyone who wants full control. It unlocks commands for multi-model workflows, retrieval (RAG), prompt/system tuning, benchmarking, and cost/performance analysis.

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
| `/save` | Save current topic to markdown |
| `/load <file>` | Load markdown conversation |
| `/files [dir]` or `/ls [dir]` | List markdown files in directory |

### Model Selection
```bash
/model  # Show all models in use (chat, detection, compression, synthesis, intent, critic)
/model list  # Show 60+ available models with pricing
/model chat gpt-4o  # Set chat (main conversation) model
/model detection custom/topic-boundary-distilbert  # Set topic detection model
/model compression ollama/phi4  # Set compression model
/model synthesis ollama/phi4  # Set web synthesis model
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
| `/topics delete <name>` | Delete topic by exact name |
| `/topics delete --pattern <pat>` | Delete topics matching pattern |
| `/topics delete --time <expr>` | Delete topics by time range |

### Topic Detection Strategies

Episodic uses a pluggable strategy system for topic detection. The default strategy combines neural boundary detection with a commitment policy to reduce false positives.

#### Configuring the Strategy
```bash
/set topic-strategy default       # Neural + Commitment (recommended)
/set topic-strategy neural        # Neural boundary detection only
/set topic-strategy commitment    # Commitment policy only
/set topic-strategy dual_window   # Dual-window cosine similarity
/set topic-strategy ensemble      # Weighted ensemble of multiple signals
```

#### Available Strategies

| Strategy | Description |
|----------|-------------|
| `default` | Neural detection + Commitment policy (recommended) |
| `neural` | DistilBERT-based boundary classification |
| `commitment` | Evidence accumulation before committing to a boundary |
| `dual_window` | Dual-window (4,1)+(4,2) cosine similarity |
| `ensemble` | Weighted combination of multiple strategies |
| `cusum` | CUSUM change-point detection on embedding drift |
| `delta` | Raw embedding delta with adaptive threshold |
| `keyword` | Explicit transition phrase detection |
| `speech_act` | Speech act classification for topic shifts |
| `time_aware` | Time-gap weighted detection |
| `relative_embedding` | Relative embedding position changes |
| `summary_probe` | LLM-based summary comparison |
| `null` | No automatic detection (manual only) |

#### Manual Detection
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

### Topic Reactivation

Episodic can detect when you return to a previously discussed topic and automatically restore its context. This uses a two-channel matching system:

- **Channel A (Semantic)**: Compares your message embedding against topic centroids
- **Channel B (Alias)**: Matches distinctive topic terms in referential queries (e.g., "what was that about Emma?")

When a match is found, the system either silently reactivates the topic or presents a disambiguation prompt if multiple topics match.

```bash
# Enable/disable reactivation (enabled by default)
/set enable-topic-reactivation true

# Show a one-liner when reactivation fires
/set show-reactivation-decisions true

# Log detailed probe features for debugging
/set reactivation-log-features true
```

### Context Recovery Modes

When assembling context for the LLM, Episodic supports three strategies:

| Mode | Description |
|------|-------------|
| `ancestry` | Traditional DAG ancestry traversal (all parent nodes) |
| `topic_local` | Topic-isolated context with semantic anchors and summaries |
| `hybrid` | Switches between ancestry and topic_local based on reactivation (default) |

```bash
/set context-recovery-mode hybrid    # Recommended (default)
/set context-recovery-mode ancestry  # Traditional behavior
/set context-recovery-mode topic_local  # Topic isolation only
```

In hybrid mode, when topic reactivation fires, context switches to `topic_local` which provides the reactivated topic's summary and semantically relevant anchors instead of raw ancestry.

### Topic Deletion
Remove unwanted topics by name, pattern, or time range:

```bash
# Delete by exact name
/topics delete python-retry-mechanisms

# Delete by pattern (case-insensitive)
/topics delete --pattern "test"
/topics delete --pattern "sourdough"

# Delete by time range (natural language)
/topics delete --time "since yesterday"
/topics delete --time "since 2 hours ago"
/topics delete --time "before last week"
/topics delete --time "between 10am and 2pm today"

# Options
/topics delete --dry-run    # Preview without deleting
/topics delete --force      # Skip confirmation
```

**What gets deleted:**
- Topic metadata (name, boundaries)
- Topic centroids (embedding data)
- Topic node associations
- Working set entries
- ChromaDB embeddings

**What is preserved:**
- Conversation nodes (message history)
- Other topics unaffected

### Test Mode
For testing and development, use test mode to isolate changes from production:

```bash
/test                       # Show status and help
/test clone                 # Copy production → test
/test on                    # Switch to test ([TEST] prompt)
/test off                   # Return to production
/test clear                 # Wipe test environment
/test status                # Detailed test database info
```

Test mode uses separate paths (`~/.episodic/test/`) for both SQLite and ChromaDB, ensuring complete isolation from production data.

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

### Conceptual Search (WordNet)

Enable WordNet-based conceptual expansion to find results related by meaning, not just keywords. For example, searching for "physics" can also find documents about "science" or "mechanics".

```bash
/set enable-conceptual-search true
/set wordnet-expansion-mode balanced   # narrow | balanced | broad | children_only
/set expansion-max-depth 2             # Hierarchy depth (1-3)
/set expansion-max-terms 10            # Max expansion terms
/set conceptual-boost-factor 0.3       # Boost for conceptual matches (0.0-1.0)
```

## Knowledge Graph

The Knowledge Graph (KG) automatically extracts entities and relationships from your conversations, enabling the LLM to recall structured facts like "Emma studies at MIT" or "the project deadline is March 15th".

### Enabling the Knowledge Graph

The KG has two modes of operation:

```bash
# Manual extraction (process conversation history on demand)
/kg update

# Real-time extraction (extract from each turn as you chat)
/set kg-realtime true

# Timed background extraction
/set kg-auto true
/set kg-interval 3600        # Extract every hour

# Enable KG context injection (inject facts into LLM context)
/set kg-context true
```

### How It Works

1. **Extraction**: Each user/assistant exchange is analyzed by an LLM to extract entities (people, places, organizations, concepts) and relationships between them
2. **Storage**: Entities and edges are stored in SQLite tables with provenance tracking (which conversation node produced each fact)
3. **Closure**: Transitive rules derive implicit facts (e.g., if A is-parent-of B and B is-parent-of C, then A is-grandparent-of C)
4. **Context Injection**: When you mention a known entity, relevant facts are injected into the LLM context so it can reference them naturally

### KG Commands

| Command | Description |
|---------|-------------|
| `/kg` or `/kg status` | Show KG status (entity/edge counts, staleness) |
| `/kg entities` | List all entities |
| `/kg entity <id>` | Show detail for entity (aliases, degree) |
| `/kg edges [entity_id]` | List edges, optionally filtered by entity |
| `/kg search <query>` | Search entities by name or alias |
| `/kg stats` | Comprehensive statistics (types, predicates, assertions) |
| `/kg update [--max N] [--lookback N] [--dry-run]` | Run batch extraction from high-water mark |
| `/kg rebuild` | Full rebuild: drop all KG data and reprocess |
| `/kg skip <node_id> [--reason ...]` | Add node to skip list |
| `/kg patch <node_id>` | Show the extraction patch for a node |
| `/kg probe <text>` | Dry-run context injection for a query |
| `/kg merge <id1> <id2> [--survivor=<id>]` | Merge two entities |
| `/kg dupes` | Find duplicate entities (same name + type) |
| `/kg eval [dataset] [--model M] [--conditions A,B,C]` | Run ablation evaluation |
| `/kg visualize [--layout cose] [--type T] [--relation R] [--save path]` | Interactive graph visualization |
| `/kg explain` | Show what happened on last KG context injection |
| `/kg blame <text>` | Show provenance for an edge from last injection |

### Example

```
> Tell me about my friend Emma
🧠 KG context injected (3 edges, 1 derived):
  Emma --studies_at--> MIT
  Emma --lives_in--> Boston
  Emma --interested_in--> machine learning
  (derived) Emma --located_in--> Massachusetts

Based on what I know, Emma is studying at MIT in Boston...
```

### KG Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kg_context` | false | Enable KG facts in context assembly |
| `kg_realtime` | false | Per-turn background extraction |
| `kg_auto` | false | Timer-based batch extraction |
| `kg_interval` | 3600 | Extraction interval in seconds |
| `kg_lookback` | 3 | Preceding turns for extraction context |
| `kg_budget` | 500 | Token budget for KG context |
| `kg_max_entities` | 5 | Max entities to match per message |
| `kg_max_edges` | 5 | Max edges per matched entity |
| `kg_max_derived` | 3 | Max derived (closure) edges total |
| `kg_include_past` | false | Include TIME_PAST edges by default |
| `kg_closure_seed_limit` | 3 | Max entities used as closure seeds |
| `kg_relevance_gate` | true | Suppress block if no direct overlap |

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
   - Configure with `/set searx-instance-url`
   - No API key needed

5. **Brave** (Requires API key)
   - Set `BRAVE_API_KEY`
   - Privacy-focused with independent index

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

### Muse Configuration
```bash
# Control response detail level (applies to both chat and muse modes)
/detail minimal      # Facts only, concise responses
/detail moderate     # Balanced detail with context (default)
/detail detailed     # In-depth explanations
/detail maximum      # Comprehensive with all nuances

# Configure source handling
/set muse-sources first-only    # Use only the first search result
/set muse-sources top-three     # Use top 3 results (default)
/set muse-sources all-relevant  # Use all relevant results
/set muse-sources selective     # Intelligently select sources

# Set context depth for follow-ups
/set muse-context-depth 5       # Include last 5 messages for context
```

## Saving and Loading Conversations

### Saving and Exporting Conversations

Episodic provides two commands for saving conversations to markdown files:

#### Simple Save (Current Topic Only)
```bash
/save                        # Save current topic with auto-generated filename
/save conversation.md        # Save current topic to specific file
```

#### Advanced Export (Multiple Topics)
```bash
/out                         # Export current topic with auto-generated filename
/out 1-3                     # Export topics 1 through 3
/out all conversation.md     # Export entire conversation to specific file
/out 2,4,6 selected.md       # Export specific topics
```

##### Topic Specifications for /out:
- `current` - Export the current/ongoing topic (default)
- `3` - Export topic number 3
- `1-5` - Export topics 1 through 5 (range)
- `1,3,5` - Export topics 1, 3, and 5 (list)
- `all` - Export all topics

#### Export Format
Saved files include:
- Date and time at the top
- Conversation exchanges with **You:** and **LLM:** prefixes
- Model information and changes
- Topic boundaries and detection notes
- Node IDs for reference (cannot be reused)

### Loading Conversations

Load previously saved conversations or markdown files from other sources:

```bash
/load conversation.md        # Load a conversation file (simple mode)
/load exports/meeting.md     # Load from specific directory

/in conversation.md          # Alternative command (same functionality)
```

**Note**: Loaded conversations create new nodes in your conversation DAG. The system can parse various markdown dialogue formats including:
- Episodic's save format
- ChatGPT exports
- Claude conversation exports
- Generic markdown with dialogue patterns

### Listing Files

Browse available markdown files:

```bash
/files                      # List files in current directory (primary command)
/ls exports                 # List files in exports directory (using alias)
/files ~/Documents/chats    # List files in specific directory
```

The listing shows:
- File size
- Modification time (relative for recent files)
- Preview of file content (title or first heading)

## File References (@file)

You can attach local files directly in chat messages using the `@file` syntax.
Episodic will extract the content and include it in the LLM context.

### Syntax
```text
@file.txt
@"path with spaces/notes.txt"
@file.pdf:vision:1-5
```

### Behavior
- **Text files**: Content is read and included in the prompt.
- **PDFs**:
  - `@file.pdf` extracts text from the document.
  - `@file.pdf:vision` renders pages as images for vision-capable models.
  - `@file.pdf:vision:1-5` limits rendering to specific pages.

### Configuration
```bash
/set file-ref-vision-pages 5     # Default pages for :vision without a range
/set file-ref-max-text-size 100000  # Max extracted text characters
/set pdf-extractor pdfplumber    # pdfplumber (default) or marker
```

### Example Workflow

#### Simple Workflow
```bash
# Have a conversation about a project
> Let's design a REST API for a task management system
🤖 I'll help you design a REST API for task management...

# Save current topic for team review
> /save api-design.md
✅ Saved to: exports/api-design.md

# Later, continue the conversation
> /load exports/api-design.md
✅ Loaded conversation from exports/api-design.md

> What about authentication for this API?
🤖 Building on our API design, here are authentication options...
```

#### Advanced Workflow
```bash
# Work on multiple topics
> /topics
[1] ✓ API Design Discussion
[2] ✓ Database Schema Planning
[3] ○ Authentication Strategy (ongoing)

# Export specific topics
> /out 1-2 api-and-database.md
✅ Exported topics 1-2 to: exports/api-and-database.md

# Export everything
> /out all complete-project.md
✅ Exported all topics to: exports/complete-project.md
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
/format bulleted             # Bullet points and lists
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
/memory             # Search conversation memories
/memory-stats       # Show memory system statistics
/forget <query>     # Remove specific memories
/cost               # Show session token usage and costs
/summary            # Summarize recent conversation
```

**Note**: The memory system is always on by default and separate from the user RAG system. It automatically indexes your conversations for intelligent context recall.

### Rollback

Roll back the conversation to a specific point, deleting all nodes after the target. This also cleans up topic boundaries and KG data for deleted nodes.

```bash
/rollback <short_id>   # Roll back to a specific node
```

Use `/ancestry` or `/show` to find the node ID you want to roll back to. This operation cannot be undone.

### Critique

Have another LLM critique the last assistant response, analyzing it for accuracy, logical gaps, and areas for improvement.

```bash
/critique              # Critique the last response
/critique <short_id>   # Critique a specific response
```

The critique uses the critic model (configurable with `/model critic <model>`), defaulting to a capable model for thorough analysis.

### Reflection

Enable multi-step reasoning where the LLM analyzes a problem, reflects on its analysis, and synthesizes a final answer.

```bash
/reflect                        # Enable reflection mode for next message
/reflect "complex problem"      # Reflect on a specific problem immediately
/reflect off                    # Disable reflection mode
```

Reflection uses configurable steps (default 3): initial analysis, self-critique, and final synthesis. This is useful for complex reasoning tasks where the first answer may miss nuances.

### MCP Server

Episodic includes a Model Context Protocol server for exposing conversation memory to external AI clients. See the [MCP Guide](docs/mcp-guide.md) for full setup details and the [Features Guide](docs/features.md) for a tool overview.

### Reset Configuration
```bash
/reset              # Show reset options
/reset all          # Reset all settings to defaults
/reset <param>      # Reset specific parameter
/reset all --save   # Reset and save to config file
```

## Experimental Features

### Alternative Topic Detection Strategies

The default strategy (Neural + Commitment) works well for most conversations. The following alternative strategies are available for experimentation:

- **`ensemble`**: Weighted combination of multiple strategies for higher accuracy
- **`cusum`**: CUSUM change-point detection on embedding drift series
- **`speech_act`**: Detects topic shifts via speech act classification
- **`time_aware`**: Incorporates time gaps between messages
- **`summary_probe`**: Uses LLM to compare topic summaries

Use `/set topic-strategy <name>` to switch. See [Topic Detection Strategies](#topic-detection-strategies) for the full list.

### Topic Evaluation Framework

Evaluate and compare strategies against annotated datasets:
```bash
/topics evaluate                    # Run evaluation with current strategy
/topics evaluate --strategy neural  # Evaluate a specific strategy
/topics calibrate                   # Calibrate thresholds
```

### Running Topic Prediction
Planned feature to show topic predictions in real-time. Not yet implemented.

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
/set automatic-topic-detection true  # or: /set topic-auto true
/set auto-compress-topics true       # or: /set comp-auto true
/set show-topics true                # See topic evolution

# Monitor context usage
/set show-cost true
```

### Offline Usage with Ollama
```bash
# Use local models for all contexts
/model chat ollama/llama3.3
/model detection ollama/phi4
/model compression ollama/phi4
/model synthesis ollama/phi4
```

### Collaborative Documentation
```bash
# Morning standup discussion
What are the key points from yesterday's API review?

# Save for team sharing
/save standup-2024-01-15.md

# Team member loads and continues
/load standup-2024-01-15.md
Based on the API review, I suggest we add rate limiting...
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
/mset chat.temperature 0.7
/mset chat.max_tokens 500
/model chat gpt-4o-mini
```

**scripts/creative-profile.txt:**
```bash
# Creative writing settings
/mset chat.temperature 1.2
/mset chat.max_tokens 2000
/mset chat.top_p 0.95
/model chat gpt-4o
/prompt creative
```

#### 2. Test Scenarios
Create reproducible test cases:

**scripts/test-topics.txt:**
```bash
/init --erase
/set min-messages-before-topic-change 2
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
/model chat gpt-4o
/muse
```

**scripts/backup-conversations.txt:**
```bash
# Save current topic for backup
/save backup-{date}.md
/topics
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
   - Adjust `/set drift-threshold 0.9` (higher = less sensitive)
   - Try `/set min-messages-before-topic-change 8`

3. **Context window exceeded**
   - Enable compression: `/set auto-compress-topics true`
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
- **topic_centroids**: Topic embedding centroids for reactivation
- **compressions_v2**: Compressed topic summaries
- **rag_documents**: Indexed documents
- **kg_entities / kg_edges / kg_assertions**: Knowledge graph data
- **kg_patches / kg_state**: KG extraction tracking
- **configuration**: Key-value settings

### Module Organization
- `episodic/cli_main.py` - CLI setup and session lifecycle
- `episodic/conversation_pipeline.py` - Turn processing orchestration
- `episodic/conversation.py` - Conversation state management
- `episodic/topics/` - Pluggable topic detection strategies (13 strategies)
- `episodic/kg/` - Knowledge graph extraction, storage, closure, context injection
- `episodic/context_recovery/` - Context assembly strategies (ancestry, topic_local, hybrid)
- `episodic/recall/` - Topic reactivation probe and alias matching
- `episodic/rag*.py` - Knowledge base and WordNet conceptual search
- `episodic/web_search.py` - Web search integration
- `episodic/web_search_providers/` - Provider implementations (DuckDuckGo, Google, Bing, Searx, Brave)
- `episodic/mcp/` - MCP server, tools, auth, threads, traces
- `episodic/token_counting.py` - Token budget tracking

## Contributing

Episodic has several areas that could use improvement:

1. **Non-linear DAG conversations** - Currently only linear
2. **Running topic prediction** - Real-time topic guessing (planned)
3. **Embedding provider alternatives** - Currently uses sentence-transformers
4. **Knowledge graph improvements** - Better extraction models, multi-hop reasoning
5. **Topic detection tuning** - Strategy evaluation and calibration
