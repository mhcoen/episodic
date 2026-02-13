# Episodic Features

This guide covers the key features and capabilities of Episodic, from basic LLM configuration to advanced knowledge management.

## 1. LLM Providers & Configuration

### Supported Providers

Episodic supports a wide range of LLM providers through LiteLLM:

**Cloud Providers (20+ supported via LiteLLM):**
- **OpenAI**: GPT-4.1, GPT-4o, GPT-3.5 Turbo
- **Anthropic**: Claude 4 (Opus, Sonnet), Claude 3.5 Sonnet, Claude 3 Haiku
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 1.5 Pro
- **Azure OpenAI**: Enterprise deployments
- **Hugging Face**: Free tier models available
- **Together AI, Mistral, Cohere**: Additional options
- **OpenRouter**: Access to multiple providers

**Local Providers:**
- **Ollama**: Run models locally (Llama 3, Mistral, Phi3, etc.)
- **LM Studio**: Local model inference

### Model Configuration

Episodic uses different models for different tasks:

```bash
# View all current models
/model

# List available models with pricing (per 1M tokens)
/model list

# Set models for specific contexts
/model chat gpt-4.1-2025-04-14  # Main conversation
/model detection ollama/phi4     # Topic detection (use instruct model)
/model compression gpt-3.5-turbo # Compression/summarization
/model synthesis claude-3-haiku  # Web search synthesis
/model critic claude-opus-4-5-20251101  # Critique responses (/critique command)
```

Configure model parameters:
```bash
/mset chat.temperature 0.7       # Creativity level
/mset detection.temperature 0    # Deterministic detection
/mset compression.max_tokens 500 # Limit summary length
```

### Setup Instructions

1. **Set API Keys** (environment variables):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GOOGLE_API_KEY="..."
   ```

2. **For Local Models**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull phi4
   ```

## 2. Voice Mode

Voice mode enables hands-free conversation with speech input and text-to-speech output.

### Features

- **Speech-to-Text (STT)**: Convert your spoken words to text
- **Text-to-Speech (TTS)**: Hear responses spoken aloud
- **Voice Activity Detection (VAD)**: Automatic speech start/stop detection
- **Provider Agnostic**: Both local (free) and cloud (paid) options
- **Cross-Platform**: Works on macOS, Linux, and Windows

### STT Providers

| Provider | Cost | Notes |
|----------|------|-------|
| `openai_whisper` (default) | ~$0.006/min | API, excellent accuracy |
| `local_whisper` | Free | Uses faster-whisper, excellent on Apple Silicon |
| `deepgram` | ~$0.008/min | Real-time streaming |

### TTS Providers

| Provider | Cost | Notes |
|----------|------|-------|
| `openai_tts` (default) | ~$0.015/min | Good quality, steerable |
| `local_piper` | Free | Fast, CPU-friendly |
| `elevenlabs` | ~$0.20/1k chars | Highest quality |

### Commands

```bash
# Basic usage
/voice           # Toggle voice mode
/voice on        # Enable voice mode
/voice off       # Disable voice mode
/voice status    # Show status and providers

# Provider configuration
/voice stt       # Show/configure STT providers
/voice tts       # Show/configure TTS providers

# Diagnostics
/voice info      # Show audio devices, test microphone
```

### Configuration

```bash
# Set providers
/set voice-stt-provider local_whisper
/set voice-tts-provider local_piper

# TTS settings
/set voice-tts-enabled true      # Enable/disable TTS output
/set voice-audio-cues true       # Enable state transition sounds

# VAD tuning
/set voice-silence-threshold-ms 1000  # Silence to end speech
/set voice-vad-aggressiveness 2       # 0-3 (higher = more filtering)
```

### Dependencies

Voice mode requires additional packages:
```bash
pip install sounddevice webrtcvad numpy

# For local STT
pip install faster-whisper

# For local TTS
pip install piper-tts
```

## 3. Web Search Integration

### Available Search Providers

- **DuckDuckGo** (default): No API key required, privacy-focused
- **Searx/SearxNG**: Self-hosted, privacy-focused meta-search
- **Google Custom Search**: Requires API key and search engine ID
- **Bing Search**: Requires API key
- **Brave Search**: Requires API key, privacy-focused with independent index

### Key Features

- **Automatic Enhancement**: Augment responses with current web information
- **Smart Caching**: Avoid redundant searches, 1-hour default cache
- **Rate Limiting**: Prevent API abuse (60 searches/hour default)
- **RAG Integration**: Fallback to web when local docs lack info

### Configuration & Usage

```bash
# Enable/disable muse mode (web search synthesis)
/muse     # Switch to muse mode  
/chat     # Switch to chat mode

# In muse mode, all input becomes web searches
latest AI developments
quantum computing news

# Configure automatic enhancement
/set web-auto true         # Auto-enhance responses
/set web.providers duckduckgo
```

## 4. Muse Mode (Web Synthesis)

### What is Muse Mode?

Muse mode transforms Episodic into a Perplexity-like AI research assistant that:
- Treats all input as web search queries (no commands needed)
- Searches multiple web sources automatically
- Extracts and reads full content from pages
- Synthesizes comprehensive answers with citations
- Maintains context for follow-up questions
- Provides source attribution and links

### How to Use

```bash
# Activate muse mode
/muse
✨ Muse mode activated!

# Ask any question
What are the latest breakthroughs in fusion energy?
```

### Synthesis Styles

Configure how muse presents information:

```bash
# Response styles (affect length and detail level)
/style concise        # Brief, direct responses
/style standard       # Clear, well-structured responses  
/style comprehensive  # Thorough, detailed responses (default)
/style custom         # Use model-specific max_tokens

# Response formats (affect presentation structure)
/format paragraph     # Flowing prose with markdown headers
/format bulleted      # Bullet points and lists
/format mixed         # Mix of paragraphs and bullets (default)
/format academic      # Formal academic style with citations
```

### Memory System

Episodic includes an always-on conversation memory system:

```bash
/memory               # Search conversation memories
/memory <query>       # Search for specific memories
/memory-stats         # Show memory system statistics
/forget <query>       # Remove specific memories
```

**Note**: This is separate from the user RAG system and automatically indexes all conversations for intelligent context recall.

## 5. Topic Detection & Management

### Automatic Organization

Episodic uses a pluggable strategy system for topic detection. The default strategy combines neural boundary detection with a commitment policy:

```bash
# View all topics
/topics

# See topic info in responses
/set show-topics true

# Enable debug mode to see detection details
/debug on topic

# Configure the detection strategy
/set topic-strategy default         # Neural + Commitment (recommended)
/set topic-strategy neural          # Neural boundary detection only
/set topic-strategy dual_window     # Dual-window cosine similarity
/set topic-strategy commitment      # Commitment policy only

# Configure minimum messages before topic change
/set min-messages-before-topic-change 8
```

13 strategies are available including `ensemble`, `cusum`, `delta`, `keyword`, `speech_act`, `time_aware`, `relative_embedding`, `summary_probe`, and `null` (manual only).

### Topic Management

```bash
# Rename ongoing topics
/topics rename

# Compress current topic
/topics compress

# View topic statistics
/topics stats

# Re-detect topics using full conversation context (hierarchical clustering)
/topics reanalyze              # Preview detected topics
/topics reanalyze apply        # Re-detect and save to database
/topics reanalyze verbose      # Show detailed merge history
```

The `reanalyze` command uses hierarchical clustering with contiguity constraint and elbow detection to find natural topic boundaries across the entire conversation, rather than the real-time sliding window approach used during chat.

### Neural Segmentation Calibration

Fine-tune topic boundary detection:

```bash
# Granularity: how many boundaries to detect
/set topic-granularity fine      # Many boundaries (threshold: 0.3)
/set topic-granularity medium    # Balanced (threshold: 0.5, default)
/set topic-granularity coarse    # Major themes only (threshold: 0.7)

# Temperature: confidence calibration
/set topic-temperature 1.0       # Default (no scaling)
/set topic-temperature 0.7       # More confident predictions
/set topic-temperature 1.5       # Less confident predictions
```

### Topic Reactivation

Episodic detects when you return to a previously discussed topic and automatically restores its context. Uses two-channel matching:

- **Semantic similarity**: Compares your message embedding against topic centroids
- **Alias matching**: Matches distinctive topic terms in referential queries

```bash
# Enable/disable (enabled by default)
/set enable-topic-reactivation true

# Context recovery mode
/set context-recovery-mode hybrid     # Switches based on reactivation (default)
/set context-recovery-mode ancestry   # Traditional DAG ancestry
/set context-recovery-mode topic_local # Topic-isolated context only
```

## 6. RAG (Knowledge Base)

### Index Your Documents

Build a personal knowledge base:

```bash
# Enable RAG
/rag on

# Index documents
/index research_paper.pdf
/index project_notes.md
/i meeting_transcript.txt  # Short form

# Index text directly
/index --text "Important information to remember"
```

### Search and Retrieve

```bash
# Search knowledge base
/search quantum computing
/s machine learning  # Short form

# Configure automatic search
/set rag-auto true              # Auto-search on every query
/set rag-relevance-threshold 0.7 # Minimum relevance score
```

### Smart Fallback

When RAG doesn't find relevant results, it automatically searches the web (if enabled):

```bash
/set rag-auto true   # Enable RAG
/set web-auto true   # Enable web fallback
# Now queries check your docs first, then web if needed
```

### Conceptual Search (WordNet)

Enable WordNet-based query expansion to find conceptually related results:

```bash
/set enable-conceptual-search true
/set wordnet-expansion-mode balanced  # narrow | balanced | broad | children_only
/set conceptual-boost-factor 0.3     # Boost for conceptual matches (0.0-1.0)
```

For example, searching for "physics" can also surface documents about "science" or "mechanics".

## 7. Knowledge Graph

The Knowledge Graph automatically extracts entities and relationships from conversations, enabling structured fact recall.

### Key Capabilities

- **Automatic extraction**: Entities (people, places, organizations) and relationships extracted per turn
- **Closure reasoning**: Transitive rules derive implicit facts (e.g., grandparent from parent-of chains)
- **Context injection**: Relevant facts are injected into LLM context when you mention known entities
- **Provenance tracking**: Every fact traces back to the conversation node that produced it

### Quick Start

```bash
# Extract from existing conversation
/kg update

# Enable real-time extraction
/set kg-realtime true

# Enable KG context injection
/set kg-context true

# Inspect the graph
/kg entities              # List entities
/kg stats                 # Statistics
/kg probe "Tell me about Emma"  # Test context injection
/kg visualize             # Interactive visualization
```

### Management Commands

```bash
/kg merge <id1> <id2>    # Merge duplicate entities
/kg dupes                 # Find duplicates
/kg explain               # Show last injection report
/kg blame <text>          # Trace provenance of an edge
/kg rebuild               # Full rebuild from scratch
```

See the [User Guide](../USER_GUIDE.md#knowledge-graph) for full command reference and configuration.

## 8. Assistant Mode

Episodic includes a built-in assistant with utility commands that work as slash commands and via natural voice.

### Core Utilities

| Command | Description |
|---------|-------------|
| `/time` | Current time |
| `/timer <duration> [label]` | Set a timer (e.g., `/timer 5m coffee`) |
| `/alarm <time> [label]` | Set an alarm (e.g., `/alarm 7am wake up`) |
| `/remind <text> in/at <time>` | Set a reminder (e.g., `/remind call mom in 1h`) |
| `/weather [location]` | Current weather (IP geolocation default) |
| `/forecast [location]` | Weather forecast |
| `/news [category]` | News headlines (general, tech, business, science, health, politics, world) |
| `/calc <expression>` | Calculator with percentages and math functions |
| `/note <text>` | Add or list notes |
| `/play <station>` | Play radio (e.g., `/play npr`) |
| `/status` | Show active timers, alarms, media state |
| `/dnd [on\|off\|duration]` | Do not disturb mode |
| `/cancel`, `/undo` | Cancel timers/alarms, undo last action |

### Calendar & Email (Google Workspace Plugin)

Calendar and email use natural language extraction — describe what you want in plain English. These work as slash commands (`/cal`, `/email`) or as natural language in voice mode and chat:

```
What's on my calendar tomorrow?
Schedule a meeting with Bob at 3pm
Check my unread email
Draft an email to Jane about the report
```

Slash command aliases: `/calendar`, `/mail`, `/gmail`. Requires `/mcp connect gsuite`.

### Voice Integration

All assistant commands work as natural speech in voice mode:
- "Set a timer for five minutes"
- "What's the weather like?"
- "What's on my calendar tomorrow?"

See [Assistant documentation](assistant.md) for full command reference and configuration.

## 9. MCP Server (Model Context Protocol)

Expose Episodic's conversation memory, knowledge base, and LLM capabilities to external AI clients (e.g., Claude Desktop, custom agents) via the Model Context Protocol.

### What It Does

- **9 tools** for reading topics, searching knowledge/memory, querying the LLM, indexing documents, and managing stateful conversation threads
- **Token authentication** — bearer tokens with optional scope restrictions and daily cost limits
- **Traces** — full audit log of every tool call with timing, status, and redacted parameters
- **Stateful threads** — external clients can hold multi-turn conversations stored in the DAG

### Quick Start

```bash
/mcp start                      # Start server on port 51983
/mcp token create my-agent      # Create auth token (shown once)
/mcp status                     # Verify server is running
```

### Tools Overview

| Category | Tools | Description |
|----------|-------|-------------|
| Read-only | `get_model_info`, `get_runtime_state`, `get_topics` | Inspect instance state |
| Search | `search_knowledge`, `search_memory` | Query RAG and conversation memory |
| LLM | `ask_llm_stateless`, `ask_llm_stateful` | One-shot and threaded LLM queries |
| Write | `index_document`, `create_thread` | Add documents, create conversation threads |

### Security

- Tokens use SHA-256 hashing — plaintext is never stored
- Thread handles are also hash-only, shown once on creation
- Daily per-client cost limits (default $10/day)
- Trace logging redacts sensitive parameters (keys, tokens, secrets)
- Server binds to localhost by default (`127.0.0.1`)

## Quick Configuration Reference

### Essential Commands

```bash
# Models
/model list              # See available models
/model chat gpt-4        # Set chat model
/mset                    # View all model parameters

# Features
/rag on/off              # Knowledge base
/muse | /chat            # Web search synthesis
/muse                    # Synthesis mode

# Settings
/set                     # View all settings
/set cost true           # Show usage costs
/set stream false        # Disable streaming
/config-docs             # Full configuration guide
```

### Common Workflows

**Research Mode:**
```bash
/rag on
/muse
/set rag-auto true
/set web-auto true
```

**Offline Mode:**
```bash
/model chat ollama/llama3
/rag off
/chat
```

**Cost-Conscious Mode:**
```bash
/model chat gpt-3.5-turbo
/model compression gpt-3.5-turbo
/set cost true
/compression stats
```