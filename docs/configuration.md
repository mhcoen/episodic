# Configuration Reference

This document describes all configuration options available in Episodic.

## Viewing Configuration

```bash
# Show configuration documentation
> /config-docs

# Show specific parameter value
> /set topic-detection-model
```

## Setting Configuration

### Model Selection
```bash
# Show current chat model
> /model

# Show all models for all contexts
> /model list

# Set models for different contexts
> /model chat gpt-4.1-2025-04-14
> /model detection ollama/phi4      # Use instruct model
> /model compression gpt-3.5-turbo
> /model synthesis claude-3-haiku
```

### Model Parameters
```bash
# Show all parameters
> /mset

# Show parameters for specific context
> /mset chat
> /mset detection

# Set specific parameters
> /mset chat.temperature 0.7
> /mset detection.temperature 0
> /mset compression.max_tokens 500
> /mset synthesis.top_p 0.9
```

### Other Configuration
```bash
# Set general configuration values
> /set debug true
> /set min-messages-before-topic-change 10

# Alternative shorter syntax still works
> /set debug true
> /set min-messages-before-topic-change 10
```

## Configuration Categories

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `active_prompt` | "default" | Active system prompt |
| `debug` | false | Enable debug output |
| `show_cost` | false | Show cost after each response |
| `show_drift` | true | Show drift scores in debug |
| `model` | "gpt-4o-mini" | Chat (main conversation) model |
| `topic_detection_model` | "huggingface/tiiuae/falcon-7b-instruct" | Topic detection model (use instruct models) |
| `compression_model` | "huggingface/tiiuae/falcon-7b-instruct" | Compression model |
| `synthesis_model` | "huggingface/tiiuae/falcon-7b-instruct" | Web search synthesis model |
| `context_depth` | 5 | Number of previous messages to include |
| `use_context_cache` | true | Enable prompt caching for supported models |
| `use_dual_window_detection` | true | Use dual-window topic detection system |

### Display Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `color_mode` | full | Color display mode (full/basic/none) |
| `text_wrap` | true | Enable text wrapping for long lines |
| `stream_responses` | true | Stream LLM responses as they arrive |
| `stream_rate` | 15 | Words per second for streaming display |
| `show_input_box` | true | Display user input in styled box |
| `use_unicode_boxes` | true | Use Unicode box characters (false for ASCII) |
| `enable_tab_completion` | true | Enable tab completion for commands and parameters |

#### Color Mode Options

The `color_mode` setting controls how Episodic displays colors in the terminal:

- **full** (default): Full 256-color palette with rich colors and gradients
  - Best for modern terminals (iTerm2, Terminal.app, VS Code, etc.)
  - Provides cyan for system messages, distinct colors for headers, etc.
  
- **basic**: Limited to 8 standard ANSI colors
  - For older terminals or when full colors aren't supported
  - Uses basic red, green, blue, cyan, magenta, yellow, black, white
  
- **none**: No colors at all, plain text only
  - For terminals without color support
  - Useful for piping output to files or other programs

Set with: `/set color-mode full`, `/set color-mode basic`, or `/set color-mode none`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `color_mode` | "full" | Color output mode: full (256 colors), basic (8 colors), none (no colors) |
| `stream_responses` | true | Stream LLM responses |
| `stream_rate` | 15 | Streaming speed (words/sec) |
| `stream_constant_rate` | false | Use constant streaming rate |
| `stream_natural_rhythm` | false | Natural speech-like streaming |
| `stream_char_mode` | true | Character-based streaming |
| `stream_char_rate` | 1000 | Characters per second |
| `wrap_text` | true | Word wrap long lines |
| `show_benchmarks` | true | Show performance metrics |

### Topic Detection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `automatic_topic_detection` | true | Enable automatic detection |
| `topic_strategy` | "default" | Strategy: default, neural, commitment, dual_window, ensemble, etc. |
| `topic_detection_model` | "custom/topic-boundary-distilbert" | Model for neural detection |
| `min_messages_before_topic_change` | 8 | Minimum messages per topic |
| `show_topics` | false | Show topic info in responses |
| `topic_strategy_params` | {} | Strategy-specific parameters |

### Topic Reactivation & Context Recovery Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_topic_reactivation` | true | Detect return to dormant topics |
| `context_recovery_mode` | "hybrid" | Context strategy: ancestry, topic_local, hybrid |
| `context_token_budget` | 4000 | Max tokens for context assembly |
| `show_reactivation_decisions` | false | Show one-liner when reactivation fires |
| `reactivation_log_features` | true | Log detailed probe features |
| `topic_context_retrieval` | false | Retrieve context from previous topics |
| `topic_context_max_messages` | 10 | Max messages from previous topics |
| `topic_context_max_tokens` | 2000 | Max tokens from previous topics |
| `anchor_count` | 3 | Semantic anchors for topic-local context |
| `anchor_similarity_threshold` | 0.5 | Minimum anchor similarity |
| `import_detection_enabled` | true | Cross-topic import detection |
| `import_token_budget` | 100 | Max tokens for imported context |

### Compression Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_compress_topics` | true | Auto-compress completed topics |
| `compression_model` | "gpt-4o-mini" | Model for compression |
| `compression_min_nodes` | 5 | Minimum nodes to compress |
| `compression_strategy` | "simple" | Strategy: simple, keymoments |
| `show_compression_notifications` | true | Notify about compressions |

### Web Search Settings

Web search parameters can be set using the shorter `web.` prefix:

| Parameter | Short Form | Default | Description |
|-----------|------------|---------|-------------|
| `web_search_enabled` | `web.enabled` | false | Enable web search |
| `web_search_providers` | `web.providers` | ["duckduckgo"] | Provider order (first is primary, others are fallbacks) |
| `web_search_fallback_enabled` | `web.fallback` | true | Enable automatic fallback |
| `web_search_fallback_cache_minutes` | `web.fallback_cache_minutes` | 5 | Cache working provider (minutes) |
| `web_search_max_results` | `web.max_results` | 5 | Number of results to retrieve |
| `web_search_cache_duration` | `web.cache` | 3600 | Cache search results (seconds) |
| `web_search_rate_limit` | `web.rate_limit` | 60 | Max searches per hour |
| `web_search_timeout` | `web.timeout` | 10 | Search timeout (seconds) |
| `web_search_synthesize` | `web.synthesize` | true | Synthesize results with LLM |
| `web_search_show_urls` | `web.show_urls` | true | Display URLs in results |

**Example Usage:**
```bash
# Configure provider fallback order
/set web.providers google,bing,duckduckgo

# Use only free providers
/set web.providers duckduckgo,searx

# Disable fallback (use only first provider)
/set web.fallback false

# Adjust cache duration
/set web.cache 7200  # Cache for 2 hours
```

### Knowledge Graph Settings

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

### Conceptual Search (WordNet) Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_conceptual_search` | false | Enable WordNet-based conceptual search |
| `search_query_expansion` | true | Auto-expand queries with related concepts |
| `expansion_max_depth` | 2 | Max hierarchy depth (1-3) |
| `expansion_max_terms` | 10 | Max expansion terms |
| `conceptual_boost_factor` | 0.3 | Boost for conceptual matches (0.0-1.0) |
| `wordnet_expansion_mode` | "balanced" | Mode: narrow, balanced, broad, children_only |

### Model Selection

Models can be configured per context:

```bash
/model chat gpt-4o              # Main conversation model
/model detection custom/topic-boundary-distilbert  # Topic detection
/model compression gpt-4o-mini  # Topic compression
/model synthesis gpt-4o-mini    # Web synthesis
/model critic anthropic/claude-opus-4-5-20251101   # Critique model
/model extraction gpt-4o-mini   # KG extraction model
```

### Model Parameters

Model parameters are organized by context:
- `chat` - Main conversation parameters (stored as `main_params`)
- `detection` - Topic detection parameters (stored as `topic_params`)
- `compression` - Compression parameters (stored as `compression_params`)
- `synthesis` - Web synthesis parameters (stored as `synthesis_params`)

Each supports:
- `temperature` (0.0-2.0) - Randomness/creativity
- `max_tokens` (integer) - Maximum response length
- `top_p` (0.0-1.0) - Nucleus sampling threshold
- `presence_penalty` (-2.0-2.0) - Penalize repeated topics
- `frequency_penalty` (-2.0-2.0) - Penalize repeated words

Note: Some models don't support all parameters (e.g., Google Gemini doesn't support presence/frequency penalties)

Example:
```bash
> /mset chat.temperature 0.8
> /mset detection.temperature 0.0
> /mset compression.max_tokens 500
```

## Environment Variables

Environment variables provide an alternative to CLI configuration, useful for automation, containers, and shell profiles.

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `HUGGINGFACE_API_KEY` | Hugging Face API key |
| `AZURE_API_KEY` | Azure OpenAI key |
| `AZURE_API_BASE` | Azure OpenAI endpoint |
| `PICOVOICE_ACCESS_KEY` | Picovoice access key (for wake word detection) |

### Web Search Provider Keys

| Variable | Description |
|----------|-------------|
| `GOOGLE_SEARCH_ENGINE_ID` | Google Custom Search engine ID |
| `BING_API_KEY` | Bing Search API key |
| `BRAVE_API_KEY` | Brave Search API key |
| `SEARX_INSTANCE_URL` | Self-hosted Searx instance URL |

### Episodic Configuration

These `EPISODIC_*` variables map directly to `/set` parameters:

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_DB_PATH` | - | Custom database location |
| `EPISODIC_DEBUG` | `debug` | Enable debug output |
| `EPISODIC_CACHE` | `use-context-cache` | Enable prompt caching |
| `EPISODIC_PROMPT` | `active-prompt` | Active system prompt name |

#### Web Search

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_WEB_ENABLED` | `web-search-enabled` | Enable web search |
| `EPISODIC_WEB_AUTO` | `web-search-auto-enhance` | Auto-enhance responses with web |
| `EPISODIC_WEB_PROVIDERS` | `web-search-providers` | Comma-separated provider list |
| `EPISODIC_WEB_RESULTS` | `web-search-max-results` | Max results per search |

#### RAG (Knowledge Base)

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_RAG_ENABLED` | `rag-enabled` | Enable RAG |
| `EPISODIC_RAG_AUTO` | `rag-auto-search` | Auto-search knowledge base |
| `EPISODIC_RAG_THRESHOLD` | `rag-search-threshold` | Relevance threshold (0.0-1.0) |
| `EPISODIC_RAG_RESULTS` | `rag-max-results` | Max results to include |

#### Topic Detection

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_TOPIC_MODEL` | `topic-detection-model` | Model for topic detection |
| `EPISODIC_TOPIC_AUTO` | `automatic-topic-detection` | Enable auto topic detection |
| `EPISODIC_TOPIC_MIN` | `min-messages-before-topic-change` | Min messages per topic |

#### Compression

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_COMPRESSION_MODEL` | `compression-model` | Model for compression |
| `EPISODIC_COMPRESS_AUTO` | `auto-compress-topics` | Enable auto compression |
| `EPISODIC_COMPRESS_MIN` | `compression-min-nodes` | Min nodes to compress |

#### Display

| Variable | CLI Equivalent | Description |
|----------|----------------|-------------|
| `EPISODIC_COLOR_MODE` | `color-mode` | Color mode (full/basic/none) |
| `EPISODIC_STREAM_RATE` | `stream-rate` | Streaming speed (words/sec) |
| `EPISODIC_SHOW_COST` | `show-cost` | Show token costs |
| `EPISODIC_SHOW_TOPICS` | `show-topics` | Show topic info |

### Example Usage

```bash
# In your shell profile (~/.bashrc, ~/.zshrc, etc.)
export OPENAI_API_KEY="sk-..."
export EPISODIC_DEBUG=true
export EPISODIC_SHOW_COST=true
export EPISODIC_TOPIC_AUTO=true
export EPISODIC_WEB_PROVIDERS="duckduckgo,brave"

# Or for a single session
EPISODIC_DEBUG=true python -m episodic
```

## Configuration Storage

Configuration is stored in the SQLite database in the `configuration` table. Changes take effect immediately without restart.

## Memory Storage

Project-specific memory is stored in `PROJECT_MEMORY.md` file in the project root. This file tracks:
- Testing framework preferences
- Recent decisions and fixes
- Current focus areas
- User preferences
- Architecture notes

This memory persists across sessions and helps maintain context for development.

## Common Configuration Patterns

### For Better Topic Detection
```bash
/set min-messages-before-topic-change 6
/set use-dual-window-detection true
/debug on topic  # See detection details
/model detection ollama/phi4  # Use instruct model
/mset detection.temperature 0.0
```

### For Faster Responses
```bash
/set stream-responses false
/set context-depth 3
/set cache-prompts true
```

### For Cost Savings
```bash
/set show-cost true
/model compression gpt-3.5-turbo
/model detection ollama/llama3
/set context-depth 3
```

### For Debugging
```bash
/set debug true
/set show-drift true
/set show-benchmarks true
```

## Resetting Configuration

To reset a value to default:
```bash
/reset parameter_name
```

To reset all configuration:
```bash
/reset all
```

To reset everything including conversation history:
```bash
/init --erase  # WARNING: This erases everything!
```