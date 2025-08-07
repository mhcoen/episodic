# Configuration Reference

This document describes all configuration options available in Episodic.

## Viewing Configuration

```bash
# Show configuration documentation
> /config-docs

# Show specific parameter value
> /set topic_detection_model
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
> /model detection ollama/phi3      # Use instruct model
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
> /set min_messages_before_topic_change 10

# Alternative shorter syntax still works
> /set debug true
> /set min_messages_before_topic_change 10
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
| `topic_detection_model` | "ollama/phi3" | Topic detection model (use instruct models) |
| `compression_model` | "gpt-3.5-turbo" | Compression model |
| `synthesis_model` | "gpt-3.5-turbo" | Web search synthesis model |
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
| `topic_detection_model` | "ollama/phi3" | Model for detection (use instruct models) |
| `min_messages_before_topic_change` | 8 | Minimum messages per topic |
| `show_topics` | false | Show topic info in responses |
| `use_dual_window_detection` | true | Use dual-window detection system |
| `dual_window_high_precision_threshold` | 0.65 | Threshold for (4,1) high precision window |
| `dual_window_safety_net_threshold` | 0.75 | Threshold for (4,2) safety net window |
| `analyze_topic_boundaries` | true | Refine topic boundaries |
| `use_llm_boundary_analysis` | true | Use LLM for boundary analysis |
| `show_hybrid_topics` | true | Show topic info with hybrid detector |
| `topic_window_size` | 5 | Window size for detection |
| `topic_similarity_threshold` | 0.3 | Similarity threshold |

### Compression Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_compress_topics` | true | Auto-compress completed topics |
| `compression_model` | "ollama/llama3" | Model for compression |
| `compression_min_nodes` | 5 | Minimum nodes to compress |
| `compression_strategy` | "simple" | Strategy: simple, keymoments |
| `show_compression_notifications` | true | Notify about compressions |

### Web Search Settings

Web search parameters can be set using the shorter `web.` prefix:

| Parameter | Short Form | Default | Description |
|-----------|------------|---------|-------------|
| `web_search_enabled` | `web.enabled` | false | Enable web search |
| `web_search_provider` | `web.provider` | "duckduckgo" | Single provider |
| `web_search_providers` | `web.providers` | ["duckduckgo"] | Provider order for fallback |
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

| Variable | Description |
|----------|-------------|
| `EPISODIC_DB_PATH` | Custom database location |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `AZURE_API_KEY` | Azure OpenAI key |
| `AZURE_API_BASE` | Azure endpoint |
| `GOOGLE_SEARCH_ENGINE_ID` | Google search engine ID for web search |
| `BING_API_KEY` | Bing API key for web search |

## Configuration Storage

Configuration is stored in the SQLite database in the `configuration` table. Changes take effect immediately without restart.

## Memory Storage

Project-specific memory is stored in `PROJECT_MEMORY.md` file in the project root. This file tracks:
- Testing framework preferences
- Recent decisions and fixes
- Current focus areas
- User preferences
- Architecture notes

This memory is specific to Claude Code sessions and persists across conversations.

## Common Configuration Patterns

### For Better Topic Detection
```bash
/set min_messages_before_topic_change 6
/set use_dual_window_detection true
/debug on topic  # See detection details
/model detection ollama/phi3  # Use instruct model
/mset detection.temperature 0.0
```

### For Faster Responses
```bash
/set stream_responses false
/set context_depth 3
/set cache_prompts true
```

### For Cost Savings
```bash
/set show_cost true
/model compression gpt-3.5-turbo
/model detection ollama/llama3
/set context_depth 3
```

### For Debugging
```bash
/set debug true
/set show_drift true
/set show_benchmarks true
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