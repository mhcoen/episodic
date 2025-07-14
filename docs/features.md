# Episodic Features

This guide covers the key features and capabilities of Episodic, from basic LLM configuration to advanced knowledge management.

## 1. LLM Providers & Configuration

### Supported Providers

Episodic supports a wide range of LLM providers through LiteLLM:

**Cloud Providers:**
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku), Claude 2
- **Google**: Gemini Pro, Gemini 1.5 Pro/Flash
- **Groq**: Fast inference for Llama, Mixtral models
- **Azure OpenAI**: Enterprise deployments

**Local Providers:**
- **Ollama**: Run models locally (Llama 3, Mistral, etc.)
- **LM Studio**: Local model inference

### Model Configuration

Episodic uses different models for different tasks:

```bash
# View all current models
/model

# Set models for specific contexts
/model chat gpt-4o              # Main conversation
/model detection ollama/llama3   # Topic detection
/model compression gpt-3.5-turbo # Compression/summarization
/model synthesis claude-3-haiku  # Web search synthesis
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
   ollama pull llama3
   ```

## 2. Web Search Integration

### Available Search Providers

- **DuckDuckGo** (default): No API key required, privacy-focused
- **Searx/SearxNG**: Self-hosted, privacy-focused meta-search
- **Google Custom Search**: Requires API key and search engine ID
- **Bing Search**: Requires API key

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
/set web_search_provider duckduckgo
```

## 3. Muse Mode (Web Synthesis)

### What is Muse Mode?

Muse mode transforms Episodic into a Perplexity-like AI research assistant that:
- Searches multiple web sources
- Extracts and reads full content
- Synthesizes comprehensive answers
- Provides source attribution

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
/set synthesis_style comprehensive  # Default
# Options: concise, standard, comprehensive, exhaustive

/set synthesis_detail balanced     # Default  
# Options: minimal, low, balanced, high, maximum
```

### Output Formats

```bash
/set synthesis_format mixed        # Default
# Options: paragraph, bullet_points, numbered_list, mixed, academic
```

## 4. Visualization

### Interactive Conversation Graph

View your conversation as an interactive directed graph:

```bash
# Start visualization server
/visualize

# Options
/visualize --browser    # Open in default browser
/visualize --window     # Native window (if supported)
```

### Features

- **Real-time Updates**: Graph updates as you chat
- **Interactive Navigation**: Double-click nodes to jump to that point
- **Visual Topic Boundaries**: See how conversations are organized
- **Zoom & Pan**: Explore large conversation trees

## 5. Topic Detection & Management

### Automatic Organization

Episodic automatically detects when conversation topics change and creates boundaries:

```bash
# View all topics
/topics

# See topic info in responses
/set show_topics true

# Configure detection sensitivity
/set min_messages_before_topic_change 8
```

### Topic Management

```bash
# Rename ongoing topics
/topics rename

# Compress current topic
/topics compress

# View topic statistics
/topics stats
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
/set rag_relevance_threshold 0.7 # Minimum relevance score
```

### Smart Fallback

When RAG doesn't find relevant results, it automatically searches the web (if enabled):

```bash
/set rag-auto true   # Enable RAG
/set web-auto true   # Enable web fallback
# Now queries check your docs first, then web if needed
```

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