# Episodic 🧠

A powerful conversational memory system for AI that creates persistent, searchable, and intelligently organized discussions. Built for everyone from casual users to AI researchers.

## What Makes Episodic Different

Unlike traditional chat interfaces, Episodic treats conversations as **living documents** that evolve over time:

- 📊 **Intelligent Organization** - Automatically detects topic changes and segments conversations
- 🔍 **Dual Search** - Query both your personal knowledge base and the web simultaneously  
- 🧩 **Multi-Model Orchestration** - Use different models for different tasks (GPT-4 for chat, Llama for analysis)
- 💾 **Conversation Compression** - Manages infinite-length discussions through intelligent summarization
- 🌐 **20+ AI Providers** - Works with OpenAI, Anthropic, Google, Ollama, Hugging Face, and more

## Quick Start

```bash
# Install
git clone https://github.com/mhcoen/episodic.git
cd episodic
pip install -e .

# Configure (choose one)
export OPENAI_API_KEY="sk-..."        # Recommended
export HUGGINGFACE_API_KEY="hf_..."   # Free tier
# Or use Ollama for local models

# Start
python -m episodic
```

## Two Modes, One Powerful System

### Simple Mode - Streamlined Interface (10 Commands)
Perfect for daily use, writing, research, and focused work:

```text
> /simple
✨ Simple Mode - Essential commands for fluid conversations

Commands:
💬 /chat, /muse        - Switch between normal and web-enhanced chat
📝 /new, /save, /load  - Manage conversation sessions  
📁 /files              - Browse saved conversations
✨ /style, /format     - Customize responses
⚙️  /help, /exit       - Get help or leave

> What are the key differences between transformers and RNNs?
[Concise, focused response with automatic topic tracking]
```

### Advanced Mode - Full Control (50+ Commands)
For power users, developers, and AI researchers:

```text
> /advanced
🔓 Advanced Mode - Complete control over your AI ecosystem

Additional capabilities:
🧬 Multi-model orchestration    - Different models for different tasks
📊 Topic analysis & compression - Manage infinite conversations
🎯 Fine-grained parameters      - Control every aspect
🔍 RAG + Web hybrid search     - Your docs + current info
📈 Performance benchmarking     - Optimize your workflows
🌳 Conversation visualization   - See your discussion structure

> /model chat gpt-4o
> /model detection ollama/llama3:instruct
> /set show_topics true
> /set auto_compress true
> /mset chat.temperature 0.7
```

## Core Features in Action

### 🎭 Muse Mode - Advanced Web Intelligence
Unlike simple web searches, Muse mode provides Perplexity-style research with source synthesis:

```text
> /muse
✨ Muse mode activated - Web-enhanced intelligence

> Compare the latest quantum computing breakthroughs from IBM, Google, and IonQ
🔍 Searching across multiple sources...
📊 Synthesizing information from 12 technical sources...

**Comprehensive Analysis:**

1. **IBM Quantum Network** (Q3 2024)
   - 433-qubit Osprey processor in production
   - Quantum volume reached 512
   - Error rates: 0.1% for two-qubit gates
   
2. **Google Quantum AI** (Latest: October 2024)
   - Demonstrated error correction below threshold
   - 70-qubit Sycamore 2 chip
   - Focus on fault-tolerant computing

3. **IonQ Forte** (November 2024)
   - 32-qubit trapped ion system
   - 99.8% single-qubit gate fidelity
   - Commercial availability on AWS Braket

[Sources: arxiv:2410.xxxxx, Nature Quantum Info, IEEE Spectrum, ...]
```

### 📚 Hybrid Knowledge System (RAG + Web)
Combines your personal documents with live web search:

```text
> /index ~/research/papers/
📚 Indexed 47 documents (2,341 chunks)

> /set rag-auto true
> /set web-auto true

> How does my proposed architecture compare to recent transformer variants?
📄 Searching your documents...
🌐 Searching recent research...

**Analysis combining your work with current research:**

Your proposed "Hierarchical Attention Networks" (from thesis_chapter4.pdf) shares 
similarities with recent work:

1. **From your documents:**
   - Multi-scale attention mechanism (page 43)
   - O(n log n) complexity vs O(n²) for standard transformers
   
2. **Recent developments (2024):**
   - "Mamba" architecture achieves similar complexity reduction
   - "RetNet" uses retention mechanism vs your hierarchical approach
   
3. **Comparative advantages of your approach:**
   - Better interpretability through explicit hierarchy
   - Lower memory footprint for long sequences...
```

### 🔄 Intelligent Context Management
Episodic automatically manages conversation flow:

```text
> /set show_topics true
> /set show_drift true

> Can you explain Docker networking?
📌 New topic detected: "docker-networking-fundamentals"

🤖 Docker networking allows containers to communicate...

> What about Kubernetes networking?
📈 Moderate drift (0.72) - Related topic continuation

🤖 Kubernetes networking builds on container concepts...

> How do I train a neural network?
🔄 High topic shift (0.91) - New topic detected
📌 Previous topic "docker-kubernetes-networking" completed (14 messages)
📌 New topic: "neural-network-training"

🤖 Training a neural network involves several key steps...
```

### 🧩 Multi-Model Orchestration
Use specialized models for different tasks:

```text
> /model list
Current model configuration:
  Chat:        openai/gpt-4o ($0.01/1K in, $0.03/1K out)
  Detection:   ollama/llama3:instruct (local, free)
  Compression: anthropic/claude-instant ($0.0008/1K)
  Synthesis:   openai/gpt-3.5-turbo ($0.0005/1K)

> /compression stats
📊 Compression Statistics:
  - Topics compressed: 8
  - Original tokens: 45,632
  - Compressed tokens: 4,921 (89.2% reduction)
  - Estimated savings: $1.37
  - Compression quality: 94.3% (information retention)
```

## Real-World Workflows

### Research & Writing
```bash
# Start a research session
/muse
/set muse-detail maximum
/style comprehensive

# Index reference materials  
/index ~/documents/sources/
/set rag-threshold 0.6

# Export for writing
/out literature-review
/out --format academic paper.md
```

### Software Development
```bash
# Configure for coding
/style concise
/format mixed
/model chat anthropic/claude-3-sonnet

# Save checkpoints
/save pre-refactor
# ... make changes ...
/save post-refactor

# Search your codebase context
/index ./src
/search authentication flow
```

### Learning & Exploration
```bash
# Track learning progress
/topics list
Topics detected:
1. "python-basics" (completed, 45 messages)
2. "web-frameworks" (completed, 67 messages) 
3. "database-design" (ongoing, 23 messages)

# Review compressed knowledge
/topics show 1
Summary: Covered variables, functions, classes, error handling...

# Continue from any point
/head <node-id>
```

## Advanced Configuration

```bash
# Performance optimization
/set stream_responses true
/set stream_rate 30
/set cache_embeddings true
/benchmark on

# Topic detection fine-tuning
/set min_messages_before_topic_change 5
/set topic_similarity_threshold 0.7
/set use_sliding_window_detection true

# Cost management
/cost                    # Current session costs
/set show_cost true      # Real-time cost display
/model compression gpt-3.5-turbo-instruct  # Cheaper compression
```

## Why Episodic?

**For Casual Users:**
- Simple mode provides a clean, distraction-free interface
- Conversations are automatically organized and saved
- Easy to export and share as markdown

**For Power Users:**
- Complete control over AI orchestration
- Sophisticated search across documents and web
- Extensible architecture for custom workflows

**For Developers:**
- Clean, modular codebase (no file >600 lines)
- Easy to add custom commands and providers
- Comprehensive debugging and benchmarking tools

**For Researchers:**
- Track and analyze conversation patterns
- Export data for analysis
- Compare model behaviors across providers

## Installation & Requirements

- Python 3.8+
- 2GB disk space for vector database
- API key for at least one provider (or Ollama for local)

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

## Documentation

- [User Guide](docs/user-guide.md) - Complete feature documentation
- [CLI Reference](docs/cli-reference.md) - All commands with examples
- [Architecture](docs/architecture.md) - System design and internals
- [API Integration](docs/api-integration.md) - Adding new providers

## Contributing

Episodic is designed to be extended. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE)

---

*Episodic: Where every conversation becomes knowledge.*