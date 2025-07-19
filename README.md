# Episodic 🧠

A conversational memory system that creates persistent, navigable conversations with Large Language Models (LLMs). Episodic automatically organizes conversations into topics, manages context windows, and provides tools for searching both local knowledge and the web.

- Episodic is unique in offering a straightforward *simple* mode, which lets users chat and intelligently search the web. This mode hides Episodic's complexity and configuration but lets users take advantage of its advanced capabilities. Simple mode handles all details of enabling chatting and searching the web. It automatically organizes conversations by topic, detects subject changes, and keeps conversational records and summaries. Conversations are easily accessible in common *markdown files*, which Episodic can both read and write.

- Episodic has also has an *advanced* mode, which is well suited to developers, academics, researchers, and anyone interested in experimenting with LLM-based applications. This unlocks a comprehensive suite of commands for multi-model orchestration, RAG, semantic detection models, prompt engineering, performance benchmarking, cost analyses, and fine-grained system control.

Users can happily remain entirely within the *simple* mode, which makes use of free systems to provide advanced capabilities. Researchers can use *advanced* mode to quickly gain access to rich computational tools and models of conversation.

## Motivation
I originally wrote this to fill a gap I couldn’t find addressed elsewhere. It has since become my preferred daily interface and framework for both routine LLM use and developing new capabilities based on them.

## ✨ Features

- **🤖 Universal LLM Interface** - Works with OpenAI, Anthropic, Google, Ollama, and 20+ providers
- **🎭 Muse Mode** - Perplexity-like web search with many providers (e.g., DuckDuckGo, Google, Brave, Searx)
- **🗄️ Persistent Memory** - Automatic topic detection and context management
- **📓 Markdown Import/Export** - Save and resume conversations anytime
- **📚 Knowledge Base (RAG)** - Index documents and search them during chats
- **💰 Cost Tracking** - Real-time token usage and costs across all providers
- **🎨 Rich CLI** - Streaming responses, theme-based colors, tab completion

## 🚀 Quick Start

📖 **New users: See [QUICK_START.md](QUICK_START.md) for a complete 5-minute setup guide using free services!**

### Installation

```bash
# Clone the repository
git clone https://github.com/mhcoen/episodic.git
cd episodic

# Install in development mode
pip install -e .
```

### Setup (Choose One)

```bash
# Option 1: Free start with Hugging Face (recommended for beginners)
# Get a free token at https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="hf_..."

# Option 2: Use OpenAI (better chat quality, costs money)
# Get a key at https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."

# Option 3: Fully local with Ollama (advanced users)
# Install from https://ollama.com, then:
ollama pull llama3
```

### First Conversation

```bash
# Start Episodic (database created automatically on first run)
python -m episodic
```

Episodic automatically configures itself based on available providers:
- **With Hugging Face**: Uses Falcon-7B-Instruct for background tasks (free tier compatible)
- **With OpenAI**: Uses GPT-4 for chat, GPT-3.5-Turbo-Instruct for analysis
- **With Ollama**: Uses local models for complete privacy

```text
# Just start chatting!
> What's the capital of France?
🤖 The capital of France is Paris.

> Tell me about its history
🤖 Paris has a rich history dating back over 2,000 years...

# Enable web search mode for current information
> /muse
✨ Muse mode activated! I'll search the web to answer your questions.

> What major events are happening in Paris this week?
✨ Based on current information, here are the major events in Paris this week:

1. **Paris Fashion Week** continues through Sunday with shows from...
2. **Olympic Legacy Exhibition** at the Grand Palais featuring...
3. **Night of Museums** - free admission to 120+ museums on Saturday...

# Your conversation is automatically saved and organized into topics!
```

### Essential Commands

```bash
/topics          # See how your conversation is organized
/out             # Save current topic to markdown
/in file.md      # Load a markdown conversation
/files           # List markdown files in directory (alias: /ls)
/search query    # Search your indexed documents (alias: /s) 
/index file      # Add file to knowledge base (alias: /i)
/muse            # Switch to Perplexity-like web search mode
/style           # Set global response style (concise/standard/comprehensive/custom)
/format          # Set global response format (paragraph/bulleted/mixed/academic)
/web             # Show current web search provider
/model           # Show current AI models
/help            # See all commands
/help <query>    # Search documentation (e.g., /help How do I use muse mode?)

# Tab completion is enabled by default - press Tab after typing:
/mo<Tab>         # Completes to /model
/set <Tab>       # Shows all configuration parameters
/model chat <Tab> # Shows available models
```

## 📖 Documentation

- **[Installation](docs/installation.md)** - Setup instructions
- **[User Guide](docs/user-guide.md)** - Comprehensive guide
- **[Features](docs/features.md)** - Detailed feature documentation
- **[CLI Reference](docs/cli-reference.md)** - All commands
- **[Configuration](docs/configuration.md)** - Settings and options

## 🎯 Use Cases

### 🎭 Muse Mode - Conversational Web Search
Muse mode transforms Episodic into a Perplexity-like AI research assistant that searches the web and synthesizes comprehensive answers:

```text
> /muse
✨ Muse mode activated! I'll search the web to answer your questions.

> What are the latest breakthroughs in fusion energy?
🔍 Searching web for: latest breakthroughs fusion energy
📚 Found 8 relevant sources
✨ Based on recent developments, here are the major breakthroughs in fusion energy:

1. **LLNL's Net Energy Gain** (December 2022): The National Ignition Facility achieved 
   fusion ignition with 3.15 MJ output from 2.05 MJ input...

2. **Commonwealth Fusion's SPARC Progress**: Their high-temperature superconducting 
   magnets have demonstrated 20 Tesla field strength...

> How does this compare to ITER's approach?
# Muse mode maintains context for follow-up questions
```

### 📚 Research Assistant
Index your papers and documents, then ask questions that search both your knowledge base and the web:

```text
> /rag on
> /index research_paper.pdf
> /index thesis_chapter3.md
📄 Indexed 2 documents (47 chunks)

> /set rag-auto true  # Auto-search knowledge base
> What are the latest developments in quantum error correction?
📚 Using sources: research_paper.pdf, thesis_chapter3.md
🌐 Also searching web for recent developments...
# Combines your documents with current web information
```

**Smart Fallback**: When RAG is enabled with `rag-auto` and `web-auto` settings, Episodic automatically searches the web if your documents don't contain relevant information (below the relevance threshold). Configure with:
```bash
/set rag-auto true              # Enable automatic RAG search
/set web-auto true              # Enable automatic web fallback
/set rag-threshold 0.7          # Adjust relevance sensitivity (0.0-1.0)
```

### 🧩 Multi-Model Workflows
Use different models for different tasks to optimize performance and cost:

```text
# Use GPT-4 for complex reasoning
> /model chat gpt-4o

# Use instruct models for background tasks
> /model detection huggingface/tiiuae/falcon-7b-instruct
> /model compression gpt-3.5-turbo-instruct
> /model synthesis huggingface/tiiuae/falcon-7b-instruct

# Configure model parameters
> /mset chat.temperature 0.7
> /mset detection.temperature 0  # Deterministic topic detection
> /mset compression.max_tokens 500

> Explain the halting problem
🤖 [GPT-4 provides detailed explanation while Falcon-7B manages topics]
```

### 💾 Long Conversation Management
Episodic automatically manages long conversations by detecting topic changes and compressing old topics:

```text
> /set topic-auto true
> /set comp-auto true
> /set show_topics true  # See topic evolution

> What's the best way to implement retry logic in Python?
📌 New topic: python-retry-patterns

🤖 The most robust approach is using the tenacity library with exponential backoff...

> How do you handle database connection failures?

🤖 For database connections, implement a connection pool with automatic reconnection...

> Is PostgreSQL better than MySQL for high-traffic applications?

🤖 PostgreSQL generally handles complex queries and concurrent writes better...

> What about horizontal scaling with read replicas?

🔄 Topic changed → Compressing previous topic
📌 New topic: database-scaling-strategies
💾 Context usage: 38% (previous topic compressed to 420 tokens)
```

### 📝 Save and Resume Conversations
Export conversations to markdown for sharing, backup, or continuing later:

```text
> Tell me about the history of computing
🤖 The history of computing spans several millennia...

> /out
✅ Conversation saved to: exports/history-of-computing-2025-01-15.md

# Later, or on another machine:
> /ls exports
📁 Markdown files in exports
📄 history-of-computing-2025-01-15.md
   Size: 3.2 KB • Modified: 2 hours ago
   Preview: History of Computing

> /in exports/history-of-computing-2025-01-15.md
✅ Conversation loaded successfully!

> What about quantum computing?
🤖 Building on our discussion of computing history, quantum computing represents...

# Export specific topics or entire conversations
> /topics
[1] ✓ History of Computing
[2] ✓ Programming Languages  
[3] ○ Quantum Computing (ongoing)

> /out 1-2 computing-basics.md  # Export topics 1 and 2
> /out all full-conversation.md  # Export everything
```

### 🏠 Offline Usage
Run completely offline with local models:

```text
# Set all contexts to use local models
> /model chat ollama/llama3
> /model detection ollama/phi3  # Instruct model for detection
> /model compression ollama/mistral  # Instruct model for compression
> /model synthesis ollama/phi3  # Instruct model for synthesis

# Disable online features (stay in chat mode)
> /rag off
> /chat

> Explain how neural networks learn
# Works completely offline with local models
```

## 🔧 Configuration

Episodic is highly configurable. While many settings can be changed interactively with the `/set` command, you can set your defaults by creating a personal configuration file.

1. Copy `episodic/config_template.json` to `~/.episodic/config.json`.
2. Edit `~/.episodic/config.json` to set your preferences, such as API keys or default models.

Common settings that can be changed via the CLI:

```bash
/set stream_responses true    # Enable response streaming
/set comp-auto true           # Automatic topic compression
/set topic-auto true          # Automatic topic detection
/set show_cost true           # Display token costs
/set debug true               # Enable debug output
/style comprehensive          # Set detailed response style globally
/format academic              # Use academic format for all responses
```

See the [Configuration Documentation](docs/configuration.md) for all configuration options.

### Global Response Formatting

Episodic provides unified response style and format controls that work across all modes (chat, RAG-enhanced, and muse synthesis):

```bash
# Response styles control length and detail level
/style concise        # Brief, direct responses (1-2 sentences when possible)
/style standard       # Clear, well-structured responses with appropriate detail  
/style comprehensive  # Thorough, detailed responses with examples and context
/style custom         # Use model-specific max_tokens settings

# Response formats control presentation structure
/format paragraph     # Flowing prose with markdown headers
/format bulleted      # Bullet points and lists for all information
/format mixed         # Mix of paragraphs and bullet points as appropriate  
/format academic      # Formal academic style with proper citations

# These settings apply universally
> /style comprehensive
> /format academic
> What is machine learning?
🤖 [Detailed academic-style response with citations across all modes]
```

The system intelligently adapts prompts based on context - for example, with small RAG contexts it emphasizes using provided sources, while with web search it focuses on synthesis.

### Model Parameters

Fine-tune model behavior across four contexts with `/mset`:

```
⚙️  Model Parameters:
───────────────────��──────────────────────────────────────────────────
Parameter            Chat     Detection Compression Synthesis  
──────────────────────────────────────────────────────────────────────
temperature           0.7        0.0        0.3        0.3     
max_tokens           2000         50        500       1500     
top_p                 1.0       0.95        1.0        1.0     
presence_penalty      0.0        0.0        0.0        0.0     
frequency_penalty     0.0        0.0        0.0        0.0     

Use '/mset <context>' to see details for a specific context
Use '/mset <context>.<param> default' to reset to default value

🤖 Current Models:
─────────────────────────────────────────────────────────────
Chat:        gpt-4
Detection:   huggingface/tiiuae/falcon-7b-instruct  
Compression: gpt-3.5-turbo-instruct
Synthesis:   huggingface/tiiuae/falcon-7b-instruct
```

Each context serves a specific purpose:
- **Chat**: Main conversation with the user
- **Detection**: Identifying topic changes  
- **Compression**: Summarizing conversation branches
- **Synthesis**: Web search result synthesis

## 🏗️ Architecture

Episodic uses a modular architecture:

- **Conversation DAG**: Messages stored as nodes in a directed acyclic graph
- **Topic Detection**: Multiple algorithms including sliding window and hybrid detection
- **RAG System**: Vector database using ChromaDB for document similarity search
- **Web Search**: Pluggable provider system (DuckDuckGo, Google, Bing, Brave, Searx)

## 🧪 Testing

This project uses `pytest` for testing. To run the full test suite, navigate to the root directory and run:

```bash
pytest
```

## 🔬 Experimental Features

Episodic includes several experimental features for exploration:

- **Hybrid Topic Detection**: Combines embedding similarity, keywords, and conversation patterns
- **Boundary Analysis**: LLM-powered detection of exact topic transition points
- **Alternative Embeddings**: Pluggable embedding providers for different use cases

See the [User Guide's section on Experimental Features](docs/user-guide.md#experimental-features) for details.

## 🤝 Contributing

Contributions are welcome! We follow a standard fork-and-pull-request workflow.

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally (`git clone <your-fork-url>`).
3.  Create a new **branch** for your feature or bug fix (`git checkout -b my-new-feature`).
4.  Make your changes and **commit** them with clear messages.
5.  **Push** your changes to your fork (`git push origin my-new-feature`).
6.  Submit a **pull request** to the main `episodic` repository.

Areas of interest:

- **Non-linear Conversations**: Implement branching conversation trees
- **Running Topic Prediction**: Real-time topic detection
- **Additional Providers**: More LLM, embedding, and search providers
- **UI/UX Improvements**: Better visualization and interaction

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal formatting and colors
- [Click](https://click.palletsprojects.com/) - Command line interface utilities
- [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/) - Interactive command line interfaces
- [NetworkX](https://networkx.org/) - Graph data structures for conversation DAG
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface
- [OpenAI Python](https://github.com/openai/openai-python) - OpenAI API client
- [Anthropic Python](https://github.com/anthropics/anthropic-sdk-python) - Anthropic API client
- [Google Generative AI](https://github.com/google/generative-ai-python) - Google AI client
- [ChromaDB](https://www.trychroma.com/) - Vector database for RAG
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework (in experimental features)
- [Plotly](https://plotly.com/python/) - Interactive visualization
- [Flask](https://flask.palletsprojects.com/) - Web framework for visualization server
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - Web scraping
- [aiohttp](https://docs.aiohttp.org/) - Asynchronous HTTP client
- [PyWebView](https://pywebview.flowrl.com/) - Native GUI for web content

## 📸 Examples & Screenshots

### Mode Switching
```bash
# Start in default chat mode
> /chat
💬 Chat mode active - conversation with AI

> Explain machine learning
🤖 Machine learning is a subset of artificial intelligence...

# Switch to muse mode for web-researched answers
> /muse
🎭 Muse mode active - web search synthesis

> Latest breakthroughs in machine learning 2024
🔍 Searching web for: latest breakthroughs machine learning 2024
📚 Found 12 relevant sources
✨ Based on recent developments, here are the major ML breakthroughs in 2024:

1. **OpenAI's GPT-4o with Advanced Reasoning** - Significant improvements in complex problem solving...
2. **Google's Gemini Ultra 1.5** - Extended context windows up to 2M tokens...
3. **Meta's Llama 3.1 405B** - Open-source model rivaling proprietary systems...

📄 Sources: Nature AI, OpenAI Blog, Google Research, Meta AI...
```

### Topic Management & Organization
```bash
> /topics
📚 Conversation Topics
══════════════════════════════════════════════════════════════
📌 machine-learning-breakthroughs-2024 (ongoing)
   Started: d4 | Messages: 6 | Model: gpt-4

📦 quantum-computing-basics (compressed)  
   Started: a1 | Ended: d3 | Messages: 12 | Model: gpt-4
   💾 Compressed to 420 tokens (95% reduction)

📑 initial-conversation
   Started: 9x | Ended: a0 | Messages: 5 | Model: gpt-3.5-turbo

💰 Total cost: $0.12 | Context usage: 28% (2,847/10,000 tokens)
```

### Research Assistant with RAG
```bash
> /rag on
✅ RAG (knowledge base) enabled

> /index research_papers/quantum_computing_2024.pdf
📄 Indexed: quantum_computing_2024.pdf (47 chunks)

> /muse
🎭 Muse mode active

> /set rag-auto true
> /set web-auto true

> How do the latest quantum error correction methods compare to existing approaches?
📚 Using knowledge base: quantum_computing_2024.pdf
🔍 Also searching web for recent developments...

✨ Based on your research paper and current developments:

**Your Paper's Findings:**
- Surface codes show 99.9% fidelity in simulations...
- Topological qubits demonstrate improved stability...

**Latest Web Research (2024):**
- IBM's new error correction protocols achieve 99.95% fidelity...
- Google's logical qubit demonstrations show promise...

The latest methods build directly on the foundations you documented, with notable improvements in...

📄 Sources: quantum_computing_2024.pdf, IBM Research, Nature Physics
```

### Multi-Model Configuration
```bash
> /model list
🤖 Available Models by Provider
══════════════════════════════════════════════════════════════
OpenAI:
  • gpt-4o                    Most capable model
  • gpt-4o-mini               Fast and cost-effective  
  • gpt-3.5-turbo             Legacy but reliable

Anthropic:  
  • claude-3-5-sonnet-20241022 Latest Claude model
  • claude-3-haiku-20240307    Fast and efficient

Local (Ollama):
  • llama3:8b                 Meta's open model
  • mistral:7b                Efficient reasoning

> /mset
⚙️ Model Parameters Across Contexts
──────────────────────────────────────────────────────────────
Context      Model               Temperature  Max Tokens  Cost/1K
──────────────────────────────────────────────────────────────
Chat         gpt-4o              0.7         2000        $0.015
Detection    ollama/llama3       0.0         50          $0.000  
Compression  gpt-3.5-turbo       0.3         500         $0.002
Synthesis    claude-3-haiku      0.3         1500        $0.001

💡 Tip: Use fast local models for detection to reduce costs
```

## 👤 Author

**Michael H. Coen**  
Email: mhcoen@gmail.com | mhcoen@alum.mit.edu  
GitHub: [@mhcoen](https://github.com/mhcoen)

---

*Episodic: AI that remembers the conversation.*
