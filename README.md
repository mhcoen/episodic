# Episodic 🧠

A conversational memory system that creates persistent, navigable conversations with Large Language Models (LLMs). Episodic automatically organizes conversations into topics, manages context windows, and provides tools for searching both local knowledge and the web.

- Episodic is unique in offering a straightforward *simple* mode, which lets users chat and intelligently search the web. This mode hides Episodic's complexity and configuration but lets users take advantage of its advanced capabilities. Simple mode handles all details of enabling chatting and searching the web. It automatically organizes conversations by topic, detects subject changes, and keeps conversational records and summaries. Conversations are easily accessible in common *markdown files*, which Episodic can both read and write.

- Episodic has also has an *advanced* mode, which is well suited to developers, academics, researchers, and anyone interested in experimenting with LLM-based applications. This unlocks a comprehensive suite of commands for multi-model orchestration, RAG, semantic detection models, prompt engineering, performance benchmarking, cost analyses, and fine-grained system control.

Users can happily remain entirely within the *simple* mode, which makes use of free systems to provide advanced capabilities. Researchers can use *advanced* mode to quickly gain access to rich computational tools and models of conversation.

## Motivation
I originally wrote this to fill a gap I couldn’t find addressed elsewhere. It has since become my preferred daily interface and framework for both routine LLM use and developing new capabilities based on them.

## ✨ Features

- **🤖 Universal LLM Interface** - Works with OpenAI, Anthropic, Google, Ollama, 20+ providers, and custom local models
- **🎭 Muse Mode** - Perplexity-like web search with many providers (e.g., DuckDuckGo, Google, Brave, Searx)
- **🎙️ Voice Mode** - Hands-free speech input and text-to-speech output
- **🔄 Local & Cloud Flexibility** - Easily switch between local (free, private) and cloud-based operation
- **🧠 Intelligent Topic Detection** - Neural segmentation validated on academic benchmarks, with configurable granularity
- **🔄 Topic Reactivation** - Seamlessly resume previous topics with full context restoration
- **📓 Markdown Import/Export** - Save and resume conversations anytime
- **📎 File References (@file)** - Attach local files directly in chat messages
- **📚 Knowledge Base (RAG)** - Index documents and search them during chats
- **💰 Cost Tracking** - Real-time token usage and costs across all providers
- **🎨 Rich CLI** - Streaming responses, theme-based colors, tab completion

 If you are here regarding the paper *When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation* ([arXiv:2512.17083](https://arxiv.org/abs/2512.17083)), see the [`paper/`](paper/) directory.

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
ollama pull phi4
```

### First Conversation

```bash
# Start Episodic (database created automatically on first run)
python -m episodic
```

Episodic automatically configures itself based on available providers:
- **With Ollama**: Uses phi4 for background tasks (free and local)
- **With OpenAI**: Uses GPT-4o-mini by default for chat, GPT-3.5-Turbo-Instruct for analysis
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
# Interface Modes
/simple          # Switch to simple mode (hides advanced features)
/advanced        # Switch to advanced mode (all features available)

# Core Commands  
/topics          # See how your conversation is organized
/save            # Save current topic to markdown
/load file.md    # Load a markdown conversation
/files           # List markdown files in directory (alias: /ls)

# Knowledge & Search
/search query    # Search your indexed documents (alias: /s) 
/index file      # Add file to knowledge base (alias: /i)
/muse            # Switch to Perplexity-like web search mode
/web             # Show current web search provider

# Voice Mode
/voice           # Toggle voice mode on/off
/voice status    # Show voice mode status and providers
/voice info      # Show audio devices and test microphone

# Customization
/style           # Set global response style (concise/standard/comprehensive/custom)
/format          # Set global response format (paragraph/bulleted/mixed/academic)
/theme           # Change color theme for the interface

# Memory & Context
/memory          # Show memory system status
/forget          # Clear memory of specific topics
/new             # Start a new conversation branch
/clear           # Clear current conversation context

# File References
@document.txt        # Include text file in message
@paper.pdf           # Include PDF (extracts text)
@image.png           # Include image for vision models

# Analysis
/critique        # Have another LLM critique the last response

# Configuration
/model           # Show current AI models
/config          # Manage configuration settings
/migrate         # Migrate database to latest version

# Help
/help            # See all commands
/help <query>    # Search documentation (e.g., /help How do I use muse mode?)

# Tab completion is enabled by default - press Tab after typing:
/mo<Tab>         # Completes to /model
/set <Tab>       # Shows all configuration parameters
/model chat <Tab> # Shows available models
```

## 📖 Documentation

- **[Installation](docs/installation.md)** - Setup instructions
- **[User Guide](USER_GUIDE.md)** - Comprehensive guide
- **[Features](docs/features.md)** - Detailed feature documentation
- **[CLI Reference](docs/cli-reference.md)** - All commands
- **[Configuration](docs/configuration.md)** - Settings and options

## 🎯 Use Cases

### 🎙️ Voice Mode - Hands-Free Conversation
Voice mode enables speech input and text-to-speech output with wake word detection for hands-free use.

**Provider Options:**
- STT (Speech-to-Text): Local Whisper (free), OpenAI Whisper API, Deepgram
- TTS (Text-to-Speech): Local Piper (free), OpenAI TTS, ElevenLabs
- Wake Word: Porcupine (local, low-latency), Local Whisper

See the [User Guide](USER_GUIDE.md) for voice mode setup and configuration.

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

### 🚀 GPT-5 Advanced Features
GPT-5 introduces unique controls for output generation:

**Verbosity Control** - Adjust response length and detail:
```text
> /set gpt.verbosity low      # Concise answers, code generation
> /set gpt.verbosity medium   # Standard responses (default)
> /set gpt.verbosity high     # Detailed explanations, analysis
```

**Reasoning Effort** - Control reasoning depth:
```text
> /set gpt.reasoning-effort minimal  # Fastest responses
> /set gpt.reasoning-effort low      # Quick with good quality
> /set gpt.reasoning-effort medium   # Balanced (default)
> /set gpt.reasoning-effort high     # Thorough reasoning
```

### 🧩 Multi-Model Workflows
Use different models for different tasks to optimize performance and cost:

```text
> /model
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

# Change individual models
> /model chat gpt-5
> /model detection custom/topic-boundary-distilbert
> /model compression ollama/phi4
> /model critic claude-opus-4-5-20251101
```

**Custom Models**: You can add your own fine-tuned local models for any purpose—topic detection, domain-specific chat, specialized summarization, etc. See [Model Configuration](docs/models-configuration.md#custom-local-models) for setup.

### 💾 Long Conversation Management
Episodic automatically manages long conversations by detecting topic changes and compressing old topics:

```text
> /set topic-auto true
> /set comp-auto true
> /set show-topics true  # See topic evolution

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

### 🔄 Topic Reactivation - Resume Any Topic
When you return to a previously discussed topic, Episodic automatically detects this and restores that topic's context—excluding unrelated conversations:

```text
> Help me debug this Python IndexError
📌 New topic: python-debugging
🤖 Let me help you with that IndexError...

[... discussion about Python ...]

> Let's talk about coffee brewing
📌 New topic: coffee-brewing
🤖 Great topic! For pour-over, the ideal ratio is...

[... discussion about coffee ...]

> Back to that Python error - what was the fix?
🔄 Resuming topic: python-debugging
🤖 Right, for that IndexError we discussed checking the list bounds...
```

**Key guarantee:** When you resume Python, the coffee conversation is completely excluded from context—no confusion, no bleed-through.

**Disambiguation:** If your message could match multiple topics (e.g., "more about Java" when you've discussed both Java programming and Java coffee), Episodic shows options:
```text
I found multiple matching topics:
[1] java-programming (12 turns ago) - "How do I use Java streams?"
[2] java-coffee (45 turns ago) - "Best Java coffee beans?"
Which topic?
```

Enable with `/set enable_topic_reactivation true`. See the [Topic Reactivation Guide](docs/user_guide_topic_reactivation.md) for details.

### 📝 Save and Resume Conversations
Export conversations to markdown for sharing, backup, or continuing later:

```text
> Tell me about the history of computing
🤖 The history of computing spans several millennia...

> /save
✅ Conversation saved to: exports/history-of-computing-2025-10-15.md

# Later, or on another machine:
> /files exports
📁 Markdown files in exports
📄 history-of-computing-2025-10-15.md
   Size: 3.2 KB • Modified: 2 hours ago
   Preview: History of Computing

> /load exports/history-of-computing-2025-10-15.md
✅ Conversation loaded successfully!

> What about quantum computing?
🤖 Building on our discussion of computing history, quantum computing represents...

# Export specific topics or entire conversations
> /topics
[1] ✓ History of Computing
[2] ✓ Programming Languages
[3] ○ Quantum Computing (ongoing)

> /save 1-2 computing-basics.md  # Export topics 1 and 2
> /save all full-conversation.md  # Export everything
```

### 🏠 Offline Usage
Run completely offline with local models:

```text
# Switch to local mode (sets all models to local, disables online features)
> /mode local
🏠 Switched to local mode

> Explain how neural networks learn
# Works completely offline with local models

# Switch back to cloud mode when needed
> /mode cloud
☁️ Switched to cloud mode
```

## 🔧 Configuration

Episodic is highly configurable. While many settings can be changed interactively with the `/set` command, you can set your defaults by creating a personal configuration file.

1. Copy `episodic/config_template.json` to `~/.episodic/config.json`.
2. Edit `~/.episodic/config.json` to set your preferences, such as API keys or default models.

Common settings that can be changed via the CLI:

```bash
/set stream-responses true    # Enable response streaming
/set comp-auto true           # Automatic topic compression
/set topic-auto true          # Automatic topic detection
/set show-cost true           # Display token costs
/set debug true               # Enable debug output
/style comprehensive          # Set detailed response style globally
/format academic              # Use academic format for all responses
```

See the [Configuration Documentation](docs/configuration.md) for all configuration options.

### Voice Mode Setup

Voice mode requires a free Picovoice access key for wake word detection.

**Get Your Free Access Key:**
1. Go to https://console.picovoice.ai/
2. Create a free account
3. Copy your Access Key from the dashboard

**Configure the Access Key:**
```bash
# Option 1: Environment variable
export PICOVOICE_ACCESS_KEY="your_access_key_here"

# Option 2: In Episodic
> /set porcupine-access-key your_access_key_here
```

**Voice Mode Settings:**
```bash
/set voice-wake-word computer          # Wake word (computer, jarvis, alexa, etc.)
/set voice-wake-word-sensitivity 0.5   # Detection sensitivity (0.0-1.0)
/set voice-idle-timeout 60             # Seconds before entering idle mode (0 = never)
/set voice-wake-word-enabled true      # Enable/disable wake word feature
```

The wake word detection runs locally with minimal CPU usage (~1-2%) and works completely offline after initial setup.

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
──────────────────────────────────────────────────────────────────────
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
Chat:        gpt-5
Detection:   ollama/phi4
Compression: ollama/phi4
Synthesis:   ollama/phi4
```

Each context serves a specific purpose:
- **Chat**: Main conversation with the user
- **Detection**: Identifying topic changes
- **Compression**: Summarizing conversation branches
- **Synthesis**: Web search result synthesis
- **Critic**: Analyzing and critiquing responses (`/critique` command)

## 🏗️ Architecture

Episodic uses a modular architecture:

- **Conversation DAG**: Messages stored as nodes in a directed acyclic graph
- **Topic Detection**: Pluggable strategy framework with multiple approaches:
  - *Neural detection*: Fine-tuned DistilBERT model (~80% W-F1 on benchmarks)
  - *Embedding-based*: Dual-window with adaptive thresholds
  - *Granularity control*: Fine/medium/coarse segmentation for different use cases
- **Context Recovery**: Topic-isolated context assembly with configurable modes (ancestry/topic_local/hybrid)
- **RAG System**: Vector database using ChromaDB for document similarity search
- **Web Search**: Pluggable provider system (DuckDuckGo, Google, Bing, Brave, Searx)

## 🧪 Testing

This project uses `pytest` for testing. To run the full test suite, navigate to the root directory and run:

```bash
pytest
```

## 🔬 Experimental Features

Episodic includes several experimental features for exploration:

- **Advanced Topic Strategies**: CUSUM, Delta-embedding, Speech-act, and Time-aware detection
- **Calibration Framework**: Domain-specific threshold tuning for optimal segmentation
- **Alternative Embeddings**: Pluggable embedding providers for different use cases

See the [User Guide's section on Experimental Features](USER_GUIDE.md#experimental-features) for details.

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

> Latest breakthroughs in machine learning 2025
🔍 Searching web for: latest breakthroughs machine learning 2025
📚 Found 12 relevant sources
✨ Based on recent developments, here are the major ML breakthroughs in 2025:

1. **OpenAI's GPT-5 with Configurable Reasoning** - Advanced reasoning with verbosity and effort controls...
2. **Google's Gemini 2.5 Pro** - Extended context windows up to 2M tokens...
3. **Meta's Llama 4 405B** - Open-source model rivaling proprietary systems...

📄 Sources: Nature AI, OpenAI Blog, Google Research, Meta AI...
```

### Topic Management & Organization
```bash
> /topics
📚 Conversation Topics
══════════════════════════════════════════════════════════════
📌 machine-learning-breakthroughs-2025 (ongoing)
   Started: 2025-10-15 | Messages: 6 | Model: gpt-5

📦 quantum-computing-basics (compressed)  
   Started: a1 | Ended: d3 | Messages: 12 | Model: gpt-5
   💾 Compressed to 420 tokens (95% reduction)

📑 initial-conversation
   Started: 9x | Ended: a0 | Messages: 5 | Model: gpt-3.5-turbo

💰 Total cost: $0.12 | Context usage: 28% (2,847/10,000 tokens)
```

### Research Assistant with RAG
```bash
> /rag on
✅ RAG (knowledge base) enabled

> /index research_papers/quantum_computing_2025.pdf
📄 Indexed: quantum_computing_2025.pdf (47 chunks)

> /muse
🎭 Muse mode active

> /set rag-auto true
> /set web-auto true

> How do the latest quantum error correction methods compare to existing approaches?
📚 Using knowledge base: quantum_computing_2025.pdf
🔍 Also searching web for recent developments...

✨ Based on your research paper and current developments:

**Your Paper's Findings:**
- Surface codes show 99.9% fidelity in simulations...
- Topological qubits demonstrate improved stability...

**Latest Web Research (2025):**
- IBM's new error correction protocols achieve 99.95% fidelity...
- Google's logical qubit demonstrations show promise...

The latest methods build directly on the foundations you documented, with notable improvements in...

📄 Sources: quantum_computing_2025.pdf, IBM Research, Nature Physics
```

### Multi-Model Configuration
```bash
> /model list
🤖 Available Models by Provider
══════════════════════════════════════════════════════════════
OpenAI:
  • gpt-5                     Latest model with verbosity control
  • gpt-4o                    Previous generation model
  • gpt-4o-mini               Fast and cost-effective  
  • gpt-3.5-turbo             Legacy but reliable

Anthropic:
  • claude-opus-4-1-20250805   Latest Opus 4.1 model
  • claude-sonnet-4-5-20250929 Latest Sonnet 4.5 model
  • claude-3.5-sonnet-20241022 Fast and efficient

Local (Ollama):
  • llama3:8b                 Meta's open model
  • mistral:7b                Efficient reasoning

> /mset
⚙️ Model Parameters Across Contexts
──────────────────────────────────────────────────────────────
Context      Model               Temperature  Max Tokens  Cost/1K
──────────────────────────────────────────────────────────────
Chat         gpt-5               1.0         None        $0.011
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
