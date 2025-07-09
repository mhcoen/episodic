# Episodic 🧠

A conversational memory system that creates persistent, navigable conversations with Large Language Models (LLMs). Episodic automatically organizes conversations into topics, manages context windows, and provides tools for searching both local knowledge and the web.

## Motivation
I originally wrote this to fill a gap I couldn’t find addressed elsewhere. It has since become my preferred daily interface and framework for both routine LLM use and developing new capabilities based on them.

## ✨ Features

- **🤖 Universal LLM Interface**: Chat with OpenAI, Anthropic, Google, Ollama, and more through one interface
- **🗄️ Persistent Memory**: All conversations stored in a local SQLite database
- **🎯 Automatic Topic Detection**: Intelligently segments conversations into semantic topics
- **📊 Context Management**: Compresses old topics to stay within LLM context limits
- **💰 Cost & Usage Tracking**: Real-time tracking of tokens used and costs across all LLM providers
- **🌐 Web Search**: Search the web and get AI-synthesized summaries of results
- **📚 Knowledge Base (RAG)**: Index and search your documents during conversations
- **🔄 Smart RAG Fallback**: Automatically searches web when your documents lack relevant info
- **🎭 Muse Mode**: Perplexity-like conversational web search with AI-synthesized answers
- **🎨 Rich CLI**: Streaming responses, colored output, text wrapping

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mhcoen/episodic.git
cd episodic

# Install in development mode
pip install -e .
```

### First Conversation

```bash
# Start Episodic (database created automatically on first run)
python -m episodic
```

```text
# Just start chatting!
> What's the capital of France?
🤖 The capital of France is Paris.

> Tell me about its history
🤖 Paris has a rich history dating back over 2,000 years...

# Enable web search mode for current information
> /muse on
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
/search query    # Search your indexed documents  
/websearch query # Search the web
/muse            # Enable Perplexity-like web search mode
/model list      # View available AI models
/help            # See all commands
/help <query>    # Search documentation (e.g., /help How do I use muse mode?)
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
/set rag_relevance_threshold 0.7  # Adjust sensitivity (0.0-1.0)
```

### 🧩 Multi-Model Workflows
Use different models for different tasks to optimize performance and cost:

```text
# Use GPT-4 for complex reasoning
> /model chat gpt-4o

# Use fast local model for topic detection  
> /model detection ollama/llama3

# Use cheap model for compression
> /model compression gpt-3.5-turbo

# Configure model parameters
> /mset chat.temperature 0.7
> /mset detection.temperature 0  # Deterministic topic detection
> /mset compression.max_tokens 500

> Explain the halting problem
🤖 [GPT-4 provides detailed explanation while Llama3 manages topics]
```

### 💾 Long Conversation Management
Episodic automatically manages long conversations by detecting topic changes and compressing old topics:

```text
> /set topic-auto true
> /set comp-auto true
> /set show_topics true  # See topic evolution

> Let's discuss machine learning fundamentals
📌 New topic: machine-learning-fundamentals

# ... extensive discussion ...

> Now I want to understand transformers in detail
🔄 Topic changed → Compressing previous topic
📌 New topic: transformer-architecture
💾 Context usage: 42% (previous topic compressed to 500 tokens)
```

### 🏠 Offline Usage
Run completely offline with local models:

```text
# Set all contexts to use local models
> /model chat ollama/llama3
> /model detection ollama/llama3  
> /model compression ollama/mistral
> /model synthesis ollama/llama3

# Disable online features
> /rag off
> /websearch off

> Explain how neural networks learn
# Works completely offline with local models
```

## 🔧 Configuration

Episodic is highly configurable. Common settings:

```bash
/set stream_responses true    # Enable response streaming
/set comp-auto true           # Automatic topic compression
/set topic-auto true          # Automatic topic detection
/set show_cost true           # Display token costs
/set debug true               # Enable debug output
```

See the [Configuration Reference](CONFIG_REFERENCE.md) for all configuration options.

### Model Parameters

Fine-tune model behavior across four contexts with `/mset`:

```
⚙️  Model Parameters:
──────────────────────────────────────────────────────────────────────
Parameter            Chat     Detection Compression Synthesis  
──────────────────────────────────────────────────────────────────────
temperature           0.7        0.0        0.3        0.3     
max_tokens           2000        50        500       1500     
top_p                 1.0       0.95        1.0        1.0     
presence_penalty      0.0        0.0        0.0        0.0     
frequency_penalty     0.0        0.0        0.0        0.0     

Use '/mset <context>' to see details for a specific context
Use '/mset <context>.<param> default' to reset to default value

🤖 Current Models:
─────────────────────────────────────────────────────────────
Chat:        gpt-4
Detection:   ollama/llama3  
Compression: gpt-3.5-turbo
Synthesis:   claude-3-haiku
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
- **Web Search**: Pluggable provider system (DuckDuckGo, Google, Bing, Searx)

## 🧪 Experimental Features

Episodic includes several experimental features for exploration:

- **Hybrid Topic Detection**: Combines embedding similarity, keywords, and conversation patterns
- **Boundary Analysis**: LLM-powered detection of exact topic transition points
- **Alternative Embeddings**: Pluggable embedding providers for different use cases

See [Experimental Features](USER_GUIDE.md#experimental-features) for details.

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- **Non-linear Conversations**: Implement branching conversation trees
- **Running Topic Prediction**: Real-time topic detection
- **Additional Providers**: More LLM, embedding, and search providers
- **UI/UX Improvements**: Better visualization and interaction

See [ADAPTIVE_TOPIC_DETECTION_PLAN.md](ADAPTIVE_TOPIC_DETECTION_PLAN.md) for planned improvements.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal formatting and colors
- [NetworkX](https://networkx.org/) - Graph data structures for conversation DAG
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings

## 📸 Screenshots

### Topic Management
```
📚 Recent Topics
═══════════════════════════════════════════════
📌 quantum-computing-basics (ongoing)
   Started: d4 | Messages: 8

📦 machine-learning-fundamentals (compressed)
   Started: a1 | Ended: d3 | Messages: 12

📑 initial-conversation
   Started: 9x | Ended: a0 | Messages: 5
```

### Context Usage
```
Tokens: 1,847 | Cost: $0.0234 USD | Context: 28% full
```

### Web Search Results
```
🔍 Web Search Results for: "latest AI developments"
══════════════════════════════════════════════════
1. ⭐ 9.2 | OpenAI Announces GPT-5 Development
   Recent breakthrough in multimodal AI capabilities...
   🔗 https://example.com/gpt5-announcement

2. ⭐ 8.7 | Google's Gemini Ultra Performance
   Comprehensive benchmark results show...
   🔗 https://example.com/gemini-benchmarks
```

## 👤 Author

**Michael H. Coen**  
Email: mhcoen@gmail.com | mhcoen@alum.mit.edu  
GitHub: [@mhcoen](https://github.com/mhcoen)

---

*Start your persistent AI conversations today with Episodic!*
