# Episodic 🧠

A conversational memory system that creates persistent, navigable conversations with Large Language Models (LLMs). Episodic automatically organizes conversations into topics, manages context windows, and provides tools for searching both local knowledge and the web.

## ✨ Features

- **🗄️ Persistent Memory**: All conversations stored in a local SQLite database
- **🎯 Automatic Topic Detection**: Intelligently segments conversations into semantic topics
- **📊 Context Management**: Compresses old topics to stay within LLM context limits
- **📚 Knowledge Base (RAG)**: Index and search your documents during conversations
- **🌐 Web Search**: Search the web for current information without leaving the conversation
- **🤖 Multi-Model Support**: Works with OpenAI, Anthropic, Google, Ollama, and more
- **🎨 Rich CLI**: Streaming responses, colored output, text wrapping

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/episodic.git
cd episodic

# Install in development mode
pip install -e .
```

### First Conversation

```bash
# Start Episodic
python -m episodic

# Initialize the database
> /init

# Just start chatting!
> What's the capital of France?
🤖 The capital of France is Paris.

> Tell me about its history
🤖 Paris has a rich history dating back over 2,000 years...

# Your conversation is automatically saved and organized!
```

### Key Commands

```bash
/topics         # See how your conversation is organized
/search query   # Search your indexed documents
/websearch query # Search the web
/model          # Switch between AI models
/help           # See all available commands
```

## 📖 Documentation

- **[User Guide](USER_GUIDE.md)** - Comprehensive guide to all features
- **[Configuration](docs/Configuration.md)** - Detailed configuration options
- **[Development](docs/Development.md)** - Architecture and contribution guidelines

## 🎯 Use Cases

### Research Assistant
Index your papers and documents, then ask questions that search both your knowledge base and the web:

```bash
> /rag on
> /index research_paper.pdf
> What are the latest developments in quantum computing?
# Searches both your documents and the web
```

### Long Conversation Management
Episodic automatically manages long conversations by detecting topic changes and compressing old topics:

```bash
> Let's talk about machine learning
# ... long discussion ...
> Now I want to discuss climate change
🔄 Topic changed
# Previous topic is compressed, new topic begins
```

### Offline Usage
Use with local models via Ollama:

```bash
> /model ollama/llama3
> How does photosynthesis work?
# Works completely offline with local models
```

## 🔧 Configuration

Episodic is highly configurable. Common settings:

```bash
/set stream true              # Enable response streaming
/set comp-auto true           # Automatic topic compression
/set rag true                 # Enable document search
/set web-enabled true         # Enable web search
```

See the [User Guide](USER_GUIDE.md#configuration) for all configuration options.

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

[MIT License](LICENSE)

## 🙏 Acknowledgments

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
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

---

*Start your persistent AI conversations today with Episodic!*